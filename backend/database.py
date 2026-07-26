import sqlite3
import os
from contextlib import contextmanager

_DB_PATH = os.getenv("DB_PATH", "agentxploit.db")


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with dict-like row access (row[col] / dict(row))."""
    conn = sqlite3.connect(os.environ.get("DB_PATH", _DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_cursor():
    """
    One connection + cursor for a single unit of work.

    Commits automatically when the `with` block exits normally, and always
    closes the connection — callers no longer need to repeat
    conn/cursor/commit/close boilerplate on every query. On an exception the
    connection is closed without committing, same as manual conn.close()
    without conn.commit() would do.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    finally:
        conn.close()


# (table, column, type) triples added after the original schema shipped.
# Each ALTER TABLE is a no-op (caught and ignored) if the column already exists,
# so this list is safe to run against both fresh and pre-existing databases.
_SCHEMA_MIGRATIONS = [
    ("messages", "compliance_score", "REAL DEFAULT NULL"),
    ("messages", "failure_type", "TEXT DEFAULT NULL"),
    ("messages", "technique", "TEXT DEFAULT NULL"),
    ("sessions", "mode", "VARCHAR(20) NOT NULL DEFAULT 'standard'"),
    ("messages", "rationale", "TEXT DEFAULT NULL"),
    ("messages", "score_explanation", "TEXT DEFAULT NULL"),
]


def create_tables() -> None:
    """Create the schema if missing, then apply any pending column migrations."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       VARCHAR(50) NOT NULL,
            sender           VARCHAR(50) NOT NULL,
            content          TEXT NOT NULL,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            compliance_score REAL DEFAULT NULL,
            failure_type     TEXT DEFAULT NULL,
            technique        TEXT DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            session_id     VARCHAR(50) PRIMARY KEY,
            target_model   VARCHAR(50) NOT NULL,
            time_elapsed   FLOAT NOT NULL,
            messages_count INTEGER NOT NULL,
            success        BOOLEAN NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id       VARCHAR(50) PRIMARY KEY,
            target_model     VARCHAR(50) NOT NULL,
            success_criteria VARCHAR(200) NOT NULL,
            max_attempts     INTEGER NOT NULL,
            status           VARCHAR(50) NOT NULL DEFAULT 'initialized',
            started_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode VARCHAR(20) NOT NULL DEFAULT 'standard'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technique_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL,
            attempt_number   INTEGER NOT NULL,
            technique        TEXT NOT NULL,
            compliance_score REAL NOT NULL,
            failure_type     TEXT NOT NULL,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    conn.commit()

    for table, col, typedef in _SCHEMA_MIGRATIONS:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass  # column already exists — nothing to do

    conn.close()


def save_technique_record(session_id: str, record) -> None:
    """Persist one AttemptRecord from AttackStrategyController to technique_history."""
    with db_cursor() as cursor:
        cursor.execute(
            """INSERT INTO technique_history
               (session_id, attempt_number, technique, compliance_score, failure_type)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session_id,
                record.attempt_number,
                record.technique,
                record.compliance_score,
                record.failure_type,
            ),
        )
