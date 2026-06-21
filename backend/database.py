import sqlite3
import os

_DB_PATH = os.getenv("DB_PATH", "agentxploit.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(os.environ.get("DB_PATH", _DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def create_tables() -> None:
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
            started_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            mode VARCHAR(20) NOT NULL DEFAULT 'standard',
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

    # Safe migration for databases created before this schema version.
    # Each ALTER TABLE is a no-op if the column already exists (caught and ignored).
    for col, typedef in [
        ("compliance_score", "REAL DEFAULT NULL"),
        ("failure_type",     "TEXT DEFAULT NULL"),
        ("technique",        "TEXT DEFAULT NULL"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE messages ADD COLUMN {col} {typedef}")
            conn.commit()
        except Exception:
            pass

    conn.close()


def save_technique_record(session_id: str, record) -> None:
    """Persist one AttemptRecord from AttackStrategyController to technique_history."""
    conn = get_connection()
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()
