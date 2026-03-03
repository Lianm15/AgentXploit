import sqlite3

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect("agentxploit.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id VARCHAR(50) NOT NULL,
            sender VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id VARCHAR(50) PRIMARY KEY,
        target_model VARCHAR(50) NOT NULL,
        success_criteria VARCHAR(200) NOT NULL,
        max_attempts INTEGER NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'initialized',
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()