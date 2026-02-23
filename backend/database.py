import sqlite3

def get_connection():
    conn = sqlite3.connect("agentxploit.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(50) PRIMARY KEY,
            target_model VARCHAR(50) NOT NULL,
            success_criteria VARCHAR(200) NOT NULL,
            max_attempts INTEGER NOT NULL
        )
    """)
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
    conn.commit()
    conn.close()