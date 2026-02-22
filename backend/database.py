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
            session_id TEXT PRIMARY KEY,
            target_model TEXT NOT NULL,
            success_criteria TEXT NOT NULL,
            max_attempts INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()