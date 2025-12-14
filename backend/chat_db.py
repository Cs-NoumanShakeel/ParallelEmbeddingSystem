"""
SQLite Database Manager for Chat History
Provides GPT-like conversation persistence
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid


class ChatDatabase:
    def __init__(self, db_path: str = "./chat_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
        print("[ChatDB] Database initialized")
    
    def create_chat(self, title: str = "New Chat", filename: str = None) -> Dict:
        """Create a new chat session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        chat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT INTO chats (id, title, filename, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, title, filename, now, now)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "id": chat_id,
            "title": title,
            "filename": filename,
            "created_at": now,
            "updated_at": now
        }
    
    def get_all_chats(self) -> List[Dict]:
        """Get all chat sessions, ordered by most recent"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, filename, created_at, updated_at 
            FROM chats 
            ORDER BY updated_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a single chat by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def update_chat(self, chat_id: str, title: str = None, filename: str = None) -> bool:
        """Update chat title or filename"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if title:
            updates.append("title = ?")
            values.append(title)
        if filename is not None:
            updates.append("filename = ?")
            values.append(filename)
        
        if updates:
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(chat_id)
            
            cursor.execute(
                f"UPDATE chats SET {', '.join(updates)} WHERE id = ?",
                values
            )
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete messages first (foreign key)
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        
        return success
    
    def add_message(self, chat_id: str, role: str, content: str) -> Dict:
        """Add a message to a chat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        msg_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, chat_id, role, content, now)
        )
        
        # Update chat's updated_at timestamp
        cursor.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "id": msg_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "created_at": now
        }
    
    def get_messages(self, chat_id: str, limit: int = 50) -> List[Dict]:
        """Get all messages for a chat"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, chat_id, role, content, created_at 
            FROM messages 
            WHERE chat_id = ? 
            ORDER BY created_at ASC
            LIMIT ?
        """, (chat_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_recent_messages(self, chat_id: str, count: int = 10) -> List[Dict]:
        """Get the most recent N messages for context"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, chat_id, role, content, created_at 
            FROM messages 
            WHERE chat_id = ? 
            ORDER BY created_at DESC
            LIMIT ?
        """, (chat_id, count))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        return [dict(row) for row in reversed(rows)]
