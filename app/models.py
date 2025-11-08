"""
User Models for Multi-User Authentication
Stores user credentials and session information
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

# Database path for users
USER_DB_PATH = Path('data/users/users.db')


def init_user_db():
    """Initialize user database with required tables"""
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(USER_DB_PATH))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                encrypted_api_key TEXT,
                encrypted_access_token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                quota_symbols INTEGER DEFAULT 500,
                quota_parallel_jobs INTEGER DEFAULT 2,
                max_file_size_mb INTEGER DEFAULT 100
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
    finally:
        conn.close()


class User(UserMixin):
    """
    User model for authentication and session management
    Implements Flask-Login's UserMixin interface
    """
    
    def __init__(self, user_id: int, username: str, email: Optional[str] = None,
                 encrypted_api_key: Optional[str] = None, 
                 encrypted_access_token: Optional[str] = None,
                 created_at: Optional[datetime] = None,
                 last_active: Optional[datetime] = None,
                 is_active: bool = True,
                 quota_symbols: int = 500,
                 quota_parallel_jobs: int = 2,
                 max_file_size_mb: int = 100):
        self.id = user_id
        self.username = username
        self.email = email
        self.encrypted_api_key = encrypted_api_key
        self.encrypted_access_token = encrypted_access_token
        self.created_at = created_at
        self.last_active = last_active
        self._is_active = is_active
        self.quota_symbols = quota_symbols
        self.quota_parallel_jobs = quota_parallel_jobs
        self.max_file_size_mb = max_file_size_mb
    
    def get_id(self):
        """Required by Flask-Login"""
        return str(self.id)
    
    @property
    def is_active(self):
        """Required by Flask-Login"""
        return self._is_active
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (without sensitive data)"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'is_active': self._is_active,
            'quota_symbols': self.quota_symbols,
            'quota_parallel_jobs': self.quota_parallel_jobs,
            'max_file_size_mb': self.max_file_size_mb
        }
    
    @staticmethod
    def create(username: str, password: str, email: Optional[str] = None,
               api_key: Optional[str] = None, access_token: Optional[str] = None) -> Optional['User']:
        """
        Create a new user
        
        Args:
            username: Username
            password: Plain password (will be hashed)
            email: Email address
            api_key: Zerodha API key (will be encrypted)
            access_token: Zerodha access token (will be encrypted)
        
        Returns:
            User object or None if creation failed
        """
        from app.secrets import encrypt_credential
        
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            password_hash = generate_password_hash(password)
            encrypted_api = encrypt_credential(api_key) if api_key else None
            encrypted_token = encrypt_credential(access_token) if access_token else None
            
            cursor = conn.execute("""
                INSERT INTO users (username, password_hash, email, encrypted_api_key, encrypted_access_token)
                VALUES (?, ?, ?, ?, ?)
            """, (username, password_hash, email, encrypted_api, encrypted_token))
            
            conn.commit()
            user_id = cursor.lastrowid
            
            # Log user creation
            log_user_action(user_id, 'user_created', f'User {username} created')
            
            return User.get_by_id(user_id)
            
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_by_id(user_id: int) -> Optional['User']:
        """Load user by ID"""
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            cursor = conn.execute("""
                SELECT id, username, email, encrypted_api_key, encrypted_access_token,
                       created_at, last_active, is_active, quota_symbols, 
                       quota_parallel_jobs, max_file_size_mb
                FROM users WHERE id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    encrypted_api_key=row[3],
                    encrypted_access_token=row[4],
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    last_active=datetime.fromisoformat(row[6]) if row[6] else None,
                    is_active=bool(row[7]),
                    quota_symbols=row[8],
                    quota_parallel_jobs=row[9],
                    max_file_size_mb=row[10]
                )
            return None
        finally:
            conn.close()
    
    @staticmethod
    def get_by_username(username: str) -> Optional['User']:
        """Load user by username"""
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            cursor = conn.execute("""
                SELECT id, username, email, encrypted_api_key, encrypted_access_token,
                       created_at, last_active, is_active, quota_symbols,
                       quota_parallel_jobs, max_file_size_mb
                FROM users WHERE username = ?
            """, (username,))
            
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row[0],
                    username=row[1],
                    email=row[2],
                    encrypted_api_key=row[3],
                    encrypted_access_token=row[4],
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    last_active=datetime.fromisoformat(row[6]) if row[6] else None,
                    is_active=bool(row[7]),
                    quota_symbols=row[8],
                    quota_parallel_jobs=row[9],
                    max_file_size_mb=row[10]
                )
            return None
        finally:
            conn.close()
    
    @staticmethod
    def verify_password(username: str, password: str) -> Optional['User']:
        """Verify username and password, return User if valid"""
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            cursor = conn.execute("""
                SELECT id, password_hash FROM users WHERE username = ? AND is_active = 1
            """, (username,))
            
            row = cursor.fetchone()
            if row and check_password_hash(row[1], password):
                user = User.get_by_id(row[0])
                if user:
                    user.update_last_active()
                return user
            return None
        finally:
            conn.close()
    
    def update_credentials(self, api_key: str, access_token: str) -> bool:
        """Update user's Zerodha credentials (encrypted)"""
        from app.secrets import encrypt_credential
        
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            encrypted_api = encrypt_credential(api_key)
            encrypted_token = encrypt_credential(access_token)
            
            conn.execute("""
                UPDATE users 
                SET encrypted_api_key = ?, encrypted_access_token = ?
                WHERE id = ?
            """, (encrypted_api, encrypted_token, self.id))
            
            conn.commit()
            
            self.encrypted_api_key = encrypted_api
            self.encrypted_access_token = encrypted_token
            
            log_user_action(self.id, 'credentials_updated', 'API credentials updated')
            return True
        except Exception:
            return False
        finally:
            conn.close()
    
    def get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get decrypted credentials (API key, Access token)"""
        from app.secrets import decrypt_credential
        
        if not self.encrypted_api_key or not self.encrypted_access_token:
            return None, None
        
        try:
            api_key = decrypt_credential(self.encrypted_api_key)
            access_token = decrypt_credential(self.encrypted_access_token)
            return api_key, access_token
        except Exception:
            return None, None
    
    def update_last_active(self):
        """Update last active timestamp"""
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            conn.execute("""
                UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?
            """, (self.id,))
            conn.commit()
            self.last_active = datetime.now()
        finally:
            conn.close()
    
    def deactivate(self):
        """Deactivate user account"""
        conn = sqlite3.connect(str(USER_DB_PATH))
        try:
            conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (self.id,))
            conn.commit()
            self._is_active = False
            log_user_action(self.id, 'user_deactivated', 'User account deactivated')
        finally:
            conn.close()


def log_user_action(user_id: int, action: str, details: Optional[str] = None, 
                    ip_address: Optional[str] = None):
    """Log user action to audit log"""
    conn = sqlite3.connect(str(USER_DB_PATH))
    try:
        conn.execute("""
            INSERT INTO user_audit_log (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, action, details, ip_address))
        conn.commit()
    finally:
        conn.close()


def get_user_audit_log(user_id: int, limit: int = 100) -> list:
    """Get user audit log"""
    conn = sqlite3.connect(str(USER_DB_PATH))
    try:
        cursor = conn.execute("""
            SELECT action, details, timestamp, ip_address
            FROM user_audit_log
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        
        return [{'action': r[0], 'details': r[1], 'timestamp': r[2], 'ip_address': r[3]} 
                for r in cursor.fetchall()]
    finally:
        conn.close()


# Initialize database on module import
init_user_db()

