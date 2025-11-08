"""
Encryption utilities for storing sensitive credentials
Uses Fernet (symmetric encryption) to encrypt API keys and access tokens
"""

import os
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Optional

# Path to store the encryption key
KEY_FILE = Path('data/.secret_key')

# Cache the encryption key in memory
_FERNET_KEY = None
_FERNET_CIPHER = None


def get_or_create_encryption_key() -> bytes:
    """
    Get or create the Fernet encryption key
    Priority: ENV VAR > File > Generate New
    
    Returns:
        Encryption key (bytes)
    """
    global _FERNET_KEY
    
    if _FERNET_KEY:
        return _FERNET_KEY
    
    # Try environment variable first (production)
    env_key = os.environ.get('FERNET_ENCRYPTION_KEY')
    if env_key:
        _FERNET_KEY = env_key.encode()
        return _FERNET_KEY
    
    # Try loading from file
    if KEY_FILE.exists():
        with open(KEY_FILE, 'rb') as f:
            _FERNET_KEY = f.read().strip()
            return _FERNET_KEY
    
    # Generate new key and save it
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FERNET_KEY = Fernet.generate_key()
    
    with open(KEY_FILE, 'wb') as f:
        f.write(_FERNET_KEY)
    
    # Set restrictive permissions (Unix-like systems)
    try:
        os.chmod(KEY_FILE, 0o600)
    except:
        pass
    
    print(f"⚠️ Generated new encryption key at {KEY_FILE}")
    print(f"⚠️ For production, set FERNET_ENCRYPTION_KEY environment variable")
    
    return _FERNET_KEY


def get_cipher() -> Fernet:
    """Get Fernet cipher instance"""
    global _FERNET_CIPHER
    
    if _FERNET_CIPHER is None:
        key = get_or_create_encryption_key()
        _FERNET_CIPHER = Fernet(key)
    
    return _FERNET_CIPHER


def encrypt_credential(plaintext: Optional[str]) -> Optional[str]:
    """
    Encrypt a credential (API key, access token, etc.)
    
    Args:
        plaintext: Plain text credential
    
    Returns:
        Base64-encoded encrypted credential (string) or None
    """
    if not plaintext:
        return None
    
    try:
        cipher = get_cipher()
        encrypted_bytes = cipher.encrypt(plaintext.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')
    except Exception as e:
        print(f"❌ Encryption error: {e}")
        return None


def decrypt_credential(encrypted: Optional[str]) -> Optional[str]:
    """
    Decrypt a credential
    
    Args:
        encrypted: Base64-encoded encrypted credential
    
    Returns:
        Decrypted plaintext or None
    """
    if not encrypted:
        return None
    
    try:
        cipher = get_cipher()
        decrypted_bytes = cipher.decrypt(encrypted.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        print(f"❌ Decryption error: {e}")
        return None


def rotate_encryption_key(new_key: bytes):
    """
    Rotate encryption key (for security maintenance)
    WARNING: This requires re-encrypting all existing credentials
    
    Args:
        new_key: New Fernet key
    """
    global _FERNET_KEY, _FERNET_CIPHER
    
    old_cipher = get_cipher()
    
    # Update key and cipher
    _FERNET_KEY = new_key
    _FERNET_CIPHER = Fernet(new_key)
    
    # Save new key
    with open(KEY_FILE, 'wb') as f:
        f.write(new_key)
    
    print("✅ Encryption key rotated")
    print("⚠️ You must re-encrypt all user credentials!")


def validate_encryption() -> bool:
    """
    Validate encryption/decryption works correctly
    
    Returns:
        True if validation passes
    """
    try:
        test_string = "test_credential_123"
        encrypted = encrypt_credential(test_string)
        decrypted = decrypt_credential(encrypted)
        return decrypted == test_string
    except Exception as e:
        print(f"❌ Encryption validation failed: {e}")
        return False


# Validate on module import
if __name__ != '__main__':
    if not validate_encryption():
        print("⚠️ WARNING: Encryption validation failed!")

