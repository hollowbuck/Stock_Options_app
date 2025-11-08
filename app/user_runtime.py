"""
Per-User Runtime Context
Manages session pools, caches, and rate limiters for each user
"""

import threading
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


# Global registry of user runtime contexts
_USER_CONTEXTS: Dict[int, 'UserRuntimeContext'] = {}
_CONTEXT_LOCK = threading.Lock()


class ThreadSafeRateLimiter:
    """Rate limiter for API calls (per user)"""
    
    def __init__(self, calls_per_second: float = 10.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self) -> float:
        """Acquire rate limit slot, returns wait time"""
        import time
        
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                self.last_call_time = time.time()
                return wait_time
            else:
                self.last_call_time = now
                return 0.0


class SmartCircuitBreaker:
    """Circuit breaker for handling API failures (per user)"""
    
    def __init__(self, failure_threshold: int = 8, recovery_timeout: int = 20):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self.lock = threading.Lock()
    
    def record_success(self):
        """Record successful API call"""
        with self.lock:
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
            if self.is_open and self.failure_count == 0:
                self.is_open = False
    
    def record_failure(self, is_rate_limit: bool = False):
        """Record failed API call"""
        if is_rate_limit:
            return  # Don't count rate limits as circuit breaker failures
        
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
    
    def can_proceed(self) -> Tuple[bool, Optional[float]]:
        """
        Check if requests can proceed
        
        Returns:
            (can_proceed, wait_time_seconds)
        """
        with self.lock:
            if not self.is_open:
                return True, None
            
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.is_open = False
                    self.failure_count = max(0, self.failure_count - 2)
                    return True, None
                else:
                    return False, self.recovery_timeout - elapsed
            
            return False, self.recovery_timeout


class OptimizedCMPCache:
    """Cache for CMP (Current Market Price) values (per user)"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache: Dict[str, Tuple[float, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.lock = threading.Lock()
    
    def get(self, symbol: str) -> Optional[float]:
        """Get cached CMP value"""
        with self.lock:
            if symbol in self.cache:
                value, timestamp = self.cache[symbol]
                if datetime.now() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[symbol]
        return None
    
    def set(self, symbol: str, value: float):
        """Set CMP value in cache"""
        with self.lock:
            self.cache[symbol] = (value, datetime.now())
    
    def clear(self):
        """Clear all cached values"""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self.lock:
            now = datetime.now()
            expired = [k for k, (v, t) in self.cache.items() if now - t >= self.ttl]
            for key in expired:
                del self.cache[key]


class OptimizedSessionPool:
    """HTTP session pool for API requests (per user)"""
    
    def __init__(self, api_key: str, access_token: str, pool_size: int = 10):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        self.api_key = api_key
        self.access_token = access_token
        self.pool_size = pool_size
        self.sessions = []
        self.lock = threading.Lock()
        
        # Create session pool
        for _ in range(pool_size):
            session = requests.Session()
            
            # Configure retries
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=20,
                pool_maxsize=50
            )
            
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            # Set default headers
            session.headers.update({
                'X-Kite-Version': '3',
                'Authorization': f'token {api_key}:{access_token}'
            })
            
            self.sessions.append(session)
    
    def get_session(self):
        """Get a session from the pool"""
        with self.lock:
            if self.sessions:
                return self.sessions.pop()
        
        # Create new session if pool is empty
        import requests
        session = requests.Session()
        session.headers.update({
            'X-Kite-Version': '3',
            'Authorization': f'token {self.api_key}:{self.access_token}'
        })
        return session
    
    def return_session(self, session):
        """Return session to the pool"""
        with self.lock:
            if len(self.sessions) < self.pool_size:
                self.sessions.append(session)
            else:
                try:
                    session.close()
                except:
                    pass
    
    def close_all(self):
        """Close all sessions in the pool"""
        with self.lock:
            for session in self.sessions:
                try:
                    session.close()
                except:
                    pass
            self.sessions.clear()


class UserRuntimeContext:
    """
    Runtime context for a user
    Manages all user-specific runtime state (sessions, caches, rate limiters)
    """
    
    def __init__(self, user_id: int, api_key: str, access_token: str):
        self.user_id = user_id
        self.api_key = api_key
        self.access_token = access_token
        self.created_at = datetime.now()
        self.last_access = datetime.now()
        
        # Initialize components
        self.session_pool = OptimizedSessionPool(api_key, access_token)
        self.cmp_cache = OptimizedCMPCache()
        self.rate_limiter = ThreadSafeRateLimiter(calls_per_second=10.0)
        self.circuit_breaker = SmartCircuitBreaker()
        
        # Processing state
        self.processing_status = {
            'is_running': False,
            'progress': 0,
            'status': 'idle',
            'messages': [],
            'start_time': None,
            'end_time': None,
            'result': None
        }
        
        self.filter_status = {
            'is_running': False,
            'progress': 0,
            'status': 'idle',
            'messages': [],
            'start_time': None,
            'output_file': None,
            'summary': None,
            'tables': {}
        }
        
        # Thread locks
        self.processing_lock = threading.Lock()
        self.filter_lock = threading.Lock()
    
    def update_access(self):
        """Update last access time"""
        self.last_access = datetime.now()
    
    def is_idle(self, timeout_minutes: int = 30) -> bool:
        """Check if context has been idle for too long"""
        idle_time = datetime.now() - self.last_access
        return idle_time > timedelta(minutes=timeout_minutes)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.session_pool.close_all()
        except:
            pass
        
        try:
            self.cmp_cache.clear()
        except:
            pass
    
    def to_dict(self) -> dict:
        """Convert context to dictionary"""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_access': self.last_access.isoformat(),
            'processing_status': self.processing_status.copy(),
            'filter_status': {
                **self.filter_status,
                'tables': {}  # Don't include table data
            },
            'circuit_breaker_open': self.circuit_breaker.is_open,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }


def get_user_runtime_context(user_id: int, api_key: str, access_token: str) -> UserRuntimeContext:
    """
    Get or create runtime context for a user
    
    Args:
        user_id: User ID
        api_key: Zerodha API key
        access_token: Zerodha access token
    
    Returns:
        UserRuntimeContext instance
    """
    with _CONTEXT_LOCK:
        if user_id not in _USER_CONTEXTS:
            _USER_CONTEXTS[user_id] = UserRuntimeContext(user_id, api_key, access_token)
        
        context = _USER_CONTEXTS[user_id]
        context.update_access()
        return context


def remove_user_runtime_context(user_id: int):
    """Remove and cleanup runtime context for a user"""
    with _CONTEXT_LOCK:
        if user_id in _USER_CONTEXTS:
            context = _USER_CONTEXTS.pop(user_id)
            context.cleanup()


def cleanup_idle_contexts(timeout_minutes: int = 30):
    """Cleanup idle runtime contexts"""
    with _CONTEXT_LOCK:
        idle_users = [
            user_id for user_id, context in _USER_CONTEXTS.items()
            if context.is_idle(timeout_minutes)
        ]
        
        for user_id in idle_users:
            context = _USER_CONTEXTS.pop(user_id)
            context.cleanup()
        
        return len(idle_users)


def get_all_active_contexts() -> Dict[int, dict]:
    """Get information about all active contexts"""
    with _CONTEXT_LOCK:
        return {
            user_id: context.to_dict()
            for user_id, context in _USER_CONTEXTS.items()
        }

