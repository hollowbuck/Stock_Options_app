# Multi-User Implementation Guide

## 🎯 Overview

This document describes the transformation of the Options Data Processing application from single-user to multi-user, production-ready architecture.

## ✅ Completed Components

### 1. User Model & Database (`app/models.py`)
- **User model** with Flask-Login integration
- **Encrypted credential storage** (API keys & access tokens)
- **User audit logging** for security tracking
- **Session management** table
- **Quotas** (symbols, parallel jobs, file size)

**Features**:
- Username/password authentication
- Encrypted API credentials (Fernet encryption)
- Last active tracking
- User activation/deactivation
- Audit log for all user actions

### 2. Encryption Utilities (`app/secrets.py`)
- **Fernet symmetric encryption** for credentials
- **Key management**: ENV VAR → File → Generate New
- **Automatic validation** on module import
- **Key rotation support** (with re-encryption warning)

**Security**:
- Credentials never stored in plaintext
- Key file has restrictive permissions (0o600)
- Decryption only in memory when needed

### 3. Per-User Workspace Manager (`app/workspace.py`)
- **Isolated directories** per user:
  - `data/users/{user_id}/processed_options/`
  - `data/users/{user_id}/calculated_columns/`
  - `data/users/{user_id}/filtered_options/`
  - `data/users/{user_id}/exports/`
  - `data/users/{user_id}/temp/`
  - `data/users/{user_id}/logs/`

**Features**:
- Automatic directory creation
- Disk usage tracking
- Old file cleanup (configurable retention)
- File listing with metadata
- Workspace deletion

### 4. Per-User Runtime Context (`app/user_runtime.py`)
- **Per-user session pools** (API requests use correct token)
- **Per-user CMP caches** (price data isolation)
- **Per-user rate limiters** (10 API calls/sec per user)
- **Per-user circuit breakers** (failure handling)
- **Processing state isolation** (no cross-user contamination)

**Features**:
- Thread-safe context management
- Idle context cleanup (30-minute timeout)
- Context registry with locks
- Per-user `processing_status` and `filter_status`

### 5. Flask-Login Integration (`app/extensions.py`)
- **LoginManager** configured
- **Session protection** enabled (strong)
- **User loader** function
- **Login redirect** configuration

### 6. Authentication Blueprint (`app/auth/auth.py`)
- `/auth/login` - User login
- `/auth/register` - User registration
- `/auth/logout` - User logout
- `/auth/credentials` - Update API credentials
- `/auth/profile` - User profile & workspace info

### 7. App Initialization (`app/__init__.py`)
- **Flask-Login integrated**
- **Session configuration** for multi-user
- **Auth blueprint registered**
- **Context processor** for templates

### 8. Performance Optimizations (`app/main/db_utils.py`, `app/main/filter.py`)
- **Serialized DB writer thread** with queue
- **Bulk writes** with `method='multi'` and `chunksize=2000`
- **Vectorized DataFrame operations** (no row-by-row loops)
- **WAL mode** for SQLite (better concurrency)
- **Connection pooling** per database path

## 🚧 Components Needing Completion

### 1. Authentication Templates
**Files to create**:
- `app/templates/auth/login.html`
- `app/templates/auth/register.html`
- `app/templates/auth/credentials.html`
- `app/templates/auth/profile.html`

**Required elements**:
- Bootstrap/modern UI
- Flash message display
- Form validation
- CSRF protection
- Remember me checkbox (login)
- Password confirmation (register)

### 2. Base Template Update
**File**: `app/templates/base.html`

**Changes needed**:
- Add user navigation (username, profile link, logout)
- Show login/register links for anonymous users
- Display user quota information
- Add flash message container

### 3. Routes Refactoring (`app/main/routes.py`)
**Critical changes**:

#### A. Add `@login_required` to all routes
```python
from flask_login import login_required, current_user

@main_bp.route('/process', methods=['POST'])
@login_required
def process_options():
    # Get user workspace
    from app.workspace import get_user_workspace
    workspace = get_user_workspace(current_user.id)
    
    # Get user runtime context
    from app.user_runtime import get_user_runtime_context
    api_key, access_token = current_user.get_credentials()
    context = get_user_runtime_context(current_user.id, api_key, access_token)
    
    # Use context.processing_status instead of global
    # Use workspace paths instead of global paths
    ...
```

#### B. Replace global state with per-user state
```python
# OLD:
processing_status['is_running'] = True

# NEW:
context.processing_status['is_running'] = True
```

#### C. Use per-user workspace paths
```python
# OLD:
output_path = 'Processed_Options/Options_Data.db'

# NEW:
workspace = get_user_workspace(current_user.id)
output_path = workspace.get_options_data_path()
```

#### D. Pass user context to background workers
```python
# In thread target function
def run_processing_job(user_id, api_key, access_token, ...):
    workspace = get_user_workspace(user_id)
    context = get_user_runtime_context(user_id, api_key, access_token)
    ...
```

### 4. Yield.py Refactoring (`app/main/yield.py`)
**Critical changes**:

#### A. Accept user context as parameter
```python
def process_symbols_parallel(symbols, expiry, context: UserRuntimeContext, workspace: UserWorkspace):
    # Use context.session_pool instead of global session_pool
    # Use context.cmp_cache instead of global cmp_cache
    # Use context.rate_limiter
    # Use context.circuit_breaker
    # Save output to workspace.get_options_data_path()
```

#### B. Remove global state
```python
# REMOVE these globals:
# session_pool, cmp_cache, symbol_matcher, rate_limiter, circuit_breaker
```

### 5. DB Utils Per-DB Pooling (`app/main/db_utils.py`)
**Status**: Partially complete

**Additional work needed**:
- Ensure connection pooling is keyed by `db_path` (currently pools by path implicitly)
- Add per-DB lock dictionaries if needed
- Test concurrent access to different user DBs

### 6. Rate Limiting Middleware
**File to create**: `app/middleware/rate_limiter.py`

**Features**:
- Per-user API request limits (e.g., 100 requests/minute to web routes)
- Per-user background job limits (max 2 concurrent jobs)
- Redis-backed counters (for multi-worker support)
- Graceful error responses (429 Too Many Requests)

### 7. Redis Integration (Optional but Recommended)
**Purpose**: Session storage & task queue for multi-worker deployments

**Install**:
```bash
pip install redis flask-session celery
```

**Configuration** (in `__init__.py`):
```python
from flask_session import Session
import redis

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.Redis(host='localhost', port=6379, db=0)
Session(app)
```

### 8. Celery Task Queue (Recommended for Production)
**Purpose**: Distributed background job processing

**Setup**:
1. Create `celery_app.py`:
```python
from celery import Celery

celery = Celery(
    'webapp_main',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery.task
def process_symbols_task(user_id, symbols, expiry):
    # Load user credentials
    user = User.get_by_id(user_id)
    api_key, access_token = user.get_credentials()
    
    # Get workspace and context
    workspace = get_user_workspace(user_id)
    context = get_user_runtime_context(user_id, api_key, access_token)
    
    # Run processing
    from app.main.yield import process_symbols_parallel
    result = process_symbols_parallel(symbols, expiry, context, workspace)
    
    return result
```

2. Start Celery worker:
```bash
celery -A celery_app worker --loglevel=info
```

3. Update routes to use Celery:
```python
from celery_app import process_symbols_task

@main_bp.route('/process', methods=['POST'])
@login_required
def process_options():
    task = process_symbols_task.delay(current_user.id, symbols, expiry)
    return jsonify({'task_id': task.id})
```

## 📋 Migration Checklist

### Phase 1: Foundation (✅ Complete)
- [x] User model and database
- [x] Encryption utilities
- [x] Per-user workspaces
- [x] Per-user runtime contexts
- [x] Flask-Login integration
- [x] Auth blueprint
- [x] Performance optimizations

### Phase 2: Templates & UI (🚧 In Progress)
- [ ] Create login.html
- [ ] Create register.html
- [ ] Create credentials.html
- [ ] Create profile.html
- [ ] Update base.html with user navigation
- [ ] Add flash message styling
- [ ] Test authentication flow

### Phase 3: Routes Migration (🚧 Pending)
- [ ] Add @login_required to all routes
- [ ] Replace global processing_status with context-based
- [ ] Replace global filter_status with context-based
- [ ] Update all file paths to use workspace
- [ ] Pass context to background threads
- [ ] Test multi-user isolation

### Phase 4: yield.py Refactoring (🚧 Pending)
- [ ] Accept context and workspace as parameters
- [ ] Remove global session_pool
- [ ] Remove global cmp_cache
- [ ] Remove global rate_limiter
- [ ] Remove global circuit_breaker
- [ ] Use workspace paths for output
- [ ] Test with multiple concurrent users

### Phase 5: Production Features (🚧 Pending)
- [ ] Add rate limiting middleware
- [ ] Implement job quotas
- [ ] Add Redis session storage
- [ ] Add Celery task queue
- [ ] Add monitoring/metrics
- [ ] Add cleanup cron jobs

### Phase 6: Security & Deployment (🚧 Pending)
- [ ] Set up HTTPS (Nginx + Certbot)
- [ ] Configure production Fernet key in environment
- [ ] Set up Gunicorn with multiple workers
- [ ] Configure Redis for sessions
- [ ] Set up log rotation
- [ ] Add resource monitoring
- [ ] Create backup scripts

## 🔐 Security Considerations

### Credential Storage
- API keys/tokens encrypted with Fernet
- Encryption key stored in ENV VAR (production) or secure file
- Never log decrypted credentials
- Credentials only decrypted in memory when needed

### Session Security
- `SESSION_COOKIE_SECURE=True` in production (HTTPS only)
- `SESSION_COOKIE_HTTPONLY=True` (prevent XSS)
- `SESSION_COOKIE_SAMESITE='Lax'` (CSRF protection)
- Session timeout: 1 hour

### File Isolation
- Each user has isolated workspace
- No access to other users' files
- Download endpoints must check user authorization

### Audit Logging
- All authentication events logged
- Credential updates logged
- File operations can be logged
- IP addresses tracked

## 🚀 Deployment

### Development
```bash
# Install dependencies
pip install flask flask-login cryptography

# Run development server
python run.py
```

### Production (Gunicorn + Nginx)

**1. Install dependencies**:
```bash
pip install gunicorn flask flask-login cryptography redis flask-session
```

**2. Set environment variables**:
```bash
export FERNET_ENCRYPTION_KEY="your-secret-key-here"
export FLASK_ENV="production"
export SECRET_KEY="your-flask-secret-key"
```

**3. Run Gunicorn**:
```bash
gunicorn -w 4 -b 127.0.0.1:8000 "app:create_app()"
```

**4. Configure Nginx**:
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

**5. Set up HTTPS** (Certbot):
```bash
sudo certbot --nginx -d yourdomain.com
```

## 📊 Monitoring & Maintenance

### Cleanup Cron Job
```bash
# Add to crontab (daily at 2 AM)
0 2 * * * python -c "from app.workspace import cleanup_all_workspaces; cleanup_all_workspaces(7)"
0 3 * * * python -c "from app.user_runtime import cleanup_idle_contexts; cleanup_idle_contexts(30)"
```

### Resource Monitoring
- Disk usage per user (via workspace.get_disk_usage())
- Active runtime contexts (via get_all_active_contexts())
- Database connections (SQLite pragma stats)
- Celery task queue length (if using Celery)

### Log Rotation
```bash
# /etc/logrotate.d/webapp_main
/path/to/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
}
```

## 🧪 Testing Multi-User Setup

1. **Register two users**:
   - User A, User B with different API credentials

2. **Test isolation**:
   - User A starts processing → check workspace A
   - User B starts processing → check workspace B
   - Verify files don't mix

3. **Test concurrent processing**:
   - Both users run filter simultaneously
   - Check status endpoints return correct user data
   - Verify no cross-contamination

4. **Test rate limiting**:
   - User A hits rate limit
   - Verify User B unaffected

5. **Test logout**:
   - User logs out
   - Verify runtime context cleaned up
   - Verify cannot access protected routes

## 📝 Next Steps

1. **Complete templates** (login, register, credentials, profile)
2. **Update base.html** with user navigation
3. **Refactor routes.py** to use per-user contexts
4. **Refactor yield.py** to accept context parameter
5. **Add rate limiting middleware**
6. **Set up Redis** for production session storage
7. **Optional: Add Celery** for distributed task processing
8. **Deploy with Gunicorn + Nginx + HTTPS**

## 🆘 Support & Issues

### Common Issues

**Issue**: "Encryption validation failed"
- **Solution**: Delete `data/.secret_key` and restart (will generate new key)

**Issue**: "User workspace not created"
- **Solution**: Check file permissions on `data/users/` directory

**Issue**: "Session not persisting"
- **Solution**: Verify `SECRET_KEY` is set and consistent across restarts

**Issue**: "Multiple workers accessing same DB"
- **Solution**: Use Redis for session storage, not filesystem

**Issue**: "Old runtime contexts not cleaning up"
- **Solution**: Run `cleanup_idle_contexts()` periodically (cron or Celery beat)

## 📚 Additional Resources

- [Flask-Login Documentation](https://flask-login.readthedocs.io/)
- [Cryptography (Fernet) Documentation](https://cryptography.io/en/latest/fernet/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Gunicorn Deployment](https://gunicorn.org/)
- [Nginx Reverse Proxy](https://docs.nginx.com/)

---

**Status**: Foundation Complete (Phase 1) ✅  
**Next**: Complete Templates & Update Routes (Phases 2-3) 🚧

