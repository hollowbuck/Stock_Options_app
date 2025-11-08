"""
Authentication Blueprint - FIXED
Handles user registration, login, logout, and credential management
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from app.models import User, log_user_action
from app.user_runtime import get_user_runtime_context, remove_user_runtime_context

auth = Blueprint('auth', __name__, url_prefix='/auth')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """User login - FIXED to use correct template"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        if not username or not password:
            flash('Please provide username and password', 'error')
            return render_template('auth.html', title="Login")  # FIXED: use auth.html
        
        user = User.verify_password(username, password)
        
        if user:
            login_user(user, remember=remember)
            log_user_action(user.id, 'login', 'User logged in', request.remote_addr)
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('main.index'))
        else:
            flash('Invalid username or password', 'error')
            log_user_action(0, 'login_failed', f'Failed login attempt for: {username}', request.remote_addr)
    
    return render_template('auth.html', title="Login")  # FIXED: use auth.html


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        email = request.form.get('email', '').strip() or None
        api_key = request.form.get('api_key', '').strip() or None
        access_token = request.form.get('access_token', '').strip() or None
        
        # Validation
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('register.html', title="Register")
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html', title="Register")
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return render_template('register.html', title="Register")
        
        # Create user
        user = User.create(username, password, email, api_key, access_token)
        
        if user:
            flash('Registration successful! Please log in.', 'success')
            log_user_action(user.id, 'register', 'User registered', request.remote_addr)
            return redirect(url_for('auth.login'))
        else:
            flash('Username already exists', 'error')
    
    return render_template('register.html', title="Register")


@auth.route('/logout')
@login_required
def logout():
    """User logout"""
    user_id = current_user.id
    log_user_action(user_id, 'logout', 'User logged out', request.remote_addr)
    
    # Cleanup user runtime context
    remove_user_runtime_context(user_id)
    
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))


@auth.route('/credentials', methods=['GET', 'POST'])
@login_required
def credentials():
    """Update Zerodha API credentials"""
    if request.method == 'POST':
        api_key = request.form.get('api_key', '').strip()
        access_token = request.form.get('access_token', '').strip()
        
        if not api_key or not access_token:
            flash('Both API key and access token are required', 'error')
            return render_template('credentials.html', title="API Credentials")
        
        if current_user.update_credentials(api_key, access_token):
            flash('Credentials updated successfully', 'success')
            log_user_action(current_user.id, 'credentials_updated', 'API credentials updated', request.remote_addr)
            
            # Remove old runtime context to force recreation with new credentials
            remove_user_runtime_context(current_user.id)
            
            return redirect(url_for('main.index'))
        else:
            flash('Failed to update credentials', 'error')
    
    # Check if user has credentials
    api_key, access_token = current_user.get_credentials()
    has_credentials = bool(api_key and access_token)
    
    return render_template('credentials.html', 
                         title="API Credentials",
                         has_credentials=has_credentials)


@auth.route('/profile')
@login_required
def profile():
    """User profile page"""
    from app.workspace import get_user_workspace
    from app.models import get_user_audit_log
    
    # Get workspace info
    workspace = get_user_workspace(current_user.id)
    usage = workspace.get_disk_usage_mb()
    files = workspace.list_files()
    
    # Get audit log
    audit_log = get_user_audit_log(current_user.id, limit=20)
    
    return render_template('profile.html',
                          title="User Profile",
                          usage=usage,
                          files=files,
                          audit_log=audit_log)