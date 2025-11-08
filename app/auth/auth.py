"""
Authentication Blueprint - FIXED
Handles user registration, login, logout, and credential management
"""

from flask import Blueprint, request, redirect, url_for, flash, render_template
from flask_login import login_required, current_user, login_user, logout_user
import os
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

@auth.route('/zerodha/login')
@login_required
def zerodha_login():
    """Redirect user to Zerodha login to get fresh access token"""
    api_key = os.getenv('ZERODHA_API_KEY')
    
    if not api_key:
        flash('Zerodha API key not configured. Please contact administrator.', 'error')
        return redirect(url_for('auth.credentials'))
    
    # Build Zerodha login URL - this redirects user to Zerodha's login page
    # After successful login, Zerodha will redirect back to our callback URL
    callback_url = url_for('auth.zerodha_callback', _external=True)
    zerodha_login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    
    flash('Redirecting to Zerodha for authentication...', 'info')
    return redirect(zerodha_login_url)


@auth.route('/zerodha/callback')
@login_required
def zerodha_callback():
    """Handle callback from Zerodha after user logs in"""
    from app.models import log_user_action
    from app.user_runtime import remove_user_runtime_context
    
    # Zerodha sends back a request_token in the URL parameters
    request_token = request.args.get('request_token')
    
    # Check if there's an error from Zerodha
    error = request.args.get('error')
    if error:
        flash(f'Zerodha authentication failed: {error}', 'error')
        return redirect(url_for('auth.credentials'))
    
    if not request_token:
        flash('Authorization failed. No request token received from Zerodha.', 'error')
        return redirect(url_for('auth.credentials'))
    
    try:
        # Get API credentials from environment
        api_key = os.getenv('ZERODHA_API_KEY')
        api_secret = os.getenv('ZERODHA_API_SECRET')
        
        if not api_key or not api_secret:
            flash('API credentials not properly configured. Please contact administrator.', 'error')
            return redirect(url_for('auth.credentials'))
        
        # Import KiteConnect to exchange request token for access token
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key=api_key)
        
        # Exchange request token for access token
        # This is the crucial step that gets us a valid access token
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Optional: Get user profile to verify the connection
        kite.set_access_token(access_token)
        profile = kite.profile()
        user_name = profile.get('user_name', 'Unknown')
        
        # Update user's credentials in database with the new access token
        if current_user.update_credentials(api_key, access_token):
            flash(f'Successfully connected to Zerodha as {user_name}! Token will be valid for 24 hours.', 'success')
            log_user_action(
                current_user.id, 
                'zerodha_token_refresh', 
                f'Access token refreshed via Zerodha login for user {user_name}', 
                request.remote_addr
            )
            
            # Clear old runtime context to force recreation with new credentials
            remove_user_runtime_context(current_user.id)
            
            return redirect(url_for('main.dashboard'))
        else:
            flash('Failed to save access token to database', 'error')
            return redirect(url_for('auth.credentials'))
            
    except Exception as e:
        error_msg = str(e)
        flash(f'Token generation failed: {error_msg}', 'error')
        log_user_action(
            current_user.id, 
            'zerodha_token_refresh_failed', 
            f'Failed to refresh token: {error_msg}', 
            request.remote_addr
        )
        return redirect(url_for('auth.credentials'))


