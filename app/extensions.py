"""
Flask Extensions Configuration
Initializes Flask-Login and other extensions
"""

from flask_login import LoginManager

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
login_manager.session_protection = 'strong'


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    from app.models import User
    try:
        return User.get_by_id(int(user_id))
    except:
        return None

