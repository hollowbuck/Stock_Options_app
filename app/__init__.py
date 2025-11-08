from flask import Flask
import os
from secrets import token_hex


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    # Configure secret key for session/flash
    app.secret_key = (
        os.environ.get("FLASK_SECRET_KEY")
        or os.environ.get("SECRET_KEY")
        or token_hex(32)
    )
    
    # Template auto-reload configuration (for development)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching
    
    # Disable Jinja template cache to ensure templates reload on every request
    app.jinja_env.auto_reload = True
    app.jinja_env.cache = {}
    
    # Session configuration for multi-user support
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    
    # Initialize Flask-Login
    from app.extensions import login_manager
    login_manager.init_app(app)
    
    # Register blueprints
    from .main import main_bp
    from .auth import auth
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth)

    # Jinja filters
    def comma_format(value):
        try:
            return f"{int(value):,}"
        except Exception:
            return value

    app.jinja_env.filters['comma'] = comma_format
    
    # Context processor to make user available in all templates
    @app.context_processor
    def inject_user():
        from flask_login import current_user
        return dict(current_user=current_user)

    return app


