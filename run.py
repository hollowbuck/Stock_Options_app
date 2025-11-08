from app import create_app
import sys
import os

app = create_app()

if __name__ == "__main__":
    # Enable reloader for development (templates will auto-reload)
    # Note: On Windows, you may see socket errors (WinError 10038) after Excel downloads.
    # This is a known non-critical issue with Flask's reloader on Windows and doesn't affect functionality.
    # The error occurs in a background thread after the request completes successfully.
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    try:
        app.run(
            host='0.0.0.0',  # Bind to all interfaces for external access
            port=port,
            debug=True, 
            use_reloader=True, 
            extra_files=None,
            # Use 'stat' reloader on Windows (less sensitive to file system events)
            reloader_type='stat' if sys.platform == 'win32' else 'auto'
        )
    except (OSError, SystemExit) as e:
        # Handle Windows socket errors gracefully (non-critical)
        if hasattr(e, 'winerror') and e.winerror == 10038:
            print(f"\n⚠️  Reloader warning (non-critical): {e}")
            print("   This is a known Windows issue with Flask's reloader.")
            print("   The application continues to work normally.\n")
        elif 'socket' in str(e).lower() and '10038' in str(e):
            print(f"\n⚠️  Reloader warning (non-critical): {e}")
            print("   This is a known Windows issue with Flask's reloader.")
            print("   The application continues to work normally.\n")
        else:
            raise


