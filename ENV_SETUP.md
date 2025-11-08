# Environment Variables Setup

## Zerodha API Credentials

This application now uses environment variables for secure credential storage instead of config files.

### Required Environment Variables

Set the following environment variables before running the application:

```bash
export ZERODHA_API_KEY="your_api_key_here"
export ZERODHA_ACCESS_TOKEN="your_access_token_here"
```

### Setting Environment Variables

#### Windows (PowerShell)
```powershell
$env:ZERODHA_API_KEY="your_api_key_here"
$env:ZERODHA_ACCESS_TOKEN="your_access_token_here"
```

#### Windows (Command Prompt)
```cmd
set ZERODHA_API_KEY=your_api_key_here
set ZERODHA_ACCESS_TOKEN=your_access_token_here
```

#### Linux/macOS
```bash
export ZERODHA_API_KEY="your_api_key_here"
export ZERODHA_ACCESS_TOKEN="your_access_token_here"
```

### Using .env File (Optional)

For development, you can use a `.env` file with `python-dotenv`:

1. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

2. Create a `.env` file in the project root:
   ```
   ZERODHA_API_KEY=your_api_key_here
   ZERODHA_ACCESS_TOKEN=your_access_token_here
   ```

3. Load it in your application (add to `run.py` or `app/__init__.py`):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Security Notes

- **Never commit `.env` files or `kite_credentials.conf` to version control**
- The `.gitignore` file already excludes these files
- For production, use your hosting platform's environment variable configuration
- Access tokens expire daily - update the `ZERODHA_ACCESS_TOKEN` environment variable as needed

### Verifying Configuration

The application will show a warning on startup if credentials are not configured:
```
⚠️  Warning: Zerodha credentials not configured in environment variables...
```

If you see this message, ensure your environment variables are set correctly.

