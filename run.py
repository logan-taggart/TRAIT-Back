import os
import signal

from app import create_app

app = create_app()

def handle_shutdown(signum, frame):
    import sys

    print("Shutting down Flask backend gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if __name__ == "__main__":
    # Change the environment to "dev" for development, "prod" for production  
    env = os.environ.get("ENV", "prod")
    port = int(os.environ.get("PORT", 5174))

    if env == "prod":
        from waitress import serve

        print(f"[PROD] Starting backend on http://127.0.0.1:{port} with waitress...")
        serve(app, host="127.0.0.1", port=port)
    else:
        print(f"[DEV] Starting backend on http://127.0.0.1:{port} with Flask...")
        app.run(host="127.0.0.1", port=port, debug=True, use_reloader=True)