"""
Layer 5: run.py

The single entry point: `python run.py`

- Checks required dependencies are importable, with a clear install message
  if something's missing (fails fast, before Flask ever starts).
- Starts the Flask app (app.py) on port 5050.
- Opens the default browser to the setup wizard.
- Prints the URL.

app.py itself prints the headline numbers to the terminal every time a
/run pipeline finishes, so the whole flow -- wizard through to dashboard --
is screen-recordable from this one terminal window.
"""

import importlib
import sys
import threading
import time
import webbrowser

REQUIRED_PACKAGES = ["flask", "pandas", "numpy", "yfinance", "requests"]
PORT = 5050
URL = f"http://127.0.0.1:{PORT}/"


def check_deps():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing required package(s): " + ", ".join(missing))
        print("Install them with:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def open_browser_when_ready():
    time.sleep(1.2)
    try:
        webbrowser.open(URL)
    except Exception:
        pass  # headless environment or no browser available -- URL is printed regardless


def main():
    check_deps()
    import app as app_module  # import after the dep check so a clean message shows first on a fresh install

    print("=" * 60)
    print("TRADE AUDITOR")
    print("=" * 60)
    print(f"Starting server on {URL}")
    print("Opening the setup wizard in your browser...")
    print("(Every /run finishes by printing the headline numbers right here.)")
    print("=" * 60)

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    app_module.app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    main()
