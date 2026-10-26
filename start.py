#!/usr/bin/env python3
"""
Behavioral CAPTCHA - Universal Zero-Config Launcher & Auto-Installer
--------------------------------------------------------------------
Runs on ANY operating system (Windows, macOS, Linux).
Detects/creates virtual environment, installs dependencies, and launches the server.

Usage:
    python start.py                 # Auto-installs if needed & launches
    python start.py --install-only  # Only install dependencies
    python start.py --port 8010     # Specify port
    python start.py --reload        # Enable live reload
"""

import os
import sys
import subprocess
import venv
import webbrowser
import time
import argparse

REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("xgboost", "xgboost"),
    ("pydantic", "pydantic"),
    ("cryptography", "cryptography"),
    ("pynput", "pynput"),
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT_DIR, ".venv")

def get_venv_executables():
    if sys.platform.startswith("win"):
        py_bin = os.path.join(VENV_DIR, "Scripts", "python.exe")
        pip_bin = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        py_bin = os.path.join(VENV_DIR, "bin", "python")
        pip_bin = os.path.join(VENV_DIR, "bin", "pip")
    return py_bin, pip_bin

def ensure_venv():
    py_bin, pip_bin = get_venv_executables()
    if not os.path.exists(py_bin):
        print(f"[*] Creating Python virtual environment in {VENV_DIR}...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print("[+] Virtual environment created successfully.")
        except Exception as err:
            print(f"[!] Standard venv creation failed: {err}")
            print("[*] Trying fallback with sys.executable...")
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    return py_bin, pip_bin

def check_missing_dependencies(py_bin):
    missing = []
    for mod_name, pkg_name in REQUIRED_MODULES:
        code = f"import {mod_name}"
        res = subprocess.run([py_bin, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            missing.append(pkg_name)
    return missing

def install_dependencies(pip_bin, py_bin):
    req_file = os.path.join(ROOT_DIR, "requirements.txt")
    print("=" * 65)
    print("  INSTALLING BEHAVIORAL CAPTCHA DEPENDENCIES")
    print("=" * 65)
    
    print("[*] Upgrading pip, setuptools, wheel...")
    subprocess.check_call([py_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if os.path.exists(req_file):
        print(f"[*] Installing requirements from {req_file}...")
        subprocess.check_call([pip_bin, "install", "-r", req_file])

    print("[*] Installing package in editable mode...")
    try:
        subprocess.check_call([pip_bin, "install", "-e", ROOT_DIR])
    except Exception as e:
        print(f"[!] Editable install optional warning: {e}")

    print("\n[+] All dependencies installed successfully!\n")

def main():
    parser = argparse.ArgumentParser(description="Behavioral CAPTCHA Universal Launcher")
    parser.add_argument("--host", default=os.getenv("BEHAVIORAL_HOST", "0.0.0.0"), help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("BEHAVIORAL_PORT", "8010")), help="Port (default: 8010)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    parser.add_argument("--install-only", action="store_true", help="Only install requirements and exit")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Step 1: Ensure venv
    py_bin, pip_bin = ensure_venv()

    # Step 2: Check & install missing dependencies
    missing = check_missing_dependencies(py_bin)
    if missing or args.install_only:
        if missing:
            print(f"[*] Missing dependencies detected: {', '.join(missing)}")
        install_dependencies(pip_bin, py_bin)

    if args.install_only:
        print("[+] Installation complete.")
        return

    # Step 3: Launch server via venv python
    print("=" * 65)
    print("  STARTING BEHAVIORAL CAPTCHA SERVER")
    print("=" * 65)
    host_display = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f" - Local Web App:  http://{host_display}:{args.port}/")
    print(f" - API Docs:       http://{host_display}:{args.port}/docs")
    print(f" - Press Ctrl+C to stop the server\n")

    if not args.no_browser:
        def _launch():
            time.sleep(1.5)
            try:
                webbrowser.open(f"http://localhost:{args.port}/")
            except Exception:
                pass
        import threading
        threading.Thread(target=_launch, daemon=True).start()

    cmd = [
        py_bin, "-m", "uvicorn", "backend.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.extend(["--reload", "--reload-dir", os.path.join(ROOT_DIR, "backend")])

    try:
        subprocess.run(cmd, cwd=ROOT_DIR)
    except KeyboardInterrupt:
        print("\n[*] Server shutdown cleanly.")

if __name__ == "__main__":
    main()
