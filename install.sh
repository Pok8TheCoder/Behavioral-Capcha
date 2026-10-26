#!/usr/bin/env bash
# Behavioral CAPTCHA - Single Command Installer for Linux & macOS
set -e

echo "======================================================================"
echo "   BEHAVIORAL CAPTCHA & BOT DETECTION ENGINE - UNIX INSTALLER"
echo "======================================================================"
echo ""

# Locate Python 3
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[!] Python 3 is not installed or not in PATH."
    echo "[*] Attempting system package manager detection..."
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing Python 3 via apt..."
        sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv
        PYTHON_BIN="python3"
    elif command -v brew >/dev/null 2>&1; then
        echo "Installing Python 3 via Homebrew..."
        brew install python
        PYTHON_BIN="python3"
    elif command -v dnf >/dev/null 2>&1; then
        echo "Installing Python 3 via dnf..."
        sudo dnf install -y python3 python3-pip
        PYTHON_BIN="python3"
    elif command -v pacman >/dev/null 2>&1; then
        echo "Installing Python 3 via pacman..."
        sudo pacman -Sy --noconfirm python python-pip
        PYTHON_BIN="python3"
    else
        echo "Please install Python 3.9+ manually from https://www.python.org/downloads/"
        exit 1
    fi
fi

echo "[+] Using Python: $($PYTHON_BIN --version 2>&1)"
echo "[*] Running bootstrap installer..."
echo ""

$PYTHON_BIN start.py "$@"
