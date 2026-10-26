@echo off
setlocal enabledelayedexpansion

title Behavioral CAPTCHA - One-Click Installer & Launcher
echo ======================================================================
echo    BEHAVIORAL CAPTCHA ^& BOT DETECTION ENGINE - ONE-CLICK INSTALLER
echo ======================================================================
echo.

:: 1. Detect Python
set PYTHON_EXE=
for %%P in (python.exe py.exe python3.exe) do (
    where %%P >nul 2>nul
    if !errorlevel! equ 0 (
        if not defined PYTHON_EXE set PYTHON_EXE=%%P
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python was not found in your system PATH!
    echo.
    echo Attempting to install Python 3 via Windows Package Manager (winget)...
    where winget >nul 2>nul
    if !errorlevel! equ 0 (
        echo Running: winget install -e --id Python.Python.3.11
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        echo.
        echo Please restart your terminal or computer after Python installation completes,
        echo then run install.bat again.
        pause
        exit /b 1
    ) else (
        echo Please download and install Python 3.9+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        start https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

echo [+] Found Python: !PYTHON_EXE!
echo [*] Initializing installation and launching server...
echo.

:: 2. Hand off to start.py
!PYTHON_EXE! start.py %*

if !errorlevel! neq 0 (
    echo.
    echo [!] Process exited with code !errorlevel!.
    pause
)
