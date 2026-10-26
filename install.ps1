<#
.SYNOPSIS
    Behavioral CAPTCHA - Single Command PowerShell Installer & Launcher
.DESCRIPTION
    Installs virtual environment, dependencies, and boots the application.
#>

[CmdletBinding()]
param (
    [int]$Port = 8010,
    [switch]$InstallOnly,
    [switch]$Reload,
    [switch]$NoBrowser
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   BEHAVIORAL CAPTCHA & BOT DETECTION ENGINE - POWERSHELL INSTALLER   " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Find Python executable
$PythonExe = (Get-Command python.exe, py.exe, python3.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)

if (-not $PythonExe) {
    Write-Warning "Python was not found on your system."
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($Winget) {
        Write-Host "Attempting automatic installation via winget..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        Write-Host "Please restart your PowerShell window after Python finishes installing and rerun this command." -ForegroundColor Green
        return
    } else {
        Write-Error "Please install Python 3.9+ from https://www.python.org/downloads/ and ensure 'Add Python to PATH' is checked."
        Start-Process "https://www.python.org/downloads/"
        return
    }
}

Write-Host "[+] Using Python: $PythonExe" -ForegroundColor Green

$ArgsList = @("start.py", "--port", "$Port")
if ($InstallOnly) { $ArgsList += "--install-only" }
if ($Reload) { $ArgsList += "--reload" }
if ($NoBrowser) { $ArgsList += "--no-browser" }

& $PythonExe $ArgsList
