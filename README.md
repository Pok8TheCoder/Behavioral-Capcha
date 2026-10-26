# Behavioral CAPTCHA & Biometric Telemetry Core

A real-time bot detection engine and high-speed telemetry recorder built using **behavioral biometrics** (mouse dynamics, velocity/acceleration profiles, curvature, jerk analysis, dual-axis scroll tracking, and privacy-filtered keystroke dynamics).

---

## ⚡ Single-Command Quick Install (Any Device)

Run the single command corresponding to your operating system or preferred tool:

### 🪟 Windows (Command Prompt / PowerShell)
Double-click `install.bat` or run:
```cmd
install.bat
```
*Or via PowerShell:*
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

### 🍎 Linux & macOS (Bash / Zsh)
```bash
./install.sh
# Or run with bash directly:
bash install.sh
```

### 🐍 Cross-Platform Universal Python (Any OS)
If you have Python 3.9+ installed:
```bash
python start.py
```
*(Automatically sets up `.venv`, verifies and installs all dependencies, and launches the web interface at `http://localhost:8010`)*

### 🐳 Docker (Zero Setup, Any Architecture)
```bash
docker compose up --build
```
*Or build and run standalone:*
```bash
docker build -t behavioral-captcha . && docker run -p 8010:8010 behavioral-captcha
```

### 📦 Pip Package Installation (CLI Tool)
```bash
pip install -e .
```
Once installed, launch the application anytime with:
```bash
behavioral-captcha
# or short alias:
bcap
```

---

## 🌟 Key Features

1. **High-Speed Input Telemetry Recorder (`input_recorder.py` / `behavioral-recorder`)**:
   - **Microsecond Precision Timestamping**: 64-bit `uint64` microsecond resolution (`0.001ms`).
   - **1ms Win32 Timer**: Low-level hardware timer initialization (`winmm.timeBeginPeriod(1)`) supporting up to $1000\text{Hz}+$ mouse polling rates.
   - **Dual Scroll & Button Telemetry**: Tracks $X, Y$, horizontal/vertical scroll deltas ($dx, dy$), and all mouse buttons (Left, Right, Middle, Side/Gaming buttons $X1, X2$).
   - **Privacy-First Keystroke Filtering**: **ZERO** logging of text, letters, digits, or passwords. Only tracks modifier/functionality/navigation keys (Ctrl, Alt, Win, Shift, Tab, Caps, Esc, Enter, Arrows, F1-F12).
   - **Compact Binary RAM Buffer**: 20 bytes/event ($100,000\text{ events} = 1.8\text{MB RAM}$).
   - **300s Periodic Autosave & Recovery Engine**: Dumps checkpoints to `temp/` every 5 minutes with `--recover` dataset conversion utility.

2. **Machine Learning Bot Classifier & API (`backend/`)**:
   - **FastAPI REST API**: Real-time telemetry verification endpoints (`/api/verify` & `/api/simulate-bot`).
   - **Feature Extraction**: Computes spatial-temporal biometric parameters (Straightness Index $S_{\text{ratio}}$, mean/std velocity, acceleration, jerk, angular variance, micro-jitter ratio, click dwell time).
   - **Trained Model Registry**: Loaded with production-tested models (Qmodel family, Random Forest challenge models, Isolation Forest anomaly detectors).

3. **Interactive Visualizer & Bot Simulator (`frontend/`)**:
   - Live trajectory canvas colored by speed gradient.
   - CAPTCHA widget & biometrics dashboard.
   - Live bot attack simulator (Linear Bot, Bezier Curve Bot, Jitter Bot, Teleport Bot).

---

## 🚀 Usage & Commands

### 1. Launch Web Dashboard & API
```bash
python start.py
# Options:
#   --port 8010       Custom port
#   --reload          Hot reload on file changes
#   --no-browser      Headless / do not open browser
```
- Web Application: [http://localhost:8010/](http://localhost:8010/)
- Live Demo: [http://localhost:8010/live_demo.html](http://localhost:8010/live_demo.html)
- API Documentation: [http://localhost:8010/docs](http://localhost:8010/docs)

### 2. Run High-Speed Input Telemetry Recorder
```bash
behavioral-recorder
# Or directly:
python input_recorder.py
```
Press `Ctrl+C` to exit. Data will automatically export to `.csv` and `.bin` in `datasets/human/`.

### 3. Run Bot Simulation Benchmarks
```bash
python challenge_bot_runner.py --runs 1
```

---

## 📜 License
[MIT License](LICENSE)
