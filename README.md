# Behavioral CAPTCHA & Biometric Telemetry Core

A real-time bot detection engine and high-speed telemetry recorder built using **behavioral biometrics** (mouse dynamics, velocity/acceleration profiles, curvature, jerk analysis, dual-axis scroll tracking, and privacy-filtered keystroke dynamics).

---

## 🌟 Key Features

1. **High-Speed Input Telemetry Recorder (`input_recorder.py`)**:
   - **Microsecond Precision Timestamping**: 64-bit `uint64` microsecond resolution (`0.001ms`).
   - **1ms Win32 Timer**: Low-level hardware timer initialization (`winmm.timeBeginPeriod(1)`) supporting up to $1000\text{Hz}+$ mouse polling rates.
   - **Dual Scroll & Button Telemetry**: Tracks $X, Y$, horizontal/vertical scroll deltas ($dx, dy$), and all mouse buttons (Left, Right, Middle, Side/Gaming buttons $X1, X2$).
   - **Privacy-First Keystroke Filtering**: **ZERO** logging of text, letters, digits, or passwords. Only tracks modifier/functionality/navigation keys (Ctrl, Alt, Win, Shift, Tab, Caps, Esc, Enter, Arrows, F1-F12).
   - **Compact Binary RAM Buffer**: 20 bytes/event ($100,000\text{ events} = 1.8\text{MB RAM}$).
   - **300s Periodic Autosave & Recovery Engine**: Dumps checkpoints to `temp/` every 5 minutes with `--recover` dataset conversion utility.

2. **Machine Learning Bot Classifier & API (`backend/`)**:
   - **FastAPI REST API**: Real-time telemetry verification endpoints (`/api/verify` & `/api/simulate-bot`).
   - **Feature Extraction**: Computes 12 spatial-temporal biometric parameters (Straightness Index $S_{\text{ratio}}$, mean/std velocity, acceleration, jerk, angular variance, micro-jitter ratio, click dwell time).
   - **Scikit-Learn Random Forest Model**: Combined heuristic rule checks and trained Random Forest model.

3. **Interactive Visualizer & Bot Simulator (`frontend/`)**:
   - Live trajectory canvas colored by speed gradient.
   - CAPTCHA widget & biometrics dashboard.
   - Live bot attack simulator (Linear Bot, Bezier Curve Bot, Jitter Bot, Teleport Bot).

---

## 🚀 Quick Start

### 1. Installation
```powershell
# Clone the repository
git clone https://github.com/Pok8TheCoder/Behavioral-Capcha.git
cd Behavioral-Capcha

# Setup Python virtual environment & dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. Run High-Speed Input Telemetry Recorder
```powershell
.venv\Scripts\python.exe input_recorder.py
```
Press `Ctrl+C` to exit. Data will automatically export to `.csv` and `.bin` in `datasets/`.

### 3. Run Backend API Server
```powershell
.venv\Scripts\python.exe -m backend.main
```
FastAPI server runs on `http://127.0.0.1:8000`.

### 4. Run Interactive Frontend Visualizer
Open `frontend/index.html` in your browser.

---

## 📜 License
MIT License
