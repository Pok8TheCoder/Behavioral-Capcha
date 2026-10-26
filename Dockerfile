# Behavioral CAPTCHA & Biometric Bot Detection Engine
FROM python:3.11-slim

# Avoid buffering and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BEHAVIORAL_HOST=0.0.0.0 \
    BEHAVIORAL_PORT=8010

WORKDIR /app

# Install system build dependencies for scikit-learn / xgboost runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (caching layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Install the package itself
RUN pip install --no-cache-dir -e .

# Expose server port
EXPOSE 8010

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/ || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8010"]
