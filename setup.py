from setuptools import setup, find_packages

setup(
    name="behavioral-captcha",
    version="1.0.0",
    description="Real-time behavioral biometric CAPTCHA and ML bot detection engine",
    author="Pok8TheCoder",
    packages=find_packages(include=["backend*", "bots*", "challenge_server*"]),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.28.0",
        "websockets>=12.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "joblib>=1.3.0",
        "xgboost>=2.0.0",
        "pydantic>=2.6.0",
        "python-multipart>=0.0.9",
        "cryptography>=42.0.0",
        "pynput>=1.7.6",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "behavioral-captcha = backend.main:cli",
            "bcap = backend.main:cli",
            "behavioral-recorder = input_recorder:main",
        ],
    },
    python_requires=">=3.9",
)
