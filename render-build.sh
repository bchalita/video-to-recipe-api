#!/usr/bin/env bash

echo "[BUILD] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[BUILD] Checking NumPy installation..."
pip show numpy
python -c "import numpy; print('[RENDER] NumPy is available:', numpy.__version__)"
