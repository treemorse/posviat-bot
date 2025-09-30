# Use Python 3.13 slim
FROM python:3.13-slim

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
# pin stable libs; headless OpenCV to avoid X11 deps
RUN pip install --no-cache-dir \
      fastapi==0.115.2 \
      "uvicorn[standard]==0.30.6" \
      python-telegram-bot==21.6 \
      cryptography==43.0.1 \
      qrcode==7.4.2 \
      pillow==10.4.0 \
      opencv-python-headless==4.10.0.84

# Copy app
COPY app.py /app/app.py

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
