# Use 3.7-slim-buster to ensure active Linux update mirrors (Required for apt-get)
FROM python:3.7-slim-buster

# Install system dependencies for audio and video processing
RUN echo "deb http://archive.debian.org/debian buster main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip to avoid legacy SSL issues
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory with correct permissions
RUN mkdir -p uploads && chmod 777 uploads

# Environment variables for production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the app using Gunicorn binding to the env PORT
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4 app:app
