# Use a Python 3.7 base image (Stable for TF 1.14)
FROM python:3.7-slim

# Install system dependencies for audio and video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory with correct permissions
RUN mkdir -p uploads && chmod 777 uploads

# Expose port 5001
EXPOSE 5001

# Environment variables for production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the app using Gunicorn
# Using 1 worker and 4 threads as a balance for AI model loading in limited RAM
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "--workers", "1", "--threads", "4", "app:app"]
