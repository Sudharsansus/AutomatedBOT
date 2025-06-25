"""
Dockerfile for Next-Generation Gold Trading Bot
This Dockerfile sets up the environment for the trading bot to run on Fly.io
"""

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for persistent storage
RUN mkdir -p /data
VOLUME /data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info

# Expose port for health checks and metrics
EXPOSE 8080

# Command to run the application
CMD ["python", "next_gen_gold_trading_bot_integration.py"]
