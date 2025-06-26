FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies and build tools for MetaTrader5
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip before installing requirements
RUN pip install --upgrade pip

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
