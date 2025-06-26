FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for Python and Playwright
# Using the official Playwright recommended dependencies for Debian-based images
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libegl1 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libopengl0 \
    libstdc++6 \
    libwayland-client0 \
    libwebkit2gtk-4.0-37 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set DISPLAY for Xvfb
ENV DISPLAY=:99

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip before installing requirements
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps chromium

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
