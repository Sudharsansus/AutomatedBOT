# Use a base image with Python and Playwright dependencies pre-installed
# This image is maintained by Playwright and includes all necessary browser dependencies
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set the working directory in the container
WORKDIR /app

# Copy your requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot\"s code and configuration into the container
COPY . .

# Create data directory for persistent storage (if needed, otherwise remove)
# RUN mkdir -p /data
# VOLUME /data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=info

# Expose port for health checks and metrics (if your app serves anything, otherwise remove)
# EXPOSE 8080

# Command to run your bot when the container starts
# This assumes your main script is next_gen_gold_trading_bot_integration.py
CMD ["python", "next_gen_gold_trading_bot_integration.py"]
