# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY live_infer.py .
COPY models/Resnet.py ./models/
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for model cache
RUN mkdir -p model_cache

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "live_infer:app", "--host", "0.0.0.0", "--port", "8000"]
