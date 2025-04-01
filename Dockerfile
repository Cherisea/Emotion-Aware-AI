# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update package list and install OpenCV dependencies
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

# Disable package download cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for store model weights
RUN mkdir -p model_cache

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "live_infer:app", "--host", "0.0.0.0", "--port", "8080"]
