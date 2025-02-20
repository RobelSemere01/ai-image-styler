# Use a base image with necessary dependencies
FROM python:3.9-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, PyTorch, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install large dependencies separately first
RUN pip install --default-timeout=200 --no-cache-dir \
    torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu transformers accelerate \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn

# Copy application code
COPY backend /app/backend

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
