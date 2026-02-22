FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system dependencies (OpenCV, YOLO requirements)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights into weights/ directory
RUN mkdir -p /app/weights && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && mv yolov8n.pt /app/weights/ 2>/dev/null || true

# Copy source code
COPY . /app

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Create log and save directories
RUN mkdir -p /app/log /app/save_folder

# Default command
CMD ["python", "src/main.py"]
