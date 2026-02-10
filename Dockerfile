FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Using a newer environment for Flower compatibility
RUN pip install --no-cache-dir \
    flwr \
    numpy \
    scipy \
    scikit-learn \
    pillow \
    tqdm \
    pandas \
    opencv-python

# Copy source code
COPY . /app

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default command (can be overridden)
CMD ["python", "src/main.py"]
