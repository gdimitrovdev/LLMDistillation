FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY dataset_preparation.py .
COPY distillation.py .

# Create data directory
RUN mkdir -p data models evaluation

# Set a volume for data and output
VOLUME ["/app/data", "/app/models", "/app/evaluation"]

# Default command to run the environment test
CMD ["python", "environment_test.py"]