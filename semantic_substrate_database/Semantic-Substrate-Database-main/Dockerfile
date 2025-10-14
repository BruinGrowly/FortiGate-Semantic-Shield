# Semantic Substrate Database - Docker Image
# Self-aware, value-aligned semantic database system

FROM python:3.8-slim

LABEL maintainer="BruinGrowly"
LABEL description="Self-aware semantic database with value alignment and 5-layer decomposition"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY *.py ./
COPY src/ ./src/
COPY examples/ ./examples/

# Create directory for database files
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SSDB_DATA_DIR=/app/data

# Expose port for potential API server (future enhancement)
EXPOSE 8000

# Default command: run core demonstration
CMD ["python", "src/semantic_substrate_database.py"]

# Alternative commands:
# Run basic example: docker run ssdb python examples/basic_example.py
# Run tests: docker run ssdb python test_semantic_database.py
# Interactive shell: docker run -it ssdb /bin/bash
