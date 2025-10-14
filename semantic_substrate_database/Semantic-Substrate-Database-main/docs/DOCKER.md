# Docker Deployment Guide
## Semantic Substrate Database

This guide explains how to run SSDB using Docker for easy deployment and reproducibility.

## 🚀 Quick Start

### Run Ultimate Demonstration
```bash
docker build -t ssdb:latest .
docker run ssdb:latest
```

### Run with Docker Compose
```bash
docker-compose up ssdb
```

## 📦 Available Images

### Production Image
- **Image**: `semantic-substrate-db:1.0.0`
- **Base**: Python 3.8 slim
- **Size**: ~200MB
- **Includes**: All SSDB layers, examples, tests

## 🎯 Common Use Cases

### 1. Run Ultimate Demonstration
```bash
docker run --rm ssdb:latest python ultimate_demonstration.py
```

### 2. Run Basic Example
```bash
docker run --rm ssdb:latest python examples/basic_example.py
```

### 3. Run Complete Test Suite
```bash
docker-compose --profile test up ssdb-test
```

### 4. Interactive Development
```bash
docker-compose --profile dev run --rm ssdb-dev
```

### 5. Persistent Database Storage
```bash
# Create data directory
mkdir -p ./data

# Run with volume mount
docker run --rm -v $(pwd)/data:/app/data ssdb:latest python examples/basic_example.py
```

## 🔧 Configuration

### Environment Variables

- `SSDB_DATA_DIR`: Directory for database files (default: `/app/data`)
- `PYTHONUNBUFFERED`: Enable Python unbuffered output (default: `1`)

### Volume Mounts

Mount local directories into the container:

```yaml
volumes:
  - ./data:/app/data              # Database files
  - ./config:/app/config:ro       # Custom configurations
  - ./examples:/app/examples      # Custom examples
```

## 🏗️ Building the Image

### Standard Build
```bash
docker build -t ssdb:latest .
```

### Build with Custom Tag
```bash
docker build -t myorg/ssdb:1.0.0 .
```

### Multi-platform Build
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t ssdb:latest .
```

## 🧪 Running Tests

### Run All Tests
```bash
docker run --rm ssdb:latest /bin/bash -c "\
  python test_semantic_database.py && \
  python test_enhanced_database.py && \
  python test_meaning_based_database.py && \
  python test_deep_dive_database.py"
```

### Run Specific Test
```bash
docker run --rm ssdb:latest python test_semantic_database.py
```

### Using Docker Compose
```bash
docker-compose --profile test up ssdb-test
```

## 🔍 Development Workflow

### 1. Start Development Container
```bash
docker-compose --profile dev run --rm ssdb-dev
```

### 2. Inside Container
```bash
# Run examples
python examples/basic_example.py

# Run tests
python test_semantic_database.py

# Interactive Python
python
>>> from deep_dive_database import DeepDiveDatabase
>>> db = DeepDiveDatabase("/app/data/test.db")
>>> # ... experiment ...
```

### 3. Hot Reload Development
Mount your local code:
```bash
docker run --rm -it -v $(pwd):/app ssdb:latest /bin/bash
```

## 📊 Production Deployment

### Docker Compose Production Setup
```yaml
version: '3.8'
services:
  ssdb-api:
    image: ssdb:1.0.0
    restart: always
    volumes:
      - ssdb-data:/app/data
    environment:
      - SSDB_DATA_DIR=/app/data
    ports:
      - "8000:8000"
    command: python api_server.py  # When API server is implemented

volumes:
  ssdb-data:
    driver: local
```

### Health Check
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import sqlite3; sqlite3.connect('/app/data/health.db').close()"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## 🐳 Docker Hub Distribution

### Tag and Push
```bash
# Tag the image
docker tag ssdb:latest bruingrowly/semantic-substrate-db:1.0.0
docker tag ssdb:latest bruingrowly/semantic-substrate-db:latest

# Push to Docker Hub
docker push bruingrowly/semantic-substrate-db:1.0.0
docker push bruingrowly/semantic-substrate-db:latest
```

### Pull and Run
```bash
docker pull bruingrowly/semantic-substrate-db:latest
docker run --rm bruingrowly/semantic-substrate-db:latest
```

## 🔐 Security Considerations

### Non-root User (Enhanced Security)
Update Dockerfile:
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 ssdb && \
    chown -R ssdb:ssdb /app

USER ssdb
```

### Read-only Root Filesystem
```bash
docker run --rm --read-only -v $(pwd)/data:/app/data ssdb:latest
```

## 🎯 Example Workflows

### Research Workflow
```bash
# 1. Start container with data persistence
docker run -d --name ssdb-research \
  -v $(pwd)/research_data:/app/data \
  ssdb:latest tail -f /dev/null

# 2. Run experiments
docker exec ssdb-research python examples/basic_example.py

# 3. Access database files
ls -la research_data/

# 4. Cleanup
docker stop ssdb-research
docker rm ssdb-research
```

### Continuous Integration
```yaml
# .github/workflows/docker.yml
name: Docker Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t ssdb-test .
      - name: Run tests
        run: docker run ssdb-test python test_semantic_database.py
```

## 📝 Troubleshooting

### Permission Issues
```bash
# Fix permissions on data directory
sudo chown -R $(id -u):$(id -g) ./data
```

### Container Won't Start
```bash
# Check logs
docker logs <container-id>

# Inspect container
docker inspect <container-id>
```

### Out of Disk Space
```bash
# Clean up unused images
docker system prune -a

# Remove specific image
docker rmi ssdb:latest
```

## 🌐 Network Configuration

### Expose API Port (Future)
```bash
docker run -p 8000:8000 ssdb:latest
```

### Custom Network
```bash
docker network create ssdb-network
docker run --network ssdb-network ssdb:latest
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [SSDB Technical Whitepaper](TECHNICAL_WHITEPAPER.md)
- [SSDB Deployment Guide](DEPLOYMENT_GUIDE.md)

---

**The future of databases is semantic, self-aware, and value-aligned.**

Built with transparency and ethical foundations.
