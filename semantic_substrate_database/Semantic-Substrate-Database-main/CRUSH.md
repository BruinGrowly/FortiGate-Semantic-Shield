# Semantic Substrate Database - Development Commands

## 🧪 Testing Commands
```bash
# Quick test
python tests/quick_test.py

# Integration tests
python tests/test_integration.py

# All tests
python tests/run_all_tests.py

# Semantic database tests
python tests/test_semantic_database.py

# Enhanced database tests
python tests/test_enhanced_database.py

# API tests
python api/test_api.py
```

## 🏗️ Development Commands
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Code formatting
black src/ tests/ examples/

# Linting
pylint src/

# Type checking
mypy src/

# Test coverage
python -m pytest --cov=src tests/
```

## 🐳 Docker Commands
```bash
# Build image
docker build -t semantic-substrate-db .

# Run container
docker run -it semantic-substrate-db

# Run with volume
docker run -v $(pwd):/app semantic-substrate-db

# Using docker-compose
docker-compose up

# Production deployment
docker-compose -f docker-compose.yml up -d
```

## 📦 Package Commands
```bash
# Build package
python setup.py sdist bdist_wheel

# Install from local
pip install dist/semantic_substrate_database-*.whl

# Upload to PyPI (requires credentials)
twine upload dist/*
```

## 🔧 Database Commands
```bash
# Initialize database
python -c "from src import SemanticSubstrateDatabase; db = SemanticSubstrateDatabase('test.db')"

# Run examples
python examples/basic_example.py
python examples/basic_usage.py
python examples/visualization_example.py

# Database operations
python -c "
from src import SemanticSubstrateDatabase
db = SemanticSubstrateDatabase('demo.db')
db.store_concept('Show compassion', 'spiritual')
results = db.search_semantic('help others')
print(f'Found {len(results)} results')
"
```

## 🚀 API Commands
```bash
# Start API server
python api/semantic_api.py

# Test API
curl -X GET http://localhost:5000/api/health
curl -X POST http://localhost:5000/api/store -H "Content-Type: application/json" -d '{"text": "Act with compassion", "context": "ethical"}'
```

## 📊 Performance Testing
```bash
# Benchmark database operations
python -c "
import time
from src import SemanticSubstrateDatabase
db = SemanticSubstrateDatabase('perf.db')
start = time.time()
for i in range(100):
    db.store_concept(f'Concept {i}', 'test')
print(f'Stored 100 concepts in {time.time()-start:.3f}s')
"
```

## 🔍 Debugging Commands
```bash
# Enable debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src import SemanticSubstrateDatabase
db = SemanticSubstrateDatabase('debug.db')
# ... operations will show debug output
"
```

## 📝 Documentation Commands
```bash
# Generate documentation
python -c "
from src import SemanticSubstrateDatabase
help(SemanticSubstrateDatabase)
"

# Check version
python -c "from src import __version__; print(__version__)"
```

## 🧹 Cleanup Commands
```bash
# Clean test databases
rm -f *.db test_*.db

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## 🔄 Git Commands
```bash
# Check status
git status

# Add all changes
git add .

# Commit with message
git commit -m "Your descriptive commit message"

# Push to remote
git push origin main

# Pull latest changes
git pull origin main
```

## 📈 Monitoring Commands
```bash
# Database statistics
python -c "
from src import SemanticSubstrateDatabase
db = SemanticSubstrateDatabase('stats.db')
stats = db.get_statistics()
print('Database Statistics:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"
```

## 🔐 Security Commands
```bash
# Check dependencies for vulnerabilities
pip list --outdated

# Update dependencies
pip install --upgrade -r requirements.txt

# Security audit
python -m bandit -r src/
```