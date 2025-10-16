# 📋 requirements_v7.txt - FortiGate Semantic Shield v7.0
# =====================================================

# Core Dependencies
numpy>=1.21.0
scipy>=1.7.0
requests>=2.25.0
pyyaml>=6.0
flask>=2.0.0
aiohttp>=3.8.0
asyncio>=3.9.0
python-dotenv>=0.19.0
python-dateutil>=2.8.0

# Database Support
sqlite3>=3.35.0
sqlalchemy>=1.4.0
alembic>=1.7.0

# Advanced Mathematics
sympy>=1.9.0
networkx>=2.6.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Async Support
uvloop>=0.16.0
ujson>=4.0.0
orjson>=3.6.0

# Web Framework Support
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
httpx>=0.22.0

# Monitoring & Observability
prometheus-client>=0.12.0
graphite-api>=0.2.0
sentry-sdk>=1.5.0

# Configuration Management
configparser>=5.2.0
python-box>=6.0.0
envyaml>=0.6.0

# Logging & Tracing
loguru>=0.6.0
rich>=12.0.0
click>=8.0.0
typer>=0.4.0

# Testing Framework
pytest>=7.0.0
pytest-asyncio>=0.19.0
pytest-cov>=3.0.0
pytest-mock>=3.6.0
coverage>=6.2.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
isort>=5.10.0

# FortiGate Integration
fortiosapi>=0.11.0
fortiosdk>=0.3.0
fortiosapi-topology>=0.3.0

# Security & Encryption
cryptography>=3.4.0
bcrypt>=3.2.0
jwt>=1.2.0
passlib>=1.7.0

# Data Processing
pandas>=1.4.0
polars>=0.14.0
dask>=2022.1.0

# Machine Learning (Optional)
torch>=1.10.0
transformers>=4.16.0
scikit-learn>=1.0.2
xgboost>=1.6.0

# Natural Language Processing (Optional)
spacy>=3.4.0
nltk>=3.6.0
textblob>=0.17.0

# Computer Vision (Optional)
opencv-python>=4.5.0
pillow>=8.3.0
imageio>=2.16.0

# Time Series (Optional)
prophet>=1.1.0
statsmodels>=0.13.0
pmdarima>=1.8.0

# Geospatial (Optional)
geopandas>=0.10.0
shapely>=1.7.0
pyproj>=3.3.0

# Networking
netaddr>=0.8.0
ipaddress>=1.0.25
dnspython>=2.2.0

# System Information
psutil>=5.8.0
py-cpuinfo>=7.3.0

# Development Tools
jupyter>=1.0.0
ipykernel>=6.4.0
notebook>=6.4.0
jupyterlab>=3.2.0

# Documentation
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0
mkdocs>=1.2.0
mkdocs-material>=8.1.0

# Deployment
docker>=5.20.0
kubernetes>=23.0.0
helm>=3.8.0
ansible-core>=2.11.0
terraform>=1.1.0

# Cloud Provider SDKs (Optional)
boto3>=1.24.0
azure-sdk>=1.2.0
google-cloud-core>=2.3.0
google-cloud-compute>=1.5.0
google-cloud-storage>=2.1.0

# Enterprise Features
ldap3>=3.1.3
python-jose>=3.3.0
cryptography-hazmat>=0.4.0
azure-identity>=1.6.0
auth0-python>=3.18.0

# Debugging
pdbpp>=0.10.0
icecream>=2.1.0
ipython>=7.31.0

# Performance Monitoring
pyinstrument>=0.4.0
py-spy>=0.3.11
memory-profiler>=0.60.0
line-profiler>=3.5.0

# Configuration Management
hydra-core>=1.1.0
click-config>=0.6.0
pyyaml-env-parser>=0.2.0

# Security Analysis
voluptuous>=0.14.0
jsonschema>=4.0.0
marshmallow>=3.10.0
cerberus>=0.14.0

# API Documentation
apispec>=5.2.0
swagger-spec-validator>=3.0.0
redoc>=12.0.0

# Quality Assurance
pre-commit>=2.17.0
commitizen>=2.20.0
semantic-release>=17.27.0
safety>=2.0.0

# Database (Optional)
psycopg2-binary>=2.9.0
mysqlclient>=2.1.0
pymongo>=4.1.0

# Cache Support
redis>=4.2.0
memcached>=1.6.0
hiredis>=2.0.0

# Message Queue (Optional)
celery>=5.2.0
kombu>=2.1.0
pika>=1.3.0

# Web Crawling (Optional)
scrapy>=2.5.0
beautifulsoup4>=4.10.0
selenium>=4.1.0
playwright>=1.18.0

# Database Migration
alembic>=1.7.0
flask-migrate>=3.1.0

# Environment Variables
python-decouple==6.0.0

# Type Checking
types-python>=3.10.0
typing-extensions>=4.0.0

# File Handling
pathlib2>=2.3.6
filetype>=1.2.0
python-magic>=0.4.25

# PDF Processing (Optional)
PyPDF2>=2.11.0
pdfplumber>=0.7.0
reportlab>=3.6.0

# Excel Processing (Optional)
openpyxl>=3.0.10
xlrd>=2.0.1
xlsxwriter>=3.0.3

# Email (Optional)
smtplib>=2.3.0
email-validator>=1.1.3

# SSH/SFTP (Optional)
paramiko>=2.8.0
pysftp>=0.6.1
fabric>=2.6.0

# HTTP Client
httpx>=0.22.0
urllib3>=1.26.0
certifi>=2022.6.15

# JSON Processing
jsonpath-ng>=1.5.3
json5-schema>=0.9.2

# URL Handling
furl>=2.1.0
urllib3>=1.26.0

# Context Managers
contextlib>=2.4.0
contextvars>=2.6.0

# FortiGate Specific Dependencies
fortiosapi>=0.11.0
fortiosdk>=0.3.0
fortiosapi-topology>=0.3.0
fortiosapi-firewall>=0.10.0

# Performance Optimization
numba>=0.56.0
cython>=0.29.22
pybind11>=2.8.0

# Development Tools
setuptools>=60.0.0
wheel>=0.37.0
build>=0.7.0
twine>=4.0.0

# Virtual Environment
virtualenv>=20.0.0
pip-tools>=22.0.0
pipenv>=2022.1.8

# Logging Enhancements
structlog>=21.1.0
colorlog>=6.6.0
loguru>=0.6.0

# HTTP Client Enhancements
httpx[http2]>=0.22.0
aiohttp[speedups]>=3.8.0

# Testing Enhancements
pytest-asyncio>=0.19.0
pytest-xdist>=2.4.0
pytest-randomly>=3.10.0
pytest-html>=3.1.1
pytest-sugar>=0.9.4

# Coverage Enhancements
coverage[toml]>=6.2
coverage-h5>=0.3.0
coverage-enable-subprocess>=0.6

# API Documentation Enhancements
sphinx-copybutton>=0.3.0
sphinx-autodoc-typehints>=1.16
sphinx-autodoc-napoleon>=0.7
sphinx-jsdoc>=3.2.0

# Development Enhancements
ipython-genutils>=0.2.0
jupyterlab-git>=0.32.0
jupyterlab-lsp>=1.1.0
jupyter-themes>=0.12.0

# Security Enhancements
python-gssapi>=1.6.1
requests-mock>=1.9.1
responses>=0.18.0
http-mock>=1.9.1

# Enterprise Dependencies
keyring>=22.0.0
pywin32>=305
winsdk>=1.0.0
pywin32-ctypes>=0.2.0

# Financial Market Data (Optional)
yfinance>=0.1.70
pandas-datareader>=0.9.0
alpha_vantage>=3.5.0
quandl>=0.8.1

# Data Visualization (Enhanced)
plotly>=5.3.0
bokeh>=2.4.0
altair>=4.1.0

# API Framework Enhancements
fastapi-users>=0.4.0
fastapi-security>=0.2.0
fastapi-permissions>=0.3.0
fastapi-asyncio>=0.2.0

# Testing Enhancements
pytest-aiohttp>=0.21.0
pytest-celery>=0.4.0
pytest-xdist[envfile]>=2.4.0
pytest-memray>=0.21.0

# Database Enhancements
alembic[asyncio]>=1.7.0
alembic-extras>=0.0.0
sqlalchemy[asyncio]>=1.4.0
asyncpg>=0.25.0

# Advanced Analytics
dash>=2.7.0
streamlit>=1.12.0
panel>=0.13.0
holoviews>=1.13.0
voila>=0.2.2

# Error Handling
sentry-sdk[fastapi]>=1.5.0
sentry-sdk[flask]>=1.5.0
sentry-sdk[sqlalchemy]>=1.5.0

# Configuration Enhancements
dynaconf>=3.1.0
python-dotenv>=0.19.0
jsonnet>=0.18.0

# Background Tasks
background-task>=2.2.0
django-qrocket>=1.3.0
huey>=2.3.0

# Monitoring Enhancements
prometheus-flask-exporter>=0.20.0
sentry-sdk[fastapi]>=1.5.0
statsd>=4.0.1

# Authentication Enhancements
python-multipart>=0.0.5
flask-login>=0.6.0
python-jwt>=2.6.0
flask-jwt-extended>=4.2.0

# API Documentation Enhancements
apispec[fastapi]>=1.5.0
apispec[django]>=0.8.0
redoc[fastapi]>=0.11.0
swagger-ui[fastapi]>=0.8.0

# Production Optimizations
gunicorn>=20.1.0
uvicorn[standard]>=0.15.0
nginx>=1.18.0

# Additional Security
bandit>=1.7.2
safety>=2.0.0
pip-audit>=2.0.0

# CLI Tools
typer[all]>=0.4.0
click[all]>=8.0.0
rich-click>=0.0.0

# Documentation Enhancements
myst-parser>=0.17.0
mkdocs-material[imaging]>=8.1.0
mkdocs-git-committers-plugin>=1.1.0
mkdocs-mermaid2-plugin>=0.5.0

# Development Tools (Updated)
black[rules]=22.0.0
black[rules.line-length]=88
black[rules.arrow]=46
black[rules.quotes]=single
black[rules.content=org/binary]
black[rules.skipping_comprehensions]=False
black[rules.deprecated_imports=py38
black[rules.extend=skipping_parts]

# Version Pinning for Stability
numpy==1.21.6
pandas==1.4.0
scipy==1.7.0
matplotlib==3.5.0
seaborn==0.11.0
fastapi==0.68.0
uvicorn==0.15.0
pydantic==1.8.0
requests==2.25.1
sqlalchemy==1.4.0
alembic==1.7.0
pytest==7.0.0
coverage==6.2.0
black==22.0.0
isort==5.10.0
flake8==4.0.0
mypy==0.950

# FortiGate Specific Version Pinning
fortiosapi==0.11.0
fortiosdk==0.3.0
fortiosapi-topology==0.3.0
fortiosapi-firewall==0.10.0
fortiosapi-wifissd==0.11.0
fortiosapi-sslvpn==0.11.0
fortiosapi-orchestration==0.2.0
fortiosapi-sandbox==0.1.0

# Security
cryptography>=3.4.8
bcrypt==3.2.0
jwt==2.6.0
passlib==1.7.4

# Performance (Optimized)
numpy==1.21.6
scipy==1.7.0
numba==0.56.0
cython==0.29.22

# ML/AI (Optional Versions)
torch==1.10.2
transformers==4.16.2
scikit-learn==1.0.2
xgboost==1.6.0
tensorflow==2.8.0

# Data Processing
pandas==1.4.0
polars==0.14.0
dask>=2022.1.0