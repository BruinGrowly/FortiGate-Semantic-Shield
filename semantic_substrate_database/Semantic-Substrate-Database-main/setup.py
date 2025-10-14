"""
Semantic Substrate Database - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'sympy>=1.9',
        'matplotlib>=3.4.0',
        'semantic-substrate-engine>=3.0.0',
    ]

setup(
    name="semantic-substrate-database",
    version="2.0.0",
    author="BruinGrowly",
    author_email="",
    description="A revolutionary ICE-centric database system for semantic meaning storage and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BruinGrowly/Semantic-Substrate-Database",
    project_urls={
        "Bug Tracker": "https://github.com/BruinGrowly/Semantic-Substrate-Database/issues",
        "Documentation": "https://github.com/BruinGrowly/Semantic-Substrate-Database/tree/main/docs",
        "Source Code": "https://github.com/BruinGrowly/Semantic-Substrate-Database",
        "Engine": "https://github.com/BruinGrowly/Semantic-Substrate-Engine",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pylint>=2.12.0',
            'black>=22.0.0',
            'mypy>=0.950',
            'httpx>=0.25.1',
        ],
        'viz': [
            'matplotlib>=3.4.0',
        ],
        'api': [
            'fastapi>=0.104.1',
            'uvicorn[standard]>=0.24.0',
            'pydantic>=2.5.0',
            'python-multipart>=0.0.6',
            'python-jose[cryptography]>=3.3.0',
            'passlib[bcrypt]>=1.7.4',
        ],
    },
    keywords=[
        'semantic-database',
        'ice-framework',
        'meaning-storage',
        'semantic-search',
        'artificial-intelligence',
        'database',
        'semantic-analysis',
        'value-alignment',
        'context-aware',
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)
