"""
NeuralBlitz v50.0 - Omega Singularity Architecture
Setup script for Python implementation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="neuralblitz",
    version="50.0.0",
    author="NeuralBlitz Core Team",
    author_email="core@neuralblitz.ai",
    description="Omega Singularity Architecture v50.0 - Irreducible Source of All Being",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuralblitz/core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "neuralblitz=neuralblitz.cli:main",
        ],
    },
)
