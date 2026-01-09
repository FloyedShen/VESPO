"""
Optimal Sampling for vLLM V1

A high-performance optimal sampling implementation for vLLM V1 engine,
designed for semi on-policy distillation and high-quality data generation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="optimal-sampling-vllm",
    version="0.1.0",
    author="Your Team",
    author_email="your-email@example.com",
    description="Optimal Sampling for vLLM V1 - Semi On-Policy Distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/optimal-sampling-vllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.11.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optimal-sampling=optimal_sampling.cli:main",
        ],
    },
)
