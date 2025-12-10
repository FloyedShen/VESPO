"""
Setup script for optimal_sampling_hf package
"""

from setuptools import setup, find_packages

setup(
    name="optimal_sampling_hf",
    version="1.0.0",
    description="Optimal Sampling with HuggingFace Transformers + Flash Attention 2",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "flash-attn>=2.0.0",  # Optional but recommended
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
