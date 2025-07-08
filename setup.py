#!/usr/bin/env python
"""
Setup script for Integral Shapley Values (ISV) package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="integral-shapley-values",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Efficient Shapley value computation using integral formulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/integral-shapley-values",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "integral-shapley=src.core.integral_shapley:main",
            "shapley-efficiency=src.experiments.efficiency_study:main",
            "shapley-smoothness=src.experiments.smoothness_analysis:main",
            "shapley-convergence=src.experiments.convergence_study:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "shapley values",
        "machine learning",
        "data valuation", 
        "game theory",
        "explainable ai",
        "integral methods",
        "numerical integration",
        "monte carlo",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/integral-shapley-values/issues",
        "Source": "https://github.com/example/integral-shapley-values",
        "Documentation": "https://integral-shapley-values.readthedocs.io/",
    },
)