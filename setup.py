"""
Setup script for Earth System Simulation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="earth_system_sim",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid Earth system simulation using PINNs and RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/earth_system_sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "isort>=5.9.1",
            "pre-commit>=2.13.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "run-simulation=scripts.run_simulation:main",
            "create-visualizations=scripts.visualize_results:main",
            "verify-setup=scripts.verify_setup:main",
            "setup-github=scripts.setup_github:main",
        ],
    },
    include_package_data=True,
    package_data={
        "earth_system_sim": [
            "config/*.yaml",
            "examples/*.ipynb",
        ]
    },
)