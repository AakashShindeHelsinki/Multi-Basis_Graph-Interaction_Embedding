from setuptools import setup, find_packages

setup(
    name="mbgie",
    version="0.1.0",
    description="Multi-Basis Graph Interaction Embedding for quantum data embedding",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="MBGIE Contributors",
    url="https://github.com/yourusername/Multi-Basis_Graph-Interaction_Embedding",
    packages=find_packages(),
    install_requires=[
        "pennylane>=0.32.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
        ],
    },
    python_requires=">=3.8",
)