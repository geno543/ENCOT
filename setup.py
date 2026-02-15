import os

from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")

    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="ENCOT",
    version="1.0.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Adibvafa Fallahpour",
    author_email="Adibvafa.fallahpour@mail.utoronto.ca",
    description=(
        "Transformer-based codon optimization for E. coli using "
        "deep learning with Augmented-Lagrangian GC control. "
        "Built on CodonTransformer for E. coli-specific optimization."
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/geno543/ENCOT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
