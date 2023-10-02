"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="mobsim",
    version="0.0.2",
    description="Individual mobility simulation",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Ye Hong",
    author_email=("hongy@ethz.ch"),
    license="Apache-2.0",
    url="https://github.com/irmlma/mobility-simulation",
    install_requires=["geopandas", "scipy", "numpy", "pyyaml", "powerlaw"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 1 - Planning",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.9",
)
