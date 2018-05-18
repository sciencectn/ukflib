
from setuptools import setup
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'DESCRIPTION.rst')) as f:
    long_description = f.read()

setup(
    name="ukflib",
    author="Chris Taylor",
    author_email="sciencectn@gmail.com",
    description="An Unscented Kalman Filter library that allows for nonadditive process and measurement noise",
    keywords="ukf unscented kalman filter",
    version="0.0.3",
    py_modules=["ukflib"],
    install_requires=["numpy>=1.14.0",
                      "matplotlib>=2.2.0",
                      "scipy>=1.0.0"],
    url="https://github.com/sciencectn/ukflib",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research"
    ]
)


