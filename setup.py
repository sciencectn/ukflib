
from setuptools import setup


setup(
    name="ukflib",
    author="Chris Taylor",
    author_email="sciencectn@gmail.com",
    description="An Unscented Kalman Filter library that allows for nonadditive process and measurement noise",
    keywords="ukf unscented kalman filter",
    version="0.0.1",
    py_modules=["ukflib"],
    install_requires=["numpy>=1.14.0",
                      "matplotlib>=2.2.0",
                      "scipy>=1.0.0"],
    url="https://github.com/sciencectn/ukflib",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research"
    ]
)


