from pathlib import Path
from setuptools import setup, find_packages

long_description = Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8")

setup(
    name="tictactoe_gym",
    version="0.1.0",
    author="Justin Hall",
    description="A custom OpenAI Gym environment for tic-tac-toe.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hall4jm/tictactoe_gym",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gym==0.26.2",
        "pygame==2.1.2",
        "numpy>=1.21,<2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
