# gym-game2048/setup.py
from setuptools import setup, find_packages

setup(
    name="gym_game2048",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gym",
        "numpy",
        "torch",
        "tqdm",
        "matplotlib",
        "tabulate"
    ]
)