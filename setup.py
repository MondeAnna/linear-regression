from setuptools import find_packages
from setuptools import setup


with open("requirements.txt") as file:
    dependencies = file.read().splitlines()

setup(
    name="oop_and_mlr",
    version="0.0.1a",
    author="MondeAnna",
    packages=find_packages(
        where="src",
        include=["utilities"],
    ),
    package_dir={"": "src"},
    install_requires=dependencies,
)
