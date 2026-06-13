from setuptools import find_packages
from setuptools import setup


INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "numpy",
    "torch",
    "tyro",
    "prettytable",
]


setup(
    name="z1_amp_mjlab",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=INSTALL_REQUIRES,
)
