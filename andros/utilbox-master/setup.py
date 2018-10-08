from setuptools import find_packages, setup

setup(name="utilbox",
    version="0.1",
    description="utility box - private common used snippet codes",
    author="Andros Tjandra",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="",
    packages=find_packages(),
    install_requires=['numpy','matplotlib','scipy']);
