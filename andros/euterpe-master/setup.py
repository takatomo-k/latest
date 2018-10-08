from setuptools import find_packages, setup

setup(name="euterpe",
    version="0.1",
    description="goddess of music",
    author="Andros Tjandra",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="",
    packages=find_packages(),
    install_requires=['numpy','scipy', 'torch', 'pytest', 'torchev', 
        'utilbox', 'tabulate', 'tqdm', 'pathos', 'librosa', 'tensorboardX', 'pandas']);
