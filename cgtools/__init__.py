"""CGMD for pyRosetta"""
import importlib.resources

PRT_DATA = importlib.resources.files('pyrotini.data')
PRT_DICT = {}
for file in PRT_DATA.iterdir():
    # name = f"{file.name}".replace('.', '_')
    name = f"{file.name}"
    path = f"{file.absolute()}"
    PRT_DICT[name] = path

# Add imports here
from .pyrotini import *


from ._version import __version__







