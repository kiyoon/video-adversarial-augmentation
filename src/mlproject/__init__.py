try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

from pathlib import Path

all = ["__version__", "__version_tuple__", "SOURCE_DIR", "ROOT_DIR"]

SOURCE_DIR = Path(__file__).absolute().parent
ROOT_DIR = SOURCE_DIR.parent.parent
