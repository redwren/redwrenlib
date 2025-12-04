
# dotgesture/__init__.py

#- Imports -----------------------------------------------------------------------------------------

from .v1 import read_file as v1
from .v2 import read_file as v2


#- Export ------------------------------------------------------------------------------------------

GESTURE_VERSION: int = 2

version_readers = {
    "0.0.1": v1,
    "2": v2,
}

__all__ = [
    "version_readers", "GESTURE_VERSION"
]

