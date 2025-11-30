
# dotgesture/__init__.py

#- Imports -----------------------------------------------------------------------------------------

from ._0_0_1 import read_file as read_0_0_1


#- Export ------------------------------------------------------------------------------------------

GESTURE_VERSION: str = "0.0.1"

version_readers = {
    "0.0.1": read_0_0_1,
}

__all__ = [
    "version_readers", "GESTURE_VERSION"
]

