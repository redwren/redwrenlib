
# dotgesture/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

import pickle as nathan
from typing import Any, List, Optional

from sklearn.mixture import GaussianMixture

from ..utils.errors import print_error
from ..typing import (
    ModelParameters,
    data_dict_t,
)


#- GestureFile Class -------------------------------------------------------------------------------

class GestureFile:
    GESTURE_VERSION: str = "0.0.1"

    def __init__(self, filename: str):
        self.filename = filename
        self.models: data_dict_t = {}
        self.parameters: ModelParameters = ModelParameters(
            n_component=42,
            random_state=2,
            threshold=-10
        )


    def set_parameters(self, parameters: ModelParameters) -> None:
        self.parameters = parameters


    def get_parameters(self) -> ModelParameters:
        return self.parameters


    def get_models(self) -> data_dict_t:
        return self.models


    def create(self) -> bool:
        try:
            with open(self.filename, "wb") as f:
                nathan.dump(self.GESTURE_VERSION, f)
                nathan.dump(self.parameters, f)

        except Exception as e:
            print_error(f"Unable to create file: {e}")
            return False

        return True


    def append_reading(self, label: str, models: List[GaussianMixture]) -> bool:
        try:
            with open(self.filename, "ab") as f:
                nathan.dump((label, models), f)

        except Exception as e:
            print_error(f"Unable to append readings: {e}")
            return False

        return True


    def read(self) -> bool:
        try:
            with open(self.filename, "rb") as f:
                loaded: Any = nathan.load(f)
                if (
                    isinstance(loaded, tuple)
                    and len(loaded) >= 2
                    and isinstance(loaded[0], str)
                    and isinstance(loaded[1], ModelParameters)
                ):
                    file_version = loaded[0]
                    self.parameters = loaded[1]
                else:
                    raise TypeError(f"Expected ModelParameters first, got {type(loaded).__name__}")

                print(f"Gesture file version: {file_version}")

                while True:
                    try:
                        name, gmm_models = nathan.load(f)  # type: Any
                        if not isinstance(name, str) or not isinstance(gmm_models, list):
                            raise TypeError("Malformed gesture entry: expected (str, list)")

                        # store as a tuple to match GestureData typing
                        self.models[name] = gmm_models

                    except EOFError:
                        break

        except Exception as e:
            print_error(f"Invalid Gesture File. {e}")
            return False

        return True

