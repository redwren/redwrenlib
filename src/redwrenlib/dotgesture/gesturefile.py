
# dotgesture/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

import h5py
from typing import List, Optional

from sklearn.mixture import GaussianMixture

from .version_reads import version_readers, GESTURE_VERSION
from ..utils.debug import alert
from ..typing import (
    ModelParameters,
    data_dict_t,
)


#- GestureFile Class -------------------------------------------------------------------------------

class GestureFile:

    # Initialise the instance with default values
    def __init__(self, filename: str):
        self.__filename: str = filename
        self.__models: data_dict_t = {}

        # Gaussian Mixture Model parameters
        self.__n_components: int = 42
        self.__random_state: int = 2
        self.__threshold: float = -10.5


    # Create mew gesture file
    def create(self) -> bool:
        try:
            with h5py.File(self.__filename, 'w') as f:
                f.create_dataset('version', data=GESTURE_VERSION)
                f.create_dataset('n_components', data=self.__n_components)
                f.create_dataset('random_state', data=self.__random_state)
                f.create_dataset('threshold', data=self.__threshold)

        except Exception as e:
            alert(f"Unable to create file: {e}")
            return False

        return True


    # Add models for a sensor to the file
    def append_reading(self, label: str, models: List[GaussianMixture]) -> bool:
        try:
            with h5py.File(self.__filename, 'a') as f:
                gmm_group = f.create_group(label) # create a new group for each sensor

                for i, model in enumerate(models):
                    group = gmm_group.create_group(f'model_{i}')
                    group.create_dataset('weights', data=model.weights_)
                    group.create_dataset('means', data=model.means_)
                    group.create_dataset('covariances', data=model.covariances_)
                    group.create_dataset('precisions_cholesky', data=model.precisions_cholesky_)
                    group.create_dataset('n_components', data=model.n_components)

        except Exception as e:
            alert(f"Unable to append readings: {e}")
            return False

        return True


    # Read the gesture file; deserialise it
    def read(self) -> bool:
        try:
            with h5py.File(self.__filename, 'r') as f:
                file_version = f['version'][()].decode('utf-8')
                print(f"file version = {file_version}")
                if file_version in version_readers:
                    model_parameters, self.__models = version_readers[file_version](f)
                    self.__threshold = model_parameters.threshold
                    self.__random_state = model_parameters.random_state
                    self.__n_components = model_parameters.n_components

                else:
                    raise ValueError(f"Unsupported file version: {file_version}")

        except Exception as e:
            alert(f"Invalid Gesture File. {e}")
            return False

        return True


    #- Getters & Setters ---------------------------------------------------------------------------

    # Get models read from/written to the gesture file
    def get_models(self) -> data_dict_t: return self.__models


    # Get saved parameters in ModelParameters type
    def get_parameters(self) -> ModelParameters:
        return ModelParameters(
                random_state=self.__random_state,
                threshold=self.__threshold,
                n_components=self.__n_components
            )


    # Set parameters, either override individual, or override all with ModelParameters arg
    def set_parameters(self,
            parameters: Optional[ModelParameters]=None,
            n_components: Optional[int]=None,
            random_state: Optional[int]=None,
            threshold:Optional[float]=None
        ) -> None:
        if parameters is not None:
            self.__threshold = parameters.threshold
            assert parameters.n_components > 0, "n_components should be above 0"
            self.__n_components = parameters.n_components
            assert parameters.random_state > 1, "random_state should be above 1"
            self.__random_state = parameters.random_state

        if n_components is not None:
            assert n_components > 0, "n_components should be above 0"
            self.__n_components = n_components

        if random_state is not None:
            assert random_state > 1, "random_state should be above 1"
            self.__random_state = random_state

        if threshold is not None:
            self.__threshold = threshold

