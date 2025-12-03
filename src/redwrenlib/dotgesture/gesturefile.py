
# dotgesture/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

import h5py
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .version_reads import version_readers, GESTURE_VERSION
from ..utils.debug import alert
from ..typing import (
    ModelParameters, GestureMatch,
    data_dict_t,
)


#- GestureFile Class -------------------------------------------------------------------------------

class GestureFile:

    # Initialise the instance with default values
    def __init__(self, filename: str):
        """Initialise the GestureFile instance with the given filename."""
        self._filename: str = filename
        self._models: data_dict_t = {}

        # Gaussian Mixture Model parameters
        self._n_components: int = 42
        self._random_state: int = 2
        self._threshold: float = -10.5


    # Create mew gesture file
    def create(self) -> bool:
        """Create a new gesture file and save initial parameters.

        Returns:
            bool: True if the file was created successfully, False if there was an error.
        """
        try:
            with h5py.File(self._filename, 'w') as f:
                f.create_dataset('version', data=GESTURE_VERSION)
                f.create_dataset('n_components', data=self._n_components)
                f.create_dataset('random_state', data=self._random_state)
                f.create_dataset('threshold', data=self._threshold)

        except Exception as e:
            alert(f"Unable to create file: {e}")
            return False

        return True


    # Add models for a sensor to the file
    def append_reading(self, label: str, models: List[GaussianMixture]) -> bool:
        """Add models for a sensor to the gesture file with the specified label.

        Returns:
            bool: True if the models were appended successfully, False if there was an error.
        """
        try:
            with h5py.File(self._filename, 'a') as f:
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
        """Read and deserialise the gesture file, updating internal model parameters.

        Returns:
            bool: True if the file was read successfully, False if there was an error.
        """
        try:
            with h5py.File(self._filename, 'r') as f:
                file_version = f['version'][()].decode('utf-8')
                print(f"file version = {file_version}")

                if file_version in version_readers:
                    model_parameters, self._models = version_readers[file_version](f)
                    self._threshold = model_parameters.threshold
                    self._random_state = model_parameters.random_state
                    self._n_components = model_parameters.n_components

                else:
                    raise ValueError(f"Unsupported file version: {file_version}")

        except Exception as e:
            alert(f"Invalid Gesture File. {e}")
            return False

        return True


    # Check if the readings trigger the gesture
    def is_gesture(self,
        timestamps: List[float], readings: Dict[str, List[float]]
    ) -> Tuple[bool, Dict[str, GestureMatch]]:
        """Check whether the provided readings trigger any gestures based on the loaded models.

        Args:
            timestamps: A list of timestamps corresponding to the sensor readings.
            readings: A dictionary of keys with sensor labels and values of lists of readings.

        Returns:
            Tuple[bool, Dict[str, GestureMatch]]: A tuple containing:
                - A boolean indicating if the gesture is a match.
                - A dictionary where keys are sensor labels and values are GestureMatch instances,
                  informing which sensors didn't make the cut (with their score).

        Raises:
            Alert: If the models are not generated or if the lengths of timestamps and readings do not match.
        """
        Result: Dict[str, GestureMatch] = {}

        if self._models == {}:
            # print line where .is_gesture() was called
            alert("Models aren't generated. Read a file.", backtrack=2)
            return False, Result

        for sensor in readings.keys():
            if len(timestamps) != len(readings[sensor]):
                # print line where .is_gesture() was called
                alert("timestamps and readings should be of the same size.", backtrack=2)
                return False, Result

            values2d = np.array([[x, y] for x, y in zip(timestamps, readings[sensor])])

            # Compute average log‑likelihood for each model
            # .score = mean log‑likelihood
            scores = [float(model.score(values2d)) for model in self._models[sensor]]

            Result[sensor] = GestureMatch(
                value = max(scores), # would help with finding more appropriate threshold
                status = max(scores) > self._threshold # best‑matching orientation
            )

        return all(r.status for r in Result.values()), Result


    #- Getters & Setters ---------------------------------------------------------------------------

    # Get models read from/written to the gesture file
    def get_models(self) -> data_dict_t:
        """Return the models read from or written to the gesture file.

        Returns:
            data_dict_t: The models associated with the gesture file.
        """
        return self._models


    # Get saved parameters in ModelParameters type
    def get_parameters(self) -> ModelParameters:
        """Return the saved parameters as a ModelParameters instance.

        Returns:
            ModelParameters: The parameters associated with the gesture file.
        """
        return ModelParameters(
                random_state=self._random_state,
                threshold=self._threshold,
                n_components=self._n_components
            )


    # Get names of all sensors in the record
    def get_keys(self) -> List[str]:
        """Return a list of all sensor labels read from the gesture file, or set with create()"""
        return list(self._models.keys())


    # Set parameters, either override individual, or override all with ModelParameters arg
    def set_parameters(self,
            parameters: Optional[ModelParameters]=None,
            n_components: Optional[int]=None,
            random_state: Optional[int]=None,
            threshold:Optional[float]=None
        ) -> None:
        """Set parameters, either overriding individual values or all at once using a ModelParameters instance.

        Parameters:
            parameters: Parameters to apply.
            n_components: Number of components to set.
            random_state: Random state to set.
            threshold: Threshold value to set.
        """
        if parameters is not None:
            self._threshold = parameters.threshold
            assert parameters.n_components > 0, "n_components should be above 0"
            self._n_components = parameters.n_components
            assert parameters.random_state > 1, "random_state should be above 1"
            self._random_state = parameters.random_state

        if n_components is not None:
            assert n_components > 0, "n_components should be above 0"
            self._n_components = n_components

        if random_state is not None:
            assert random_state > 1, "random_state should be above 1"
            self._random_state = random_state

        if threshold is not None:
            self._threshold = threshold

