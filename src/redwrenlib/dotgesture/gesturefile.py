
# dotgesture/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

import h5py
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture

from .version_reads import version_readers, GESTURE_VERSION
from ..utils.debug import alert
from ..typing import (
    GestureMatch, SensorData,
    data_dict_t, numeric_t,
)


#- GestureFile Class -------------------------------------------------------------------------------

class GestureFile:

    # Initialise the instance with default values
    def __init__(self, filename: str):
        """Initialise the GestureFile instance with the given filename."""
        self._filename: str = filename
        self._gesture_data: data_dict_t = {}


    # Create mew gesture file
    def create(self) -> bool:
        """Create a new gesture file and save file version.

        Returns:
            bool: True if the file was created successfully, False if there was an error.
        """
        try:
            with h5py.File(self._filename, 'w') as f:
                f.create_dataset('version', data=GESTURE_VERSION)

        except Exception as e:
            alert(f"Unable to create file: {e}")
            return False

        return True


    # Append to the record
    def append_reading(self, label: str, model: List[GaussianMixture]) -> bool:
        """"""
        try:
            if label not in self._gesture_data.keys():
                self._gesture_data[label] = SensorData()

            self._gesture_data[label].models += model

        except Exception as e:
            alert(f"Unable to append readings: {e}")
            return False

        return True


    # Add models for a sensor to the file
    def write(self, override: bool = False) -> bool:
        """Add models for a sensor to the gesture file with the specified label.

        Returns:
            bool: True if the models were appended successfully, False if there was an error.
        """
        if self._gesture_data == {}:
            return False

        file_open_as = 'w' if override else 'a'

        try:
            with h5py.File(self._filename, file_open_as) as f:
                for model in self._gesture_data.keys():
                    # if group exists, then append to it, else create a new group
                    if model in f:
                        gmm_group = f[model]
                        model_start_index = len(gmm_group)
                    else:
                        gmm_group = f.create_group(model) # create a new group for each sensor
                        model_start_index = 0

                    def _user_variables(label: str, value: numeric_t):
                        if label in gmm_group.keys():
                            gmm_group[label][...] = value
                        else:
                            gmm_group.create_dataset(label, data=value)

                    _user_variables('n_components', self._gesture_data[model].n_components)
                    _user_variables('random_state', self._gesture_data[model].random_state)
                    _user_variables('threshold', self._gesture_data[model].threshold)

                    for i, data in enumerate(self._gesture_data[model].models):
                        group = gmm_group.create_group(f'model_{model_start_index + i}')
                        group.create_dataset('weights', data=data.weights_)
                        group.create_dataset('means', data=data.means_)
                        group.create_dataset('covariances', data=data.covariances_)
                        group.create_dataset('precisions_cholesky', data=data.precisions_cholesky_)
                        group.create_dataset('n_components', data=data.n_components)

        except Exception as e:
            alert(f"Unable to write readings: {e}")
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
                    self._gesture_data = version_readers[file_version](f)

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

        if self._gesture_data == {}:
            # print line where .is_gesture() was called, backtrack=2
            alert("Models aren't generated. Read a file.", backtrack=2)
            return False, Result

        for sensor in readings.keys():
            if len(timestamps) != len(readings[sensor]):
                alert("timestamps and readings should be of the same size.", backtrack=2)
                return False, Result

            if sensor not in self._gesture_data.keys():
                alert(f"\"{sensor}\" is not a valid key in the read record.", backtrack=2)
                return False, Result

            values2d = np.array([[x, y] for x, y in zip(timestamps, readings[sensor])])

            # Compute average log‑likelihood for each model
            # .score = mean log‑likelihood
            scores = [float(model.score(values2d)) for model in self._gesture_data[sensor].models]

            Result[sensor] = GestureMatch(
                value = max(scores), # would help with finding more appropriate threshold
                status = max(scores) > self._gesture_data[sensor].threshold # best‑matching orientation
            )

        return all(r.status for r in Result.values()), Result


    #- Getters & Setters ---------------------------------------------------------------------------

    # Get models read from/written to the gesture file
    def get_gesture_data(self) -> data_dict_t:
        """Return the models read from or written to the gesture file.

        Returns:
            data_dict_t: The models associated with the gesture file.
        """
        return self._gesture_data


    # Get saved parameters in ModelParameters type
    def get_parameters(self, label: str) -> Dict[str, numeric_t]:
        """Return the saved parameters as a ModelParameters instance.

        Returns:
            ModelParameters: The parameters associated with the gesture file.
        """
        return {
            "n_components": self._gesture_data[label].n_components,
            "random_state": self._gesture_data[label].random_state,
            "threshold": self._gesture_data[label].threshold
        }


    # Get names of all sensors in the record
    def get_keys(self) -> List[str]:
        """Return a list of all sensor labels read from the gesture file, or set with create()"""
        return list(self._gesture_data.keys())


    # Set parameters, either override individual, or override all with ModelParameters arg
    def set_parameters(self,
            label: str,
            n_components: Optional[int] = None,
            random_state: Optional[int] = None,
            threshold:Optional[float] = None
        ) -> None:
        """Set parameters: overriding individual values.

        Parameters:
            n_components: Number of components to set.
            random_state: Random state to set.
            threshold: Threshold value to set.
        """
        if n_components is not None:
            assert n_components > 0, "n_components should be above 0"
            self._gesture_data[label].n_components = n_components

        if random_state is not None:
            assert random_state > 1, "random_state should be above 1"
            self._gesture_data[label].random_state = random_state

        if threshold is not None:
            self._gesture_data[label].threshold = threshold

