
# readfile/0_0_1.py

#- Imports -----------------------------------------------------------------------------------------

import h5py
from typing import Any, List, Optional, Tuple

from sklearn.mixture import GaussianMixture

from ...utils.debug import alert
from ...typing import (
    ModelParameters,
    data_dict_t,
)

#- Read Metho --------------------------------------------------------------------------------------

def read_file(f: h5py.File) -> Tuple[ModelParameters, data_dict_t]:
    try:
        models_dict: data_dict_t = {}

        model_parameters: ModelParameters = ModelParameters(
            threshold=f['threshold'][()],
            n_components=f['n_components'][()],
            random_state=f['random_state'][()]
        )

        for name in f.keys():
            models_dict[name] = []
            gmm_group = f[name]

            if isinstance(gmm_group, h5py.Group):
                for model_name in gmm_group.keys():
                    model_group = gmm_group[model_name]

                    weights = model_group['weights'][()]
                    means = model_group['means'][()]
                    covariances = model_group['covariances'][()]
                    precisions_cholesky = model_group['precisions_cholesky'][()]
                    n_components = model_group['n_components'][()]

                    model_instance = GaussianMixture(n_components=n_components)
                    model_instance.weights_ = weights
                    model_instance.means_ = means
                    model_instance.covariances_ = covariances
                    model_instance.precisions_cholesky_ = precisions_cholesky

                    models_dict[name].append(model_instance)

    except Exception as e:
        alert(f"Unable to parse file. {e}")

    return model_parameters, models_dict

