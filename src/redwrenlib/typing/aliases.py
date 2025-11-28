
# typing/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

from typing import List
from sklearn.mixture import GaussianMixture


#- Aliases -----------------------------------------------------------------------------------------

data_dict_t = dict[str, List[GaussianMixture]]
float2d_t = List[List[float]]
float3d_t = List[List[List[float]]]
int2d_t = List[List[int]]

