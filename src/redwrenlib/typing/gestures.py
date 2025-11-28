
# typing/gesturefile.py

#- Imports -----------------------------------------------------------------------------------------

from dataclasses import dataclass


#- Data Classes ------------------------------------------------------------------------------------

# Immutable container for model configuration and inputs used when creating gestures.
@dataclass(frozen=True)
class ModelParameters:
    random_state: int
    n_component: int
    threshold: float

