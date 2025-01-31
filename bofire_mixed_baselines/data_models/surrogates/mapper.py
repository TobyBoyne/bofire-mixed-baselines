from bofire.data_models.surrogates import api as surrogates_data_models
from bofire.surrogates.api import map as bofire_map_surrogate
from bofire.surrogates.surrogate import Surrogate

from bofire_mixed_baselines.data_models.surrogates import api as bmb_data_models
from bofire_mixed_baselines.surrogates.bart import BARTSurrogate

SURROGATE_MAP: dict[type[surrogates_data_models.Surrogate], type[Surrogate]] = {
    bmb_data_models.BARTSurrogate: BARTSurrogate,
}


def surrogate_map(
    data_model: surrogates_data_models.Surrogate,
) -> bmb_data_models.AnySurrogate:
    if data_model.__class__ not in SURROGATE_MAP:
        return bofire_map_surrogate(data_model)
    cls = SURROGATE_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
