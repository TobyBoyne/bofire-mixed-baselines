from bofire.data_models.features.api import DiscreteInput
from typing import Literal
from bofire.data_models.features.api import CategoricalInput, ContinuousInput, AnyFeature

def build_integer_input(*, key: str, unit: str | None = None, bounds: tuple[int, int]):
    lb, ub = bounds
    values = list(range(lb, ub + 1))
    return DiscreteInput(key=key, unit=unit, values=values)

def get_feature_bounds(
    feature: AnyFeature, encoding: Literal["bitmask", "ordinal"] | None = None
) -> tuple[float, float] | list[str] | list[int]:
    if isinstance(feature, CategoricalInput):
        cats = feature.categories
        if encoding == "bitmask":
            bitmask_ub = (1 << len(cats)) - 1
            return (0, bitmask_ub)
        elif encoding == "ordinal":
            return list(range(len(cats)))
        return cats
    elif isinstance(feature, DiscreteInput):
        return (feature.lower_bound, feature.upper_bound)
    elif isinstance(feature, ContinuousInput):
        return feature.bounds

    raise TypeError(f"Cannot get bounds for feature of type {feature.type}")