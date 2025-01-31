from bofire.data_models.strategies import api as strategies_data_models
from bofire.strategies.api import map as bofire_map_strategy

from bofire_mixed_baselines.data_models.strategies import api as bmb_data_models
from bofire_mixed_baselines.strategies.bart_grid import BARTGridStrategy
from bofire_mixed_baselines.strategies.relaxed_sobo import RelaxedSoboStrategy
from bofire_mixed_baselines.strategies.smac import SMACStrategy

STRATEGY_MAP = {
    bmb_data_models.SMACStrategy: SMACStrategy,
    bmb_data_models.RelaxedSoboStrategy: RelaxedSoboStrategy,
    bmb_data_models.BARTGridStrategy: BARTGridStrategy,
}


def strategy_map(
    data_model: strategies_data_models.Strategy,
) -> bmb_data_models.AnyStrategy:
    if data_model.__class__ not in STRATEGY_MAP:
        return bofire_map_strategy(data_model)
    cls = STRATEGY_MAP[data_model.__class__]
    return cls(data_model=data_model)
