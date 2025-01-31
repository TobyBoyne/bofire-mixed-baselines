from bofire.data_models.strategies.api import AnyStrategy as BofireAnyStrategy

from bofire_mixed_baselines.data_models.strategies.bart_grid import (
    BARTGridStrategy,  # noqa: F401
)
from bofire_mixed_baselines.data_models.strategies.relaxed_sobo import (
    RelaxedSoboStrategy,  # noqa: F401
)
from bofire_mixed_baselines.data_models.strategies.smac import (
    SMACStrategy,  # noqa: F401
)

AnyStrategy = BofireAnyStrategy | SMACStrategy | RelaxedSoboStrategy | BARTGridStrategy
