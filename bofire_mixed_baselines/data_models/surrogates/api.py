from bofire.data_models.surrogates.api import AnySurrogate as BofireAnySurrogate

from bofire_mixed_baselines.data_models.surrogates.bart import (
    BARTSurrogate,  # noqa: F401
)

AnySurrogate = BofireAnySurrogate | BARTSurrogate
