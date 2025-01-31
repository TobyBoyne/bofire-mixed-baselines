import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from pandas import DataFrame

from bofire_mixed_baselines.domain_utils import build_integer_input



class DiscreteAckley(Benchmark):
    """
    adapted from: https://arxiv.org/pdf/2210.10199"""

    def __init__(self, discrete_dim=10, cont_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.dim = discrete_dim + cont_dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        build_integer_input(key=f"x_{i}", bounds=(0, 1))
                        for i in range(discrete_dim)
                    ),
                    *(
                        ContinuousInput(key=f"x_{i+discrete_dim}", bounds=(-1.0, 1.0))
                        for i in range(cont_dim)
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        x_int = X[self.domain.inputs.get_keys(includes=DiscreteInput)].to_numpy()
        x_cont = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        # map x_int from {0, 1} to {-1, 1}
        x_int = 2 * x_int - 1
        z = np.concatenate([x_int, x_cont], axis=1)
        a = 20.0
        b = 0.2
        c = 2 * np.pi
        d = self.dim
        y = (
            -a * np.exp(-b * np.sqrt(1 / d * np.sum(z**2, axis=1)))
            - np.exp(1 / d * np.sum(np.cos(c * z), axis=1))
            + a
            + np.exp(1)
        )
        return pd.DataFrame(data=y[:, None], columns=self.domain.outputs.get_keys())


class DiscreteRosenbrock(Benchmark):
    """Adapted from: https://arxiv.org/pdf/2210.10199"""

    def __init__(self, discrete_dim=6, cont_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = discrete_dim + cont_dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        build_integer_input(key=f"x_{i}", bounds=(-1, 2))
                        for i in range(discrete_dim)
                    ),
                    *(
                        ContinuousInput(key=f"x_{i+discrete_dim}", bounds=(-5.0, 10.0))
                        for i in range(cont_dim)
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        x_int = X[self.domain.inputs.get_keys(includes=DiscreteInput)].to_numpy()
        x_cont = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        # map x_int from [-1, 2] to [-5, 10]
        x_int = 5 * x_int
        z = np.concatenate([x_int, x_cont], axis=1)
        y = np.sum(
            100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1 - z[:, :-1]) ** 2, axis=1
        )
        return pd.DataFrame(data=y[:, None], columns=self.domain.outputs.get_keys())
