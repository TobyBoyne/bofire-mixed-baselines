from abc import abstractmethod

import pandas as pd
from bofire.benchmarks.api import Benchmark


class BanditBenchmark(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = self._load_data(**kwargs)

        if kwargs.get("negate", False):
            output_keys = self.domain.outputs.get_keys()
            self.data[output_keys] = -self.data[output_keys]

    @abstractmethod
    def _load_data(self, **kwargs) -> pd.DataFrame:
        pass

    def _f(self, candidates):
        return self.data.loc[candidates.index]

    def get_optima(self):
        opt_idx = self.data[self.domain.outputs.get_keys()[0]].idxmin()
        return self.data.iloc[opt_idx]
