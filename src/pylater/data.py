import typing

import scipy.stats

import numpy as np
import numpy.typing as npt


class Dataset:

    __slots__ = ("name", "rt_s", "promptness", "ecdf_p")

    _T = typing.TypeVar("_T", bound=npt.NBitBase)

    def __init__(self, name:str, rt_s: npt.NDArray[np.number[_T]]) -> None:
        self.name = name
        self.rt_s = rt_s

        self.promptness = 1.0 / self.rt_s.astype(float)

        self.ecdf_p = 1 - scipy.stats.ecdf(sample=self.rt_s).cdf.probabilities
