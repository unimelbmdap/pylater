import dataclasses
import typing

import scipy.stats

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Dataset():
    name: str
    rt_s: npt.NDArray[np.number]  # type: ignore
    promptness: npt.NDArray[np.floating] = dataclasses.field(init=False)  # type: ignore
    ecdf: npt.NDArray[np.floating] = dataclasses.field(init=False)  # type: ignore

    def __post_init__(self) -> None:
        self.promptness = 1.0 / self.rt_s.astype(float)


class D:

    _T = typing.TypeVar("_T", bound=npt.NBitBase)

    def __init__(self, name:str, rt_s: npt.NDArray[np.number[_T]]) -> None:
        self.name = name
        self.rt_s = rt_s

def form_ecdf(dataset: Dataset) -> npt.NDArray[np.floating]:

    result = scipy.stats.ecdf(sample=dataset.rt_s)

    ecdf_p: npt.NDArray[np.floating] = 1 - result.cdf.probabilities

    return ecdf_p
