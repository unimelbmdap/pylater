import typing
import dataclasses

import numpy as np
import numpy.typing as npt

import scipy.stats


class Dataset:

    __slots__ = ("name", "rt_s", "promptness", "ecdf_p")

    _T = typing.TypeVar("_T", bound=npt.NBitBase)

    def __init__(self, name:str, rt_s: npt.NDArray[np.number[_T]]) -> None:
        self.name = name
        self.rt_s = rt_s

        self.promptness = 1.0 / self.rt_s.astype(float)

        self.ecdf_p = 1 - scipy.stats.ecdf(sample=self.rt_s).cdf.probabilities


@dataclasses.dataclass
class FitSettings:
    share_a: bool = False
    share_sigma: bool = False
    share_sigma_e: bool = False
    with_early_component: bool = False
    intercept_form: bool = False


class FitParams:

    def __init__(
        self,
        datasets: typing.Sequence[Dataset],
        fit_settings: FitSettings
    ) -> None:

        self.n_datasets = len(datasets)

        self.n_a = 1 if fit_settings.share_a else self.n_datasets
        self.n_sigma = 1 if fit_settings.share_sigma else self.n_datasets
        self.n_sigma_e = (
            0 if not fit_settings.with_early_component
            else (1 if fit_settings.share_sigma_e else self.n_datasets)
        )
        self.n_mu = self.n_datasets if fit_settings.intercept_form else self.n_a



def fit_data(
    data: Dataset | typing.Sequence[Dataset],
    fit_settings: FitSettings,
) -> None:

    datasets = (data,) if isinstance(data, Dataset) else data

    n_datasets = len(datasets)


