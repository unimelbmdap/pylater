import collections
import csv
import functools
import importlib.resources
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import scipy.stats


class Dataset:
    __slots__ = ("name", "rt_s", "promptness", "ecdf_p", "ecdf_x")

    _T = typing.TypeVar("_T", bound=npt.NBitBase)

    def __init__(self, name: str, rt_s: npt.NDArray[np.number[_T]]) -> None:
        self.name = name
        self.rt_s = rt_s

        self.promptness = 1.0 / self.rt_s.astype(float)

        ecdf = scipy.stats.ecdf(sample=self.rt_s).cdf

        self.ecdf_p = 1 - ecdf.probabilities
        self.ecdf_x = ecdf.quantiles


@functools.lru_cache
def load_cw1995() -> dict[str, Dataset]:
    csv_dir = importlib.resources.files("pylater.resources")
    csv_path = pathlib.Path(str(csv_dir.joinpath("Carpenter_Williams_Nature_1995.csv")))

    temp_data: dict[str, list[float]] = collections.defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            name = "_".join([row["participant"], row["condition"]])
            temp_data[name].append(float(row["time"]) / 1000.0)

    return {
        name: Dataset(name=name, rt_s=np.array(rt_s))
        for (name, rt_s) in temp_data.items()
    }


def __getattr__(name: str) -> dict[str, Dataset]:
    if name == "cw1995":
        return load_cw1995()

    error_info = f"No known attribute named {name}"

    raise AttributeError(error_info)
