import collections
import csv
import functools
import importlib.resources
import pathlib

import numpy as np
import numpy.typing as npt

import scipy.stats


class Dataset:
    __slots__ = ("name", "rt_s", "promptness", "ecdf", "ecdf_p", "ecdf_x")

    def __init__(
        self,
        name: str,
        rt_s: npt.ArrayLike,
    ) -> None:
        """
        Create a dataset from observed reaction times.

        Parameters
        ----------
        name
            Name of the dataset.
        rt_s
            Reaction times, in seconds.

        Returns
        -------
        Dataset
            The dataset.
        """

        self.name = name
        self.rt_s: npt.NDArray[np.float64] = np.array(rt_s, dtype=np.float64)

        self.promptness = 1.0 / self.rt_s

        self.ecdf = scipy.stats.ecdf(sample=self.rt_s)

        self.ecdf_p = self.ecdf.cdf.probabilities
        self.ecdf_x = self.ecdf.cdf.quantiles

    def __repr__(self) -> str:
        return f"Dataset named '{self.name}' with {len(self.rt_s)} data points"


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
