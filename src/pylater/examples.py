import collections
import csv
import importlib.resources
import pathlib
import functools

import numpy as np

import pylater.data


@functools.lru_cache
def load_cw1995() -> dict[str, pylater.data.Dataset]:

    csv_dir = importlib.resources.files("pylater.resources")
    csv_path = pathlib.Path(str(csv_dir.joinpath("Carpenter_Williams_Nature_1995.csv")))

    temp_data: dict[str, list[float]] = collections.defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            name = "_".join([row["participant"], row["condition"]])
            temp_data[name].append(float(row["time"]) / 1000.0)

    datasets = {
        name: pylater.data.Dataset(name=name, rt_s=np.array(rt_s))
        for (name, rt_s) in temp_data.items()
    }

    return datasets


def __getattr__(name:str) -> dict[str, pylater.data.Dataset]:

    if name == "cw1995":
        return load_cw1995()

    raise AttributeError(f"No known attribute named {name}")
