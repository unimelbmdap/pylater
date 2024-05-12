from __future__ import annotations

import enum
import typing

import numpy as np

import pymc as pm

import pylater.data
import pylater.dist


class ShareType(enum.Enum):
    SHIFT = "shift"
    SWIVEL = "swivel"


def build_default_model(
    datasets: typing.Sequence[pylater.data.Dataset],
    share_type: str | None = None,
) -> pm.Model:

    share_type = (
        ShareType(share_type)
        if share_type is not None
        else None
    )

    n_datasets = len(datasets)

    dataset_names = [dataset.name for dataset in datasets]

    if share_type is ShareType.SHIFT:
        n_sigma = 1
        n_k = n_datasets
        sigma_dims = "shared"
        k_dims = "dataset"
    elif share_type is ShareType.SWIVEL:
        n_sigma = n_datasets
        n_k = 1
        sigma_dims = "dataset"
        k_dims = "shared"
    else:
        assert share_type is None
        n_sigma = n_k = n_datasets
        sigma_dims = k_dims = "dataset"

    with pm.Model(
        coords={
            "dataset": dataset_names,
            "shared": ("shared",),
        },
    ) as model:

        # 95% CI of [0.375, 1.5]
        sigma = pm.LogNormal(
            "sigma",
            mu=np.log(0.75),
            sigma=np.log(2) / 2,
            dims=sigma_dims,
        )

        sigma_all = pm.math.pt.repeat(
            x=sigma,
            repeats=n_datasets - n_sigma + 1,
        )

        # 95% CI of [2.5, 10]
        k = pm.LogNormal(
            "k",
            mu=np.log(5),
            sigma=np.log(2) / 2,
            dims=k_dims,
        )

        k_all = pm.math.pt.repeat(
            x=k,
            repeats=n_datasets - n_k + 1,
        )

        # 95% CI of [2, 8]
        sigma_e_mod = pm.LogNormal(
            "sigma_e_mod",
            mu=np.log(4),
            sigma=np.log(2) / 2,
            dims="dataset",
        )

        mu = pm.Deterministic(
            "mu",
            sigma_all * k_all,
            dims="dataset",
        )

        sigma_e = pm.Deterministic(
            "sigma_e",
            sigma_all * sigma_e_mod,
            dims="dataset",
        )

        for (i_dataset, dataset) in enumerate(datasets):

            pylater.LATER(
                name=f"obs_{dataset.name}",
                mu=mu[i_dataset],
                sigma=sigma_all[i_dataset],
                sigma_e=sigma_e[i_dataset],
                observed_rt_s=dataset.rt_s,
            )

    return model


def demo():
    data = pylater.data.cw1995["a_p95"]

    with pm.Model(check_bounds=False) as model:
        mu = pm.Normal("mu", mu=3, sigma=1.5)
        sigma = pm.HalfNormal("sigma", sigma=3)
        sigma_e = pm.HalfNormal("sigma_e", sigma=5)

        pylater.dist.LATER(
            name="obs",
            mu=mu,
            sigma=sigma,
            sigma_e=sigma_e,
            observed_rt_s=data.rt_s,
        )

    return model


def demo_shared():
    data = pylater.data.cw1995

    data = {key: value for (key, value) in data.items() if key.startswith("b")}

    with pm.Model(check_bounds=False) as model:
        mu = pm.Normal("mu", mu=3, sigma=1.5, size=len(data))
        sigma = pm.HalfNormal("sigma", sigma=3)
        sigma_e = pm.HalfNormal("sigma_e", sigma=5, size=len(data))

        for i, (key, value) in enumerate(data.items()):
            pylater.dist.model(
                name=f"obs_{key}",
                mu=mu[i],
                sigma=sigma,
                sigma_e=sigma_e[i],
                observed=value.promptness,
            )

    return model
