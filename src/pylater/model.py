
from __future__ import annotations

import enum
import typing
import warnings

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
    """
    Assemble a LATER model using a default set of priors.

    Parameters
    ----------
    datasets
        Observed data to model.
    share_type
        With multiple datasets, parameters can be shared according to a 'shift'
        or a 'swivel' arrangement.

    Returns
    -------
    pm.Model
        The PyMC model.

    Notes
    -----
    * This uses a default set of priors that may not be appropriate for a given use
    case. A warning is raised to highlight this; it can be silenced using the `warnings`
    built-in package.
    """

    warnings.warn(
        message="Note that this uses priors that may not be appropriate for your use case",
        stacklevel=2,
    )

    sharing = (
        ShareType(share_type)
        if share_type is not None
        else None
    )

    n_datasets = len(datasets)

    if n_datasets > 1 and sharing is None:
        raise ValueError(
            "With multiple datasets, must provide a `share_type` argument"
        )

    dataset_names = [dataset.name for dataset in datasets]

    if sharing is None:
        n_sigma = n_k = n_datasets
        sigma_dims = k_dims = "dataset"
    elif sharing is ShareType.SHIFT:
        n_sigma = 1
        n_k = n_datasets
        sigma_dims = "shared"
        k_dims = "dataset"
    elif sharing is ShareType.SWIVEL:
        n_sigma = n_datasets
        n_k = 1
        sigma_dims = "dataset"
        k_dims = "shared"

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
