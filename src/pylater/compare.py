from __future__ import annotations

import typing

import xarray as xr

import arviz as az


def combine_multiple_likelihoods(
    idata: az.data.inference_data.InferenceData,
    combined_var_name: str = "obs",
    var_names : typing.Sequence[str] | None = None,
    overwrite: bool = False,
    combined_dim_name: str = "trial",
    copy_idata: bool = False,
) -> az.data.inference_data.InferenceData:
    """
    Combine likelihoods from multiple observations into a single variable.

    Parameters
    ----------
    idata
        Inference data object containing log-likelihood samples.
    combined_var_name
        Name to call the combined variable.
    var_names
        Variable names to include in the combination; if `None`, include all
        variables with a likelihood.
    overwrite
        Whether to overwrite `combined_var_name`, if it already exists.
    combined_dim_name
        Name of the combined dimension.
    copy_idata
        Whether to add the new variable to the provided `idata` or to a copy.

    Returns
    -------
    az.data.inference_data.InferenceData
        Inference data object with the new combined log-likelihood.
    """

    if not hasattr(idata, "log_likelihood"):
        raise ValueError(
            "No log-likelihood found in `idata`; use `pm.compute_log_likelihood()`"
        )

    if combined_var_name in idata.log_likelihood and not overwrite:
        msg = f"Variable {combined_var_name} already exists; either remove or set `overwrite=True`"
        raise ValueError(msg)

    ll_var_names = (
        list(var_names)
        if var_names is not None
        else [
            var_name
            for var_name in idata.log_likelihood
        ]
    )

    if len(ll_var_names) == 0:
        raise ValueError("No log-likelihood values found")

    modified_idata = (
        idata.copy()
        if copy_idata
        else idata
    )

    assert hasattr(modified_idata, "log_likelihood")

    modified_idata.log_likelihood[combined_var_name] = xr.concat(
        objs=[
            modified_idata.log_likelihood[ll_var_name].rename(
                {f"{ll_var_name}_dim_0": combined_dim_name}
            )
            for ll_var_name in ll_var_names
        ],
        dim=combined_dim_name,
    )

    return modified_idata
