import typing

import xarray as xr

import arviz as az


def combine_multiple_likelihoods(
    idata: az.data.inference_data.InferenceData,
    obs_var_name: str = "obs",
    var_names : typing.Sequence[str] | None = None,
    overwrite: bool = False,
    combined_dim_name: str = "trial",
    copy_idata: bool = False,
) -> az.data.inference_data.InferenceData:

    if obs_var_name in idata.log_likelihood and not overwrite:
        msg = f"Variable {obs_var_name} already exists; either remove or set `overwrite=True`"
        raise ValueError(msg)

    ll_var_names = (
        list(var_names)
        if var_names is not None
        else [
            var_name
            for var_name in idata.log_likelihood
            if var_name.startswith(obs_var_name + "_")
        ]
    )

    if len(ll_var_names) == 0:
        msg = f"No log-likelihood values found starting with {obs_var_name}_"
        raise ValueError(msg)

    modified_idata = (
        idata.copy(deep=True)
        if copy_idata
        else idata
    )

    modified_idata.log_likelihood[obs_var_name] = xr.concat(
        objs=[
            modified_idata.log_likelihood[ll_var_name].rename(
                {f"{ll_var_name}_dim_0": combined_dim_name}
            )
            for ll_var_name in ll_var_names
        ],
        dim=combined_dim_name,
    )

    return modified_idata
