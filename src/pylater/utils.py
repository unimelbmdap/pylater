import functools

import numpy as np
import scipy.stats

import xarray as xr

import arviz as az

import matplotlib.pyplot as plt
import matplotlib.figure

import pylater.plot


def plot_prior_predictive(
    idata: az.data.inference_data.InferenceData,
    observed_variable_name: str = "obs",
) -> matplotlib.figure.Figure:

    ecdf = _form_ecdf(idata=idata)

    quantiles = ecdf.quantile(dim="sample", q=[0.025, 0.5, 0.975])

    (figure, axes) = pylater.plot.reciprobit_figure()

    for q in [0.025, 0.975]:
        axes.plot(ecdf.rt.values, quantiles.sel(quantile=q).values, color="grey")

    axes.plot(ecdf.rt.values, quantiles.sel(quantile=0.5).values, "k")


def _form_ecdf(
    idata: az.data.inference_data.InferenceData,
    observed_variable_name: str = "obs",
) -> xr.DataArray:

    dataset = az.extract(
        data=idata,
        group="prior",
        combined=True,
        keep_dataset=True,
        var_names=observed_variable_name,
    )

    data, = dataset.data_vars.values()

    min_rt_s = 51 / 1000
    max_rt_s = 2000 / 1000

    x_rt_s = np.logspace(np.log10(min_rt_s), np.log10(max_rt_s), 1001)

    def gen_ecdf(sample_data: xr.DataArray) -> xr.DataArray:

        ecdf = scipy.stats.ecdf(sample=np.squeeze(sample_data.values))

        ecdf_p = ecdf.cdf.evaluate(x=x_rt_s)

        return xr.DataArray(
            data=ecdf_p,
            coords={"rt": x_rt_s},
        )

    return data.groupby(group="sample", squeeze=False).map(gen_ecdf)
