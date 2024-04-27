import arviz as az
import matplotlib.figure
import numpy as np
import scipy.stats
import xarray as xr

import pylater.plot


def plot_prior_predictive(
    idata: az.data.inference_data.InferenceData,
    observed_variable_name: str = "obs",
    min_rt_s: float = 50 / 1000.0,
    max_rt_s: float = 2000 / 1000.0,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    ecdf = _form_ecdf(
        idata=idata,
        observed_variable_name=observed_variable_name,
        min_rt_s=min_rt_s,
        max_rt_s=max_rt_s,
    )

    quantiles = ecdf.quantile(dim="sample", q=[0.025, 0.5, 0.975])

    (figure, axes) = pylater.plot.reciprobit_figure()

    axes.fill_between(
        ecdf.rt.values,
        quantiles.sel(quantile=0.025).values,
        quantiles.sel(quantile=0.975).values,
        color="lightgrey",
        label="95% credible interval",
    )

    axes.plot(ecdf.rt.values, quantiles.sel(quantile=0.5).values, "k", label="Median")

    # add 50%
    axes.plot(
        [min_rt_s, max_rt_s],
        [0.5] * 2,
        color="grey",
        linestyle="--",
        alpha=0.5,
    )

    axes.legend()

    return (figure, axes)


def _form_ecdf(
    idata: az.data.inference_data.InferenceData,
    observed_variable_name: str = "obs",
    min_rt_s: float = 50 / 1000.0,
    max_rt_s: float = 2000 / 1000.0
) -> xr.DataArray:
    dataset: xr.Dataset = az.extract(
        data=idata,
        group="prior",
        combined=True,
        keep_dataset=True,
        var_names=observed_variable_name,
    )

    (data,) = dataset.data_vars.values()

    x_rt_s = np.logspace(np.log10(min_rt_s), np.log10(max_rt_s), 101)

    def gen_ecdf(sample_data: xr.DataArray) -> xr.DataArray:
        ecdf = scipy.stats.ecdf(sample=np.squeeze(sample_data.values))

        ecdf_p = ecdf.cdf.evaluate(x=x_rt_s)

        return xr.DataArray(
            data=ecdf_p,
            coords={"rt": x_rt_s},
        )

    ecdf_da = data.groupby(group="sample", squeeze=False).map(gen_ecdf)

    return ecdf_da.where(np.logical_and(ecdf_da > 0, ecdf_da < 1))
