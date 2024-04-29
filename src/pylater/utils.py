import arviz as az
import matplotlib as mpl
import matplotlib.figure
import numpy as np
import scipy.stats
import xarray as xr

import pylater.plot


def plot_predictive(
    idata: az.data.inference_data.InferenceData,
    group: str,
    observed_variable_name: str = "obs",
    with_observed: bool = False,
    with_p50_line: bool = False,
    min_rt_s: float = 50 / 1000.0,
    max_rt_s: float = 2000 / 1000.0,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    ecdf = _form_ecdf(
        idata=idata,
        group=group,
        observed_variable_name=observed_variable_name,
        min_rt_s=min_rt_s,
        max_rt_s=max_rt_s,
    )

    quantiles = ecdf.quantile(dim="sample", q=[0.025, 0.5, 0.975])

    quantiles = quantiles.clip(0.0001, 1 - 0.0001)

    (figure, axes) = pylater.plot.reciprobit_figure(
        min_rt_s=min_rt_s,
        max_rt_s=max_rt_s,
    )

    style = pylater.plot.get_mpl_defaults()

    with mpl.rc_context(rc=style):

        axes.fill_between(
            ecdf.rt.values,
            quantiles.sel(quantile=0.025).values,
            quantiles.sel(quantile=0.975).values,
            color="lightgrey",
            label="95% credible interval",
        )

        axes.plot(
            ecdf.rt.values,
            quantiles.sel(quantile=0.5).values,
            "grey",
            label="Median",
        )

        if with_p50_line:
            # add 50%
            axes.plot(
                [min_rt_s, max_rt_s],
                [0.5] * 2,
                color="grey",
                linestyle="--",
                alpha=0.5,
            )

        if with_observed:
            trial_data = 1 / idata.observed_data[observed_variable_name]
            trial_ecdf = scipy.stats.ecdf(sample=trial_data.values)

            x_rt_s = np.logspace(np.log10(min_rt_s), np.log10(max_rt_s), 1001)
            trial_ecdf_p = np.clip(
                trial_ecdf.cdf.evaluate(x_rt_s),
                0.0001,
                1 - 0.0001,
            )

            #trial_ecdf.cdf.plot(ax=axes, label="Observed data", color="k")
            axes.step(
                x_rt_s,
                trial_ecdf_p,
                label="Observed data",
                color="k",
            )

        axes.legend()

    return (figure, axes)


def _form_ecdf(
    idata: az.data.inference_data.InferenceData,
    group: str,
    observed_variable_name: str = "obs",
    min_rt_s: float = 50 / 1000.0,
    max_rt_s: float = 2000 / 1000.0
) -> xr.DataArray:
    dataset: xr.Dataset = az.extract(
        data=idata,
        group=group,
        combined=True,
        keep_dataset=True,
        var_names=observed_variable_name,
    )

    (data,) = dataset.data_vars.values()

    x_rt_s = np.logspace(np.log10(min_rt_s), np.log10(max_rt_s), 1001)

    def gen_ecdf(sample_data: xr.DataArray) -> xr.DataArray:
        ecdf = scipy.stats.ecdf(sample=np.squeeze(sample_data.values))

        ecdf_p = ecdf.cdf.evaluate(x=x_rt_s)

        return xr.DataArray(
            data=ecdf_p,
            coords={"rt": x_rt_s},
        )

    ecdf_da = data.groupby(group="sample", squeeze=False).map(gen_ecdf)

    return ecdf_da
