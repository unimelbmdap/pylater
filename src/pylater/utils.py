import functools

import numpy as np

import xarray as xr

import arviz as az

import matplotlib.pyplot as plt
import matplotlib.figure


def plot_prior_predictive(
    idata: az.data.inference_data.InferenceData,
) -> matplotlib.figure.Figure:

    (figure, axes) = plt.subplots()


def _form_histogram(
    idata: az.data.inference_data.InferenceData
) -> xr.DataArray:

    dataset = az.extract(
        data=idata,
        group="prior",
        combined=True,
        keep_dataset=True,
    )

    data, = dataset.data_vars.values()

    rt_dim, = (dim_name for dim_name in data.dims if dim_name != "sample")

    # sniff the first sample to get the bins
    (_, bins) = np.histogram(
        a=data.isel(sample=0).values,
        bins="auto",
    )

    bin_delta = bins[1] - bins[0]
    bin_centres = bins[:-1] + bin_delta / 2

    def histogram(sample_data: xr.DataArray) -> xr.DataArray:

        (hist, _) = np.histogram(a=sample_data.values, bins=bins, density=True)

        return xr.DataArray(
            data=hist,
            coords={"bin_centre": bin_centres},
        )

    hist = data.groupby(group="sample", squeeze=False).map(histogram)

    return hist
