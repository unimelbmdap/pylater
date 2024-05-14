from __future__ import annotations

import enum
import typing

import arviz as az
import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms

import IPython.display

import numpy as np
import scipy.stats
import xarray as xr

import pylater.axes


class DataPlotType(enum.Enum):
    STEP = "step"
    SCATTER = "scatter"


class PredictiveDataType(enum.Enum):
    PRIOR = "prior"
    POSTERIOR = "posterior"


class ReciprobitPlot:

    style: typing.ClassVar[dict[str, list[str] | float | bool]] = {
        # font sizes
        "font.size": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        # other params
        "lines.linewidth": 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.frameon": False,
        "font.sans-serif": [
            "Arial",
            "Nimbus Sans",
            "Nimbus Sans L",
            "Helvetica",
        ],
    }

    def __init__(
        self,
        fig_ax: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None = None,
        min_rt_s: float = 75 / 1000,
        max_rt_s: float = 2000 / 1000,
        min_p: float = 0.0,
        max_p: float = 1.0,
        linthresh: float = 0.1 / 100,
        linscale: float = 0.05,
        axis_position_offset: float = 4,
        apply_style: bool = True,
    ) -> None:
        """
        Create a Matplotlib figure in reciprobit space.

        Parameters
        ----------
        fix_ax
            An already-created figure/axes pair.
        min_rt_s, max_rt_s
            Minimum and maxium reaction time values, in seconds, for the x axis.
        p_min, p_max
            Minimum and maximum probability values, for the y axis.
        apply_style
            Whether to apply a custom styling to the figure or leave the defaults.
        """

        self._linthresh = linthresh
        self._linscale = linscale
        self._axis_position_offset = axis_position_offset

        self._style = ReciprobitPlot.style if apply_style else {}

        with mpl.rc_context(rc=self.style):

            (self.fig, self._ax) = (
                fig_ax
                if fig_ax is not None
                else plt.subplots()
            )

            tick_locations_ms = np.array([50, 75, 100, 150, 200, 300, 500, 1000])

            self._ax.set_xticks(ticks=tick_locations_ms / 1000)
            self._ax.set_xscale(value="reciprobit_time")
            self._ax.set_xlabel(xlabel="Latency (ms)")

            self.min_rt_s = min_rt_s
            self.max_rt_s = max_rt_s

            self._ax_promptness = self._ax.secondary_xaxis(location="top")
            self._ax_promptness.set_xscale(
                value="reciprobit_time",
                axis_type=pylater.axes.AxisType.PROMPTNESS,
            )
            self._ax_promptness.set_xticks(ticks=tick_locations_ms / 1000)
            self._ax_promptness.set_xlabel(xlabel="Promptness (1/s)")

            y_ticks = (
                np.array(
                    [0.0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98, 99, 99.5, 99.9, 100.0]
                )
                / 100
            )

            self.min_p = min_p
            self.max_p = max_p

            self._ax.set_yscale(
                value="probit",
                linthresh=self._linthresh,
                linscale=self._linscale,
            )
            self._ax.set_yticks(ticks=y_ticks)
            self._ax.set_ylabel(ylabel="Cumulative probability (%)")

            for spine in (
                *tuple(self._ax.spines.values()),
                *tuple(self._ax_promptness.spines.values()),
            ):
                spine.set_position(("outward", self._axis_position_offset))

    def plot_data(
        self,
        data: pylater.data.Dataset,
        plot_type: str = "step",
        n_points: int = 1000,
        **kwargs: str | float,
    ) -> ReciprobitPlot:

        data_plot_type = DataPlotType(plot_type)

        if "label" not in kwargs:
            kwargs["label"] = data.name

        if data_plot_type is DataPlotType.STEP:

            x_rt_s = np.logspace(
                np.log10(self.min_rt_s),
                np.log10(self.max_rt_s),
                n_points,
            )

            trial_ecdf_p = np.clip(
                data.ecdf.cdf.evaluate(x_rt_s),
                0.0001,
                1 - 0.0001,
            )

            with mpl.rc_context(rc=self.style):
                self._ax.step(
                    x_rt_s,
                    trial_ecdf_p,
                    **kwargs,
                )

        elif data_plot_type is DataPlotType.SCATTER:

            with mpl.rc_context(rc=self.style):
                paths = self._ax.scatter(
                    data.ecdf_x,
                    data.ecdf_p,
                    **kwargs,
                )

        return self

    def plot_model(
        self,
        idata: az.data.inference_data.InferenceData,
        n_points: int = 1000,
        **kwargs: str | float,
    ) -> ReciprobitPlot:

        dataset: xr.Dataset = az.extract(
            data=idata,
            group="posterior",
            combined=True,
        )

        dataset = dataset.expand_dims(dim="x", axis=0)

        x_rt_s = np.logspace(
            np.log10(self.min_rt_s),
            np.log10(self.max_rt_s),
            n_points,
        )

        log_p = pylater.dist.logcdf(
            value=1 / x_rt_s[:, np.newaxis],
            mu=dataset.mu.values,
            sigma=dataset.sigma.values,
            sigma_e=dataset.sigma_e.values,
        ).eval()

        p = 1 - np.exp(log_p)

        quantiles = np.quantile(p, q=[0.5, 0.025, 0.975], axis=1)

        quantiles = np.clip(quantiles, 0.0001, 1 - 0.0001)

        with mpl.rc_context(rc=self.style):

            if "alpha" not in kwargs:
                kwargs["alpha"] = 0.5

            self._ax.fill_between(
                x_rt_s,
                quantiles[1, :],
                quantiles[2, :],
                **kwargs,
            )

            self._ax.plot(
                x_rt_s,
                quantiles[0, :],
            )

        return self

    def plot_predictive(
        self,
        idata: az.data.inference_data.InferenceData,
        predictive_type: str,
        observed_variable_name: str = "obs",
        n_points: int = 1000,
        **kwargs: str | float,
    ) -> ReciprobitPlot:

        pred_type = PredictiveDataType(predictive_type)

        x_rt_s = np.logspace(
            np.log10(self.min_rt_s),
            np.log10(self.max_rt_s),
            n_points,
        )

        group = pred_type.value + "_predictive"

        dataset: xr.Dataset = az.extract(
            data=idata,
            group=group,
            combined=True,
            keep_dataset=True,
            var_names=observed_variable_name,
        )

        (data,) = dataset.data_vars.values()

        def gen_ecdf(sample_data: xr.DataArray) -> xr.DataArray:
            ecdf = scipy.stats.ecdf(sample=np.squeeze(sample_data.values))

            ecdf_p = ecdf.cdf.evaluate(x=x_rt_s)

            return xr.DataArray(
                data=ecdf_p,
                coords={"rt": x_rt_s},
            )

        da = data.groupby(
            group="sample", squeeze=False
        ).map(gen_ecdf)

        quantiles = da.quantile(dim="sample", q=[0.025, 0.5, 0.975])

        quantiles = quantiles.clip(0.0001, 1 - 0.0001)

        with mpl.rc_context(rc=self.style):

            if "alpha" not in kwargs:
                kwargs["alpha"] = 0.5

            self._ax.fill_between(
                da.rt.values,
                quantiles.sel(quantile=0.025).values,
                quantiles.sel(quantile=0.975).values,
                **kwargs,
            )

            self._ax.plot(
                da.rt.values,
                quantiles.sel(quantile=0.5).values,
            )

        return self

    @property
    def min_rt_s(self) -> float:
        return self._min_rt_s

    @min_rt_s.setter
    def min_rt_s(self, value: float) -> None:
        self._min_rt_s = value
        self._ax.set_xlim(xmin=value)

    @property
    def max_rt_s(self) -> float:
        return self._max_rt_s

    @max_rt_s.setter
    def max_rt_s(self, value: float) -> None:
        self._max_rt_s = value
        self._ax.set_xlim(xmax=value)

    @property
    def min_p(self) -> float:
        return self._min_p

    @min_p.setter
    def min_p(self, value: float) -> None:
        self._min_p = value
        self._ax.set_ylim(ymin=value)

    @property
    def max_p(self) -> float:
        return self._max_p

    @max_p.setter
    def max_p(self, value: float) -> None:
        self._max_p = value
        self._ax.set_ylim(ymax=value)
