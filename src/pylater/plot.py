from __future__ import annotations

import enum
import typing

import numpy as np

import scipy.stats

import xarray as xr

import arviz as az

import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms

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
        linthresh
            The y-axis is on a probit scale between `linthresh` and `1 - linthresh`,
            and a linear scale otherwise.
        linscale
            The probit range of the y-axis covers `1 - 2 * linscale` of the total range.
        axis_position_offset
            The amount, in points, to offset the axes from the plot region.
        apply_style
            Whether to apply a custom styling to the figure or leave the defaults.
        """

        self._linthresh = linthresh
        self._linscale = linscale
        self.axis_position_offset = axis_position_offset

        self._style = ReciprobitPlot.style if apply_style else {}

        with mpl.rc_context(rc=self.style):

            (self.fig, self.ax) = (
                fig_ax
                if fig_ax is not None
                else plt.subplots()
            )

            tick_locations_ms = np.array([50, 75, 100, 150, 200, 300, 500, 1000])

            self.ax.set_xticks(ticks=tick_locations_ms / 1000)
            self.ax.set_xscale(value="reciprobit_time")
            self.ax.set_xlabel(xlabel="Latency (ms)")

            self.min_rt_s = min_rt_s
            self.max_rt_s = max_rt_s

            self.ax_promptness = self.ax.secondary_xaxis(location="top")
            self.ax_promptness.set_xscale(
                value="reciprobit_time",
                axis_type=pylater.axes.AxisType.PROMPTNESS,
            )
            self.ax_promptness.set_xticks(ticks=tick_locations_ms / 1000)
            self.ax_promptness.set_xlabel(xlabel="Promptness (1/s)")

            y_ticks = (
                np.array(
                    [0.0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98, 99, 99.5, 99.9, 100.0]
                )
                / 100
            )

            self.min_p = min_p
            self.max_p = max_p

            self.ax.set_yscale(
                value="probit",
                linthresh=self._linthresh,
                linscale=self._linscale,
            )
            self.ax.set_yticks(ticks=y_ticks)
            self.ax.set_ylabel(ylabel="Cumulative probability (%)")

            for spine in (
                *tuple(self.ax.spines.values()),
                *tuple(self.ax_promptness.spines.values()),
            ):
                spine.set_position(("outward", self.axis_position_offset))

    def plot_data(
        self,
        data: pylater.data.Dataset,
        plot_type: str = "step",
        n_points: int = 1000,
        **kwargs: str | float,
    ) -> ReciprobitPlot:
        """
        Plot an ECDF of data observations.

        Parameters
        ----------
        data
            Dataset containing the observations.
        plot_type
            Plots the data as a 'step' plot (`step`) or as individual
            points (`scatter`).
        n_points
            For 'step' plots, how many points to use when evaluating the ECDF.
        **kwargs
            Any additional keyword arguments are passed directly to `plt.step` or
            `plt.scatter`.

        Returns
        -------
        ReciprobitPlot
            The `ReciprobitPlot` instance.
        """

        data_plot_type = DataPlotType(plot_type)

        if "label" not in kwargs:
            kwargs["label"] = data.name

        if "color" not in kwargs and "c" not in kwargs:
            kwargs["c"] = "black"

        if data_plot_type is DataPlotType.STEP:

            x_rt_s = np.logspace(
                np.log10(self.min_rt_s),
                np.log10(self.max_rt_s),
                n_points,
            )

            trial_ecdf_p = data.ecdf.cdf.evaluate(x_rt_s)

            with mpl.rc_context(rc=self.style):
                self.ax.step(
                    x_rt_s,
                    trial_ecdf_p,
                    clip_on=False,
                    **kwargs,
                )

        elif data_plot_type is DataPlotType.SCATTER:

            with mpl.rc_context(rc=self.style):
                self.ax.scatter(
                    data.ecdf_x,
                    data.ecdf_p,
                    clip_on=False,
                    **kwargs,
                )

        with mpl.rc_context(rc=self.style):
            plt.legend()

        return self

    def plot_model(
        self,
        idata: az.data.inference_data.InferenceData,
        n_points: int = 1000,
        ci_range: float = 0.95,
        dataset_name: str | None = None,
        fill_kwargs: dict[str, typing.Any] | None = None,
        line_kwargs: dict[str, typing.Any] | None = None,
    ) -> ReciprobitPlot:
        """
        Plot a summary of model evaluations using parameters from a posterior
        distribution.

        Parameters
        ----------
        idata
            Inference data object containing posterior samples.
        n_points
            How many points to use when evaluating the model.
        ci_range
            Width of the credible interval.
        dataset_name
            Name of the dataset to plot. This is used to select a coordinate
            within a `dataset` dimension in the posterior samples. If `None`,
            assumes that there is only a single coordinate in the `dataset`
            dimension.
        fill_kwargs
            Keyword arguments passed directly to `plt.fill_between`, for the
            credible interval.
        line_kwargs
            Keyword arguments passed directly to `plt.line`, for the median.

        Returns
        -------
        ReciprobitPlot
            The `ReciprobitPlot` instance.
        """

        if fill_kwargs is None:
            fill_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        (lower_q, upper_q) = q_from_ci_range(ci_range=ci_range)

        dataset: xr.Dataset = az.extract(
            data=idata,
            group="posterior",
            combined=True,
        )

        subset = (
            dataset.sel(dataset=dataset_name)
            if dataset_name is not None
            else dataset.squeeze()
        ).expand_dims(dim="x", axis=0)

        x_rt_s = np.logspace(
            np.log10(self.min_rt_s),
            np.log10(self.max_rt_s),
            n_points,
        )

        log_p = pylater.dist.logcdf(
            value=1 / x_rt_s[:, np.newaxis],
            mu=subset.mu.values,
            sigma=subset.sigma.values,
            sigma_e=subset.sigma_e.values,
        ).eval()

        p = 1 - np.exp(log_p)

        quantiles = np.quantile(p, q=[0.5, lower_q, upper_q], axis=1)

        with mpl.rc_context(rc=self.style):

            if "alpha" not in fill_kwargs:
                fill_kwargs["alpha"] = 0.5

            if "label" not in fill_kwargs:
                fill_kwargs["label"] = f"{ci_range:.0%} credible interval"

            self.ax.fill_between(
                x_rt_s,
                quantiles[1, :],
                quantiles[2, :],
                clip_on=False,
                **fill_kwargs,
            )

            if "label" not in line_kwargs:
                line_kwargs["label"] = "Median"

            self.ax.plot(
                x_rt_s,
                quantiles[0, :],
                clip_on=False,
                **line_kwargs,
            )

        with mpl.rc_context(rc=self.style):
            plt.legend()

        return self

    def plot_predictive(
        self,
        idata: az.data.inference_data.InferenceData,
        predictive_type: str,
        observed_var_name: str | None = None,
        n_points: int = 1000,
        ci_range: float = 0.95,
        fill_kwargs: dict[str, typing.Any] | None = None,
        line_kwargs: dict[str, typing.Any] | None = None,
    ) -> ReciprobitPlot:
        """
        Plot a summary of draws from a prior or posterior predictive
        distribution.

        Parameters
        ----------
        idata
            Inference data object containing posterior samples.
        predictive_type
            Either `prior` or `posterior`.
        n_points
            How many points to use when evaluating the model.
        ci_range
            Width of the credible interval.
        observed_var_name
            Name of the observed variable in the predictive samples. If `None`,
            assumes that there is only a single variable with observations.
        fill_kwargs
            Keyword arguments passed directly to `plt.fill_between`, for the
            credible interval.
        line_kwargs
            Keyword arguments passed directly to `plt.line`, for the median.

        Returns
        -------
        ReciprobitPlot
            The `ReciprobitPlot` instance.
        """

        if fill_kwargs is None:
            fill_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

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
        )

        data = (
            dataset.to_dataarray()
            if observed_var_name is None
            else dataset[observed_var_name]
        )

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

        (lower_q, upper_q) = q_from_ci_range(ci_range=ci_range)

        quantiles = da.quantile(
            dim="sample",
            q=[0.5, lower_q, upper_q],
        )

        with mpl.rc_context(rc=self.style):

            if "alpha" not in fill_kwargs:
                fill_kwargs["alpha"] = 0.5

            if "label" not in fill_kwargs:
                fill_kwargs["label"] = f"{ci_range:.0%} credible interval"

            self.ax.fill_between(
                da.rt.values,
                quantiles.sel(quantile=lower_q).values,
                quantiles.sel(quantile=upper_q).values,
                clip_on=False,
                **fill_kwargs,
            )

            if "label" not in line_kwargs:
                line_kwargs["label"] = "Median"

            self.ax.plot(
                da.rt.values,
                quantiles.sel(quantile=0.5).values,
                clip_on=False,
                **line_kwargs,
            )

            plt.legend()

        return self

    @property
    def min_rt_s(self) -> float:
        return self._min_rt_s

    @min_rt_s.setter
    def min_rt_s(self, value: float) -> None:
        self._min_rt_s = value
        self.ax.set_xlim(xmin=value)

    @property
    def max_rt_s(self) -> float:
        return self._max_rt_s

    @max_rt_s.setter
    def max_rt_s(self, value: float) -> None:
        self._max_rt_s = value
        self.ax.set_xlim(xmax=value)

    @property
    def min_p(self) -> float:
        return self._min_p

    @min_p.setter
    def min_p(self, value: float) -> None:
        if value < 0:
            raise ValueError("`min_p` must be >= 0")
        self._min_p = value
        self.ax.set_ylim(ymin=value)

    @property
    def max_p(self) -> float:
        return self._max_p

    @max_p.setter
    def max_p(self, value: float) -> None:
        if value > 1:
            raise ValueError("`max_p` must be <= 1")
        self._max_p = value
        self.ax.set_ylim(ymax=value)


def q_from_ci_range(ci_range: float) -> tuple[float, float]:
    return ((1 - ci_range) / 2, 1 - (1 - ci_range) / 2)
