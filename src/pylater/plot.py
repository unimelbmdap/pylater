import enum

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.scale
import matplotlib.transforms
import matplotlib.ticker
import numpy as np
import numpy.typing as npt
import scipy.stats


class AxisType(enum.Enum):
    TIME = "time"
    PROMPTNESS = "promptness"


class ReciprobitTimeTransform(matplotlib.transforms.Transform):

    input_dims = output_dims = 1

    def __init__(self) -> None:
        super().__init__()

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float_]:
        masked = np.ma.masked_where(condition=a <= 0.0, a=a)  # type: ignore
        values: npt.NDArray[np.float_] = -1.0 / masked
        return values

    def inverted(self) -> matplotlib.transforms.Transform:
        return ReciprobitTimeTransformInverted()


class ReciprobitTimeTransformInverted(matplotlib.transforms.Transform):

    input_dims = output_dims = 1

    def __init__(self) -> None:
        super().__init__()

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float_]:
        return -1.0 / np.array(a)

    def inverted(self) -> matplotlib.transforms.Transform:
        return ReciprobitTimeTransform()



class ReciprobitTimeScale(matplotlib.scale.ScaleBase):

    name = "reciprobit_time"

    def __init__(
        self,
        axis: matplotlib.axis.Axis,
        axis_type: AxisType = AxisType.TIME,
    ) -> None:
        super().__init__(axis=axis)
        self.axis_type = axis_type

    def get_transform(self) -> matplotlib.transforms.Transform:
        return ReciprobitTimeTransform()

    def set_default_locators_and_formatters(self, axis: matplotlib.axis.Axis) -> None:

        def _tick_formatter(x: float, pos: float) -> str:
            if self.axis_type == AxisType.TIME:
                return f"{x * 1000:,.5g}"
            elif self.axis_type == AxisType.PROMPTNESS:
                if x <= 0:
                    return ""
                else:
                    return f"{1 / x:,.2g}"

        axis.set_major_formatter(
            formatter=matplotlib.ticker.FuncFormatter(func=_tick_formatter)
        )

    def limit_range_for_scale(
        self,
        vmin: float,
        vmax: float,
        minpos: float,
    ) -> tuple[float, float]:
        if not np.isfinite(minpos):
            minpos = 1e-7
        return (minpos if vmin <= 0 else vmin, vmax)


class ProbitTransform(matplotlib.transforms.Transform):

    input_dims = output_dims = 1

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float_]:
        z: npt.NDArray[np.float_] = scipy.stats.norm.ppf(q=a)
        return z

    def inverted(self) -> matplotlib.transforms.Transform:
        return InverseProbitTransform()


class InverseProbitTransform(matplotlib.transforms.Transform):

    input_dims = output_dims = 1

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float_]:
        p: npt.NDArray[np.float_] = scipy.stats.norm.cdf(x=a)
        return p

    def inverted(self) -> matplotlib.transforms.Transform:
        return ProbitTransform()


class ProbitScale(matplotlib.scale.ScaleBase):

    name = "probit"

    def get_transform(self) -> matplotlib.transforms.Transform:
        return ProbitTransform()

    def set_default_locators_and_formatters(self, axis: matplotlib.axis.Axis) -> None:

        def _tick_formatter(x: float, pos: float) -> str:
            return f"{x*100:.3g}"

        axis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(func=_tick_formatter)
        )

    def limit_range_for_scale(
        self,
        vmin: float,
        vmax: float,
        minpos: float,
    ) -> tuple[float, float]:
        if not np.isfinite(minpos):
            minpos = 1e-7
        return (
            minpos if vmin <= 0 else vmin,
            1 - minpos if vmax >= 1 else vmax
        )


def reciprobit_figure(
    rt_s_min: float = 50 / 1000,
    rt_s_max: float = 2000 / 1000,
    p_min: float = 0.001,
    p_max: float = 1 - 0.001,
    apply_default_style: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    style = (
        get_mpl_defaults()
        if apply_default_style
        else {}
    )

    with mpl.rc_context(rc=style):
        (fig, ax) = plt.subplots()

        tick_locations_ms = np.array([50, 100, 150, 200, 300, 500, 1000])

        ax.set_xlim((rt_s_min, rt_s_max))
        ax.set_xticks(ticks=tick_locations_ms / 1000)
        ax.set_xscale(value="reciprobit_time")
        ax.set_xlabel(xlabel="Latency (ms)")

        ax_promptness = ax.secondary_xaxis(location="top")
        ax_promptness.set_xscale(value="reciprobit_time", axis_type=AxisType.PROMPTNESS)
        ax_promptness.set_xticks(ticks=tick_locations_ms / 1000)
        ax_promptness.set_xlabel(xlabel="Promptness (1/s)")

        y_ticks = (
            np.array(
                [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98, 99, 99.5, 99.9]
            ) / 100
        )

        ax.set_ylim((p_min, p_max))
        ax.set_yscale(value="probit")
        ax.set_yticks(ticks=y_ticks)
        ax.set_ylabel(ylabel="Cumulative probability (%)")

        #ax_z = ax.secondary_yaxis(location="right")
        #ax_z.set_yticks(ticks=y_ticks)
        #ax_z.set_yscale(value="linear")

    return (fig, ax)


def get_mpl_defaults() -> dict[str, list[str] | float | bool]:

    defaults: dict[str, list[str] | float | bool] = {
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

    return defaults


matplotlib.scale.register_scale(scale_class=ReciprobitTimeScale)
matplotlib.scale.register_scale(scale_class=ProbitScale)
