from __future__ import annotations

import enum
import typing

import matplotlib.axes
import matplotlib.figure
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms
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

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:
        with np.errstate(divide="ignore", invalid="ignore"):
            values: npt.NDArray[np.float64] = -1.0 / np.array(a)

        return values

    def inverted(self) -> matplotlib.transforms.Transform:
        return ReciprobitTimeTransformInverted()


class ReciprobitTimeTransformInverted(matplotlib.transforms.Transform):
    input_dims = output_dims = 1

    def __init__(self) -> None:
        super().__init__()

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:
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
        def _tick_formatter(x: float, pos: float) -> str:  # noqa: ARG001
            if self.axis_type == AxisType.TIME:
                return f"{x * 1000:,.5g}"
            if self.axis_type == AxisType.PROMPTNESS:
                if x <= 0:
                    return ""
                return f"{1 / x:,.2g}"
            return ""

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

    def __init__(
        self,
        linthresh: float = 0.1 / 100,
        linscale: float = 0.1,
    ) -> None:

        super().__init__()

        self.linthresh = linthresh
        self.linscale = linscale

        self.lin_bounds_z = abs(scipy.stats.norm.ppf(q=self.linthresh))

        self.z_width = self.lin_bounds_z * 2


    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:

        a = np.array(a)
        transformed = np.full_like(a, np.nan)

        in_z_region = np.logical_and(transformed >= self.linthresh, a <= (1 - self.linthresh))

        transformed[in_z_region] = interval_convert(
            value=scipy.stats.norm.ppf(q=a[in_z_region]),
            old_interval=(-self.lin_bounds_z, +self.lin_bounds_z),
            new_interval=(self.linscale, 1 - self.linscale),
        )

        lower_linear = np.logical_and(np.logical_not(in_z_region), a < self.linthresh)

        transformed[lower_linear] = interval_convert(
            value=transformed[lower_linear],
            old_interval=(0, self.linthresh),
            new_interval=(0, self.linscale),
        )

        upper_linear = np.logical_and(np.logical_not(in_z_region), a > (1 - self.linthresh))

        transformed[upper_linear] = interval_convert(
            value=transformed[upper_linear],
            old_interval=(1 - self.linthresh, 1),
            new_interval=(1 - self.linscale, 1),
        )

        return transformed

    def inverted(self) -> matplotlib.transforms.Transform:
        return InverseProbitTransform(
            linthresh=self.linthresh,
            linscale=self.linscale,
        )


class InverseProbitTransform(matplotlib.transforms.Transform):
    input_dims = output_dims = 1

    def __init__(
        self,
        linthresh: float = 0.1 / 100,
        linscale: float = 0.1,
    ) -> None:

        super().__init__()

        self.linthresh = linthresh
        self.linscale = linscale

        self.lin_bounds_z = abs(scipy.stats.norm.ppf(q=self.linthresh))

        self.z_width = self.lin_bounds_z * 2

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:

        a = np.array(a)
        transformed = np.full_like(a, np.nan)

        in_z_region = np.logical_and(a >= self.linscale, a <= (1 - self.linscale))

        transformed[in_z_region] = scipy.stats.norm.cdf(
            x=interval_convert(
                value=a[in_z_region],
                old_interval=(self.linscale, 1 - self.linscale),
                new_interval=(-self.lin_bounds_z, +self.lin_bounds_z),
            ),
        )

        lower_linear = np.logical_and(np.logical_not(in_z_region), a < self.linscale)

        transformed[lower_linear] = interval_convert(
            value=a[lower_linear],
            old_interval=(0, self.linscale),
            new_interval=(0, self.linthresh),
        )

        upper_linear = np.logical_and(np.logical_not(in_z_region), a > self.linscale)

        transformed[upper_linear] = interval_convert(
            value=a[upper_linear],
            old_interval=(1 - self.linscale, 1),
            new_interval=(1 - self.linthresh, 1),
        )

        return transformed

    def inverted(self) -> matplotlib.transforms.Transform:
        return ProbitTransform(
            linthresh=self.linthresh,
            linscale=self.linscale,
        )


class ProbitScale(matplotlib.scale.ScaleBase):
    name = "probit"

    def __init__(
        self,
        axis: matplotlib.axis.Axis,
        linthresh: float = 0.1 / 100,
        linscale: float = 0.05,
    ) -> None:
        super().__init__(axis=axis)

        self._transform = ProbitTransform(
            linthresh=linthresh,
            linscale=linscale,
        )

    def get_transform(self) -> matplotlib.transforms.Transform:
        return self._transform

    def set_default_locators_and_formatters(self, axis: matplotlib.axis.Axis) -> None:
        def _tick_formatter(x: float, pos: float) -> str:  # noqa: ARG001
            return f"{x*100:.3g}"

        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(func=_tick_formatter))

    def limit_range_for_scale(
        self,
        vmin: float,
        vmax: float,
        minpos: float,
    ) -> tuple[float, float]:
        if not np.isfinite(minpos):
            minpos = 0.0
        return (minpos if vmin <= 0 else vmin, 1 - minpos if vmax >= 1 else vmax)


T = typing.TypeVar("T", float, npt.NDArray[np.float64])


def interval_convert(
    value: T,
    old_interval: tuple[float, float],
    new_interval: tuple[float, float],
) -> T:
    """Convert values from one range to another.

    Parameters
    ----------
    value
        Value to convert.
    old_interval
        Interval from which ``value`` was obtained (min, max)
    new_interval
        Desired new interval (min, max)

    Returns
    -------
    new_value: number
        ``value`` transformed to the new interval.

    """

    (old_min, old_max) = old_interval
    old_range = old_max - old_min

    (new_min, new_max) = new_interval
    new_range = new_max - new_min

    new_value = (((value - old_min) * new_range) / old_range) + new_min

    return new_value


matplotlib.scale.register_scale(scale_class=ReciprobitTimeScale)
matplotlib.scale.register_scale(scale_class=ProbitScale)
