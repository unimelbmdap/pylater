from __future__ import annotations

import enum

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

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:
        z: npt.NDArray[np.float64] = scipy.stats.norm.ppf(q=a)
        return z

    def inverted(self) -> matplotlib.transforms.Transform:
        return InverseProbitTransform()


class InverseProbitTransform(matplotlib.transforms.Transform):
    input_dims = output_dims = 1

    def transform_non_affine(self, a: npt.ArrayLike) -> npt.NDArray[np.float64]:
        p: npt.NDArray[np.float64] = scipy.stats.norm.cdf(x=a)
        return p

    def inverted(self) -> matplotlib.transforms.Transform:
        return ProbitTransform()


class ProbitScale(matplotlib.scale.ScaleBase):
    name = "probit"

    def get_transform(self) -> matplotlib.transforms.Transform:
        return ProbitTransform()

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
            minpos = 1e-7
        return (minpos if vmin <= 0 else vmin, 1 - minpos if vmax >= 1 else vmax)


matplotlib.scale.register_scale(scale_class=ReciprobitTimeScale)
matplotlib.scale.register_scale(scale_class=ProbitScale)
