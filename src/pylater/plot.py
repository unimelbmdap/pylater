import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import numpy as np
import numpy.typing as npt
import scipy.stats



def reciprobit_figure() -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    (fig, ax) = plt.subplots()

    format_reciprobit_axes(ax=ax)

    return (fig, ax)


def format_reciprobit_axes(
    ax: matplotlib.axes.Axes,
    y_min=0.001,
    y_max=1 - 0.001,
) -> None:
    ax.set_xlim(0.1,0.5)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_yscale("function", functions=(probit_forward, probit_inverse))
    ax.set_xscale("function", functions=(invert, invert))

    #sec = ax.secondary_xaxis("top") #, functions=(invert, invert))
    #sec.set_xlim(0.1,0.5)
    #sec.set_xscale("linear") #, functions=(i, i))
    #sec.set_xticks((ax.get_xticks()))

def i(values): return 1/values

def invert(values: npt.ArrayLike) -> npt.NDArray[np.float_]:
    return -1 / np.array(values)


def probit_forward(values: npt.ArrayLike) -> npt.NDArray[np.float_]:
    "Linear probability to probit space"

    z: npt.NDArray[np.float_] = scipy.stats.norm.ppf(q=values)

    return z


def probit_inverse(values: npt.ArrayLike) -> npt.NDArray[np.float_]:
    "Probit to linear probability space"

    p: npt.NDArray[np.float_] = scipy.stats.norm.cdf(x=values)

    return p
