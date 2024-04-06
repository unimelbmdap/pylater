
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats


def probit_forward(values: npt.ArrayLike) -> npt.NDArray:
    "Linear probability to probit space"

    z = scipy.stats.norm.ppf(q=values)

    z = np.where(
        np.isfinite(z),
        z,
        np.where(
            z > 0,
            10,
            -10,
        ),
    )

    return z


def probit_inverse(values: npt.ArrayLike) -> npt.NDArray:
    "Probit to linear probability space"

    return scipy.stats.norm.cdf(x=values)


def get_reciprobit_figure():

    # not really working yet

    (fig, ax) = plt.subplots()

    ax.set_ylim(0.001, 1 - 0.001)
    ax.set_yscale("function", functions=(probit_forward, probit_inverse))

    return (fig, ax)
