from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt


class LATER:

    __doc__ = """A custom PyMC distribution for a LATER model.

    Parameters
    ----------
    name
        Identifier for the distribution.
    mu
        Mean of the primary component.
    sigma
        Standard deviation of the primary component.
    sigma_e
        Standard deviation of the early component.
    observed_rt_s
        Observed reaction times, in units of seconds.
    **kwargs
        Additional arguments are passed directly to `pm.CustomDist`.

    Returns
    -------
    pm.CustomDist
        Distribution for use with a PyMC model.

    Notes
    -----
    * The model parameters are in units of promptness (reciprocal of time).
    * Random samples from the model are in units of time.

    """

    def __new__(  # type: ignore
        cls,
        name: str,
        mu: float | pm.Distribution,
        sigma: float | pm.Distribution,
        sigma_e: float | pm.Distribution,
        observed_rt_s: npt.NDArray[np.float64] | None = None,
        **kwargs: str | float | npt.NDArray[np.float64],
    ) -> pm.CustomDist:

        observed_promptness = 1 / observed_rt_s if observed_rt_s is not None else None

        return pm.CustomDist(
            name,
            mu,
            sigma,
            sigma_e,
            logp=logp,
            logcdf=logcdf,
            random=random,
            observed=observed_promptness,
            **kwargs,
        )


def logp(
    value: pt.TensorVariable,  # type: ignore
    mu: pt.TensorVariable,  # type: ignore
    sigma: pt.TensorVariable,  # type: ignore
    sigma_e: pt.TensorVariable,  # type: ignore
) -> pt.TensorVariable:  # type: ignore
    early_mu = 0

    a = pm.Normal.logp(value=value, mu=mu, sigma=sigma) + pm.Normal.logcdf(
        value=value, mu=early_mu, sigma=sigma_e
    )
    b = pm.Normal.logp(value=value, mu=early_mu, sigma=sigma_e) + pm.Normal.logcdf(
        value=value, mu=mu, sigma=sigma
    )

    return pt.logsumexp(x=pt.stack(tensors=(a, b), axis=0), axis=0)  # type: ignore


def logcdf(
    value: pt.TensorVariable,  # type: ignore
    mu: pt.TensorVariable,  # type: ignore
    sigma: pt.TensorVariable,  # type: ignore
    sigma_e: pt.TensorVariable,  # type: ignore
) -> pt.TensorVariable:  # type: ignore
    early_mu = 0

    a = pm.Normal.logcdf(value=value, mu=early_mu, sigma=sigma_e)
    b = pm.Normal.logcdf(value=value, mu=mu, sigma=sigma)

    return a + b


def random(
    mu: npt.NDArray[np.float64] | float,
    sigma: npt.NDArray[np.float64] | float,
    sigma_e: npt.NDArray[np.float64] | float,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> npt.NDArray[np.float64] | float:
    if rng is None:
        rng = np.random.default_rng()

    later = rng.normal(loc=mu, scale=sigma, size=size)
    early = rng.normal(loc=0, scale=sigma_e, size=size)

    promptness = np.where(later > early, later, early)

    return 1 / promptness
