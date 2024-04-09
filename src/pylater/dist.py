
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt


class LATER:

    def __new__(
        cls,
        name: str,
        mu: float | pm.Distribution,
        sigma: float | pm.Distribution,
        sigma_e: float | pm.Distribution,
        observed: npt.NDArray[np.float_] | None = None,
    ) -> pm.CustomDist:

        observed_promptness = 1 / observed if observed is not None else None

        return pm.CustomDist(
            name,
            mu,
            sigma,
            sigma_e,
            logp=logp,
            random=random,
            observed=observed_promptness,
        )


def logp(
    value: pt.TensorVariable,  # type: ignore
    mu: pt.TensorVariable,  # type: ignore
    sigma: pt.TensorVariable,  # type: ignore
    sigma_e: pt.TensorVariable,  # type: ignore
) -> pt.TensorVariable:  # type: ignore

    early_mu = 0

    a = (
        pm.Normal.logp(value=value, mu=mu, sigma=sigma)
        + pm.Normal.logcdf(value=value, mu=early_mu, sigma=sigma_e)
    )
    b = (
        pm.Normal.logp(value=value, mu=early_mu, sigma=sigma_e)
        + pm.Normal.logcdf(value=value, mu=mu, sigma=sigma)
    )

    return pt.logsumexp(x=pt.stack(tensors=(a, b), axis=0), axis=0)  # type: ignore


def random(
    mu: npt.NDArray[np.float_] | float,
    sigma: npt.NDArray[np.float_] | float,
    sigma_e: npt.NDArray[np.float_] | float,
    rng: np.random.Generator | None = None,
    size : tuple[int] | None = None,
) -> npt.NDArray[np.float_] | float:

    if rng is None:
        rng = np.random.default_rng()

    later = rng.normal(loc=mu, scale=sigma, size=size)
    early = rng.normal(loc=0, scale=sigma_e, size=size)

    return np.where(later > early, later, early)


