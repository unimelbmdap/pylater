
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
import sympy


def logp_(
    value: pt.TensorVariable,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
    sigma_e: pt.TensorVariable,
) -> pt.TensorVariable:

    early_mu = 0

    exp = pt.exp
    erf = pt.erf

    sqrt_2 = np.sqrt(2)

    p = (
        (
            (
                exp(-(((value - mu)**2) / (2 * sigma**2)))
                * (1 + erf((value - early_mu) / (sqrt_2 * sigma_e)))
            ) / sigma
            + (
                exp(-(((value - early_mu)**2) / (2 * sigma_e**2)))
                * (1 + erf((value - mu) / (sqrt_2 * sigma)))
            ) / sigma_e
        ) / (2 * np.sqrt(2 * np.pi))
    )

    return pt.log(p)

def logp(
    value: pt.TensorVariable,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
    sigma_e: pt.TensorVariable,
) -> pt.TensorVariable:

    early_mu = 0

    a = (
        pm.Normal.logp(value=value, mu=mu, sigma=sigma)
        + pm.Normal.logcdf(value=value, mu=early_mu, sigma=sigma_e)
    )
    b = (
        pm.Normal.logp(value=value, mu=early_mu, sigma=sigma_e)
        + pm.Normal.logcdf(value=value, mu=mu, sigma=sigma)
    )

    logp = pt.logsumexp(x=pt.stack(tensors=(a, b), axis=0), axis=0)

    return logp


def random(
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
    sigma_e: np.ndarray | float,
    rng: np.random.Generator | None = None,
    size : tuple[int] | None =None,
) -> np.ndarray | float:

    if rng is None:
        rng = np.random.default_rng()

    later = rng.normal(loc=mu, scale=sigma, size=size)
    early = rng.normal(loc=0, scale=sigma_e, size=size)

    y = np.where(later > early, later, early)

    return y


def model(
    name: str,
    mu: float | pm.Distribution,
    sigma: float | pm.Distribution,
    sigma_e: float | pm.Distribution,
    observed: npt.NDArray | None = None,
) -> pm.Distribution:

    return pm.CustomDist(
        name,
        mu,
        sigma,
        sigma_e,
        logp=logp,
        random=random,
        observed=observed,
    )


class LATER:

    def __new__(
        cls,
        name: str,
        mu: float | pm.Distribution,
        sigma: float | pm.Distribution,
        sigma_e: float | pm.Distribution,
        observed: npt.NDArray | None = None,
    ):

        return pm.CustomDist(
            name,
            mu,
            sigma,
            sigma_e,
            logp=logp,
            random=random,
            observed=observed,
        )




def symbolic_p():

    mu, sigma, sigma_e, x = sympy.symbols("mu sigma sigma_e x")

    exp = sympy.exp
    erf = sympy.erf

    p = (
        (
            (
                exp(-(((x - mu)**2) / (2 * sigma**2)))
                * (1 + erf((x - mu) / (sympy.sqrt(2) * sigma_e)))
            ) / sigma
            + (
                exp(-(((x - 0)**2) / (2 * sigma_e**2)))
                * (1 + erf((x - mu) / (sympy.sqrt(2) * sigma)))
            ) / sigma_e
        ) / (2 * sympy.sqrt(2 * sympy.pi))
    )

    return p


def symbolic_normal(subscript="_1"):

    mu = sympy.Symbol(f"mu{subscript}")
    sigma = sympy.Symbol(f"sigma{subscript}")
    x = sympy.Symbol(f"x{subscript}")

    p = (
        (1 / (sigma * sympy.sqrt(2 * sympy.pi))) *
        sympy.exp(
            -1 * (((x - mu) ** 2) / (2 * sigma ** 2))
        )
    )

    return p
