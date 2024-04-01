
import numpy as np

import pymc as pm

import pylater.examples
import pylater.dist


def demo():

    data = pylater.examples.load_cw1995()["a_p95"]

    with pm.Model() as model:

        mu = pm.Normal("mu", mu=3, sigma=1.5)
        sigma = pm.HalfNormal("sigma", sigma=3)
        sigma_e = pm.HalfNormal("sigma_e", sigma=5)

        pylater.dist.model(
            name="obs",
            mu=mu,
            sigma=sigma,
            sigma_e=sigma_e,
            observed=data.promptness,
        )

    return model



def demo_shared():

    data = pylater.examples.load_cw1995()

    data = {key: value for (key, value) in data.items() if key.startswith("b")}

    with pm.Model() as model:

        mu = pm.Normal("mu", mu=3, sigma=1.5, size=len(data))
        sigma = pm.HalfNormal("sigma", sigma=3)
        sigma_e = pm.HalfNormal("sigma_e", sigma=5, size=len(data))

        for (i, (key, value)) in enumerate(data.items()):

            pylater.dist.model(
                name=f"obs_{key}",
                mu=mu[i],
                sigma=sigma,
                sigma_e=sigma_e[i],
                observed=value.promptness,
            )

    return model
