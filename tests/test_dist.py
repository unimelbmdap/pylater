import numpy as np

import pylater.dist


def test_logp() -> None:
    value = 1.0
    mu = 3.0
    sigma = 1.0
    sigma_e = 5.0

    logp = pylater.dist.logp(value=value, mu=mu, sigma=sigma, sigma_e=sigma_e).eval()

    assert np.isfinite(logp)

    # check the size is done properly
    shape = (3, 4, 1)

    nd_logp = pylater.dist.logp(
        value=np.ones(shape) * value,
        mu=np.ones(shape) * mu,
        sigma=np.ones(shape) * sigma,
        sigma_e=np.ones(shape) * sigma_e,
    ).eval()

    assert nd_logp.shape == shape


def test_random() -> None:
    mu = 3.0
    sigma = 1.0
    sigma_e = 5.0

    # test that samples can be drawn
    samples = pylater.dist.random(mu=mu, sigma=sigma, sigma_e=sigma_e)

    # test that they can be drawn reproducibility
    seeded_samples = tuple(
        pylater.dist.random(
            mu=mu,
            sigma=sigma,
            sigma_e=sigma_e,
            rng=np.random.default_rng(seed=124121),
        )
        for _ in range(2)
    )
    assert np.all(seeded_samples[0] == seeded_samples[1])

    # and that it is different to before
    assert samples != seeded_samples[0]

    # check that the size parameter works
    sized_shape = (10, 1, 4)
    sized_samples = pylater.dist.random(
        mu=mu, sigma=sigma, sigma_e=sigma_e, size=sized_shape
    )
    assert sized_samples.shape == sized_shape
