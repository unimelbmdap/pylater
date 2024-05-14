# pylater

A Python library for working with LATER ('Linear Approach to Threshold with Ergodic Rate for Reaction Times') models using Bayesian methods in [PyMC](https://www.pymc.io).

LATER is a model for distributions of reaction times, such as the time take for a person to press a button or to move their eyes after the onset of a task demand (see Carpenter & Noorani, 2023, for an overview of the model and its applications).
Briefly, the LATER model stipulates that a distribution of recorded reaction times can be described by the competitive combination of two Normal distributions that operate in the space of the reciprocal of reaction time ('promptness').
The primary distribution has free parameters for location ($\mu$) and scale ($\sigma$) and the other distribution, called the 'early' distribution, has a fixed location parameter ($\mu_e = 0$) and a free scale parameter ($\sigma_e$).
According to the LATER model, a response time on a single trial is given by the reciprocal of the maximum of independent draws from these two distributions.

This library provides three main features:

* A LATER distribution class that can be used in PyMC models.
* A visualisation helper to produce Matplotlib figures in the 'reciprobit' space used by LATER practitioners.
* Example data digitised from Carpenter & Williams (1995).

> [!NOTE]
> Also see [https://github.com/unimelbmdap/LATERmodel/](LATERmodel) for an R package with a non-Bayesian implementation of LATER and with a [graphical interface](https://later.researchsoftware.unimelb.edu.au/).

## Installation

The library can be installed using `pip`:

```bash
pip install pylater
```

## Usage




## References

* Carpenter, R.H.S. & Noorani, I. (2023) LATER: The Neurophysiology of Decision-Making. Cambridge University Press. [doi: 10.1017/9781108920803](https://doi.org/10.1017/9781108920803)
* Carpenter, R.H.S. & Williams, M.L.L. (1995) Neural computation of log likelihood in control of saccadic eye movements. *Nature, 377* (6544), 59–62. [doi: 10.1038/377059a0](https://doi.org/10.1038/377059a0)
