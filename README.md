
# pylater

A Python library for working with LATER ('Linear Approach to Threshold with Ergodic Rate for Reaction Times') models using Bayesian methods in [PyMC](https://www.pymc.io).

The LATER model stipulates that a distribution of recorded reaction times (e.g., the time taken for a human to press a button or move their eyes in response to the onset of a task demand) can be described by the competitive combination of two normal distributions that operate in the space of reciprocal reaction time (promptness).
The primary distribution has free parameters for location ($\mu$) and scale ($\sigma$) and the other distribution, called the 'early' distribution, has a fixed location parameter ($\mu_e = 0$) and a free scale parameter ($\sigma_e$).
According to the LATER model, a response time on a single trial is given by the reciprocal of the maximum of independent draws from these two distributions.

This library provides three main features:

* A LATER distribution class that can be used in PyMC models.
* A visualisation helper to produce matplotlib figures in the 'reciprobit' space used by LATER practitioners.
* Example data digitised from Carpenter & Williams (1995).


## Installation

The library can be installed using `pip`:

```bash
pip install pylater
```

## Usage




## Development

### Installation

```bash
git clone https://github.com/unimelbmdap/pylater
```

### Type checking

```bash
hatch run types:check
```

### Linting and formatting

```bash
hatch fmt
```

### Testing

```bash
hatch run tests:test
```

### Interactive use

```bash
hatch run interactive:session

## References

* Carpenter, R.H.S. & Williams, M.L.L. (1995) Neural computation of log likelihood in control of saccadic eye movements. *Nature, 377* (6544), 59–62. [doi: 10.1038/377059a0](https://doi.org/10.1038/377059a0)
