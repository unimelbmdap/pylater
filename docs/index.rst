Documentation
=============

.. toctree::
    :maxdepth: 1
    :hidden:

    notebooks/reciprobit_plots
    notebooks/priors
    notebooks/model_fitting
    notebooks/model_comparison
    api

This is a Python library for working with LATER ('Linear Approach to Threshold with Ergodic Rate for Reaction Times') models using Bayesian methods in `PyMC <https://www.pymc.io>`_.

LATER is a model for distributions of reaction times, such as the time take for a person to press a button or to move their eyes after the onset of a task demand (see :cite:t:`Carpenter2023` for an overview of the model and its applications).

This library provides four main features:

* A LATER distribution class that can be used in PyMC models (``pylater.LATER``).
* A visualisation helper to produce Matplotlib figures in the 'reciprobit' space used by LATER practitioners (``pylater.ReciprobitPlot``).
* An example of a model constructed with default priors (``pylater.build_default_model``).
* Reaction time data digitised from :cite:t:`Carpenter1995` (``pylater.data.cw1995``).

.. figure:: _static/pylater_example.png

    An example of a reciprobit plot, showing a condition from :cite:t:`Carpenter1995` and a summary of its posterior retrodictive distribution.

Briefly, the LATER model stipulates that a distribution of recorded reaction times can be described by the competitive combination of two Normal distributions that operate in the space of the reciprocal of reaction time ('promptness').
The primary distribution has free parameters for location (:math:`\mu`) and scale (:math:`\sigma`) and the other distribution, called the 'early' distribution, has a fixed location parameter (:math:`\mu_e = 0`) and a free scale parameter (:math:`\sigma_e`).
According to the LATER model, a response time on a single trial is given by the reciprocal of the maximum of independent draws from these two distributions.

.. note:: Also see `LATERmodel <https://unimelbmdap.github.io/LATERmodel/>`_ for an R package with a non-Bayesian implementation of LATER and with a `graphical interface <https://later.researchsoftware.unimelb.edu.au/>`_.

Installation
------------

The library can be installed using ``pip``:

.. code-block:: bash

    pip install https://github.com/unimelbmdap/pylater

Quickstart
----------

.. code-block:: python

    import pymc as pm
    import pylater

    # load the data from the 50% condition from participant a
    data = pylater.data.cw1995["a_p50"]

    # build a default PyMC model
    model = pylater.model.build_default_model(datasets=[data])

    # sample prior predictives
    with model:
        idata = pm.sample_prior_predictive()

    # visualise prior predictives using a reciprobit plot
    priors_plot = pylater.ReciprobitPlot()
    priors_plot.plot_predictive(idata=idata, predictive_type="prior")

    # sample a posterior
    with model:
        idata = pm.sample()

    # look at posterior statistics
    pm.stats.summary(data=idata)

    # sample posterior predictives
    with model:
        idata = pm.sample_posterior_predictive(trace=idata, extend_inferencedata=True)

    # visualise posterior retrodictives using a reciprobit plot, with overlaid data
    posterior_plot = pylater.ReciprobitPlot()
    posterior_plot.plot_predictive(idata=idata, predictive_type="posterior")
    posterior_plot.plot_data(data=data)

Authors
-------

* Damien Mannion, Melbourne Data Analytics Platform (MDAP), University of Melbourne
* Maria del Mar Quiroga, Melbourne Data Analytics Platform (MDAP), University of Melbourne
* Edoardo Tescari, Melbourne Data Analytics Platform (MDAP), University of Melbourne
* Andrew Anderson, Department of Optometry and Vision Sciences, University of Melbourne

References
----------

.. bibliography::
    

