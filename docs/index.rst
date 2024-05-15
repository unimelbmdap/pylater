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

Usage
-----

References
----------

.. bibliography::
    
