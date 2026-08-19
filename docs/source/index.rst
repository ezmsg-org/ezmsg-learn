ezmsg.learn
============

Machine learning modules for the `ezmsg <https://www.ezmsg.org>`_ framework.

.. note::
   **This package is experimental and under active development.**

Overview
--------

``ezmsg-learn`` provides machine learning processing units designed for streaming signals in the ezmsg framework.

Modules include:

* **Linear models** - Linear regression, SLDA, CCA, SGD
* **Non-linear models** - Multi-layer perceptrons (MLP)
* **Dimensionality reduction** - Incremental PCA and other decomposition methods
* **Utilities** - Helper functions for ML workflows

Most modules support both:

* **Offline initialization** with known weights
* **Online adaptation** with streaming labeled data

Installation
------------

The base install is NumPy-only; the machine-learning backends are optional
extras, so a deployment that uses only the lightweight processors does not pay
for a PyTorch or scikit-learn install:

.. code-block:: bash

   pip install ezmsg-learn              # numpy-only processors
   pip install "ezmsg-learn[sklearn]"   # + pandas, river, scikit-learn
   pip install "ezmsg-learn[torch]"     # + torch
   pip install "ezmsg-learn[all]"       # everything

Or install directly from GitHub:

.. code-block:: bash

   pip install "git+https://github.com/ezmsg-org/ezmsg-learn#egg=ezmsg-learn[all]"

Importing a module whose backend is not installed raises an ``ImportError``
naming the extra to install.

Dependencies
^^^^^^^^^^^^

The base install requires:

* ``ezmsg`` - Core ezmsg framework
* ``ezmsg-baseproc`` - Processor base classes
* ``ezmsg-sigproc`` - Signal processing extensions
* ``numpy`` - Numerical computing
* ``scipy`` - Scientific computing
* ``array-api-compat`` - Array API portability layer

Optional extras
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Extra
     - Adds
     - Covers
   * - *(none)*
     - —
     - ``process.ssr``, ``process.flatten``, ``process.seqseqsampler``,
       ``process.refit_kalman``, ``model.cca``, ``model.refit_kalman``
   * - ``sklearn``
     - ``pandas``, ``river``, ``scikit-learn``
     - ``process.adaptive_linear_regressor``, ``process.linear_regressor``,
       ``process.sgd``, ``process.slda``, ``process.sklearn``, ``dim_reduce.*``
   * - ``torch``
     - ``torch``
     - ``process.base``, ``process.torch``, ``process.rnn``,
       ``process.transformer``, ``process.mlp_old``, ``model.mlp``,
       ``model.rnn``, ``model.transformer``
   * - ``all``
     - both of the above
     - everything, including all ``collection.sample_adapt_regressor`` backends

:mod:`ezmsg.learn.collection.sample_adapt_regressor` imports its backend
lazily, so it needs only the extra for the ``model_type`` in use — and none at
all for ``model_type="kalman"``.

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/classification
   guides/array_api
   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
