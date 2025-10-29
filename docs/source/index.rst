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

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/ezmsg-org/ezmsg-learn

Dependencies
^^^^^^^^^^^^

This package requires:

* ``ezmsg`` - Core ezmsg framework
* ``ezmsg-sigproc`` - Signal processing extensions
* ``numpy`` - Numerical computing
* ``scikit-learn`` - Machine learning utilities
* ``torch`` - Deep learning framework
* ``river`` - Online machine learning

Quick Start
-----------

For general ezmsg tutorials and guides, visit `ezmsg.org <https://www.ezmsg.org>`_.

For package-specific examples and usage, see the :doc:`api/index` documentation.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
