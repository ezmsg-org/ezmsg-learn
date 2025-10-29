API Reference
=============

This page contains the complete API reference for ``ezmsg.learn``.

.. contents:: Modules
   :local:
   :depth: 1

Dimensionality Reduction
-------------------------

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.learn.dim_reduce.incremental_decomp
   ezmsg.learn.dim_reduce.adaptive_decomp

Linear Models
-------------

.. note::
   The ``ezmsg.learn.linear_model`` module is deprecated. Please use ``ezmsg.learn.process`` instead.

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.learn.linear_model.adaptive_linear_regressor
   ezmsg.learn.linear_model.linear_regressor
   ezmsg.learn.linear_model.sgd
   ezmsg.learn.linear_model.slda
   ezmsg.learn.linear_model.cca

Models
------

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.learn.model.cca
   ezmsg.learn.model.mlp
   ezmsg.learn.model.rnn
   ezmsg.learn.model.transformer
   ezmsg.learn.model.refit_kalman

Utilities
---------

.. autosummary::
   :toctree: generated
   :recursive:

   ezmsg.learn.util
