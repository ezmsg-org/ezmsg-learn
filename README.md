# ezmsg-learn

This repository contains a Python package with modules for machine learning (ML)-related processing in the [`ezmsg`](https://www.ezmsg.org) framework. As ezmsg is intended primarily for processing unbounded streaming signals, so are the modules in this repo.

> If you are only interested in offline analysis without concern for reproducibility in online applications, then you should probably look elsewhere.

Processing units include dimensionality reduction, linear regression, and classification that can be initialized with known weights, or adapted on-the-fly with incoming (labeled) data. Machine-learning code depends on `river`, `scikit-learn`, and `numpy`, and `torch`.
