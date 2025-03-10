# ezmsg-learn

This repository contains a Python package with modules for machine learning (ML)-related processing in the [`ezmsg`](https://www.ezmsg.org) framework. As ezmsg is intended primarily for processing unbounded streaming signals, so are the modules in this repo.

> If you are only interested in offline analysis without concern for reproducibility in online applications, then you should probably look elsewhere.

Processing units include dimensionality reduction, linear regression, and classification that can be initialized with known weights, or adapted on-the-fly with incoming (labeled) data. Machine-learning code depends on `river`, `scikit-learn`, `numpy`, and `torch`.

## Getting Started

This ezmsg namespace package is still highly experimental and under active development. It is not yet available on PyPI, so you will need to install it from source. The easiest way to do this is to use the `pip` command to install the package directly from GitHub:

```bash
pip install git+ssh://git@github.com/ezmsg-org/ezmsg-learn
```

Note that this package depends on a specific version of `ezmsg-sigproc` (specifically, [this branch]("70-use-protocols-for-axisarray-transformers")) that has yet to be merged and released. This may conflict with your project's separate dependency on ezmsg-sigproc. However, this specific version of ezmsg-sigproc should be backwards compatible with its main branch, so in your project you can modify the dependency on ezmsg-sigproc to point to the new branch. e.g.,

```bash
pip install git+ssh://git@github.com/ezmsg-org/ezmsg-sigproc@70-use-protocols-for-axisarray-transformers
```
