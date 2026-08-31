# ezmsg-learn

This repository contains a Python package with modules for machine learning (ML)-related processing in the [`ezmsg`](https://www.ezmsg.org) framework. As ezmsg is intended primarily for processing unbounded streaming signals, so are the modules in this repo.

> If you are only interested in offline analysis without concern for reproducibility in online applications, then you should probably look elsewhere.

Processing units include dimensionality reduction, linear regression, and classification that can be initialized with known weights, or adapted on-the-fly with incoming (labeled) data.

## Installation

The base install is NumPy-only. The machine-learning backends are optional extras, so a deployment that uses only the lightweight processors does not pay for a PyTorch or scikit-learn install:

```bash
pip install ezmsg-learn              # numpy-only processors
pip install "ezmsg-learn[sklearn]"   # + pandas, river, scikit-learn
pip install "ezmsg-learn[torch]"     # + torch
pip install "ezmsg-learn[all]"       # everything
```

Or install the latest development version:

```bash
pip install "git+https://github.com/ezmsg-org/ezmsg-learn@dev#egg=ezmsg-learn[all]"
```

Importing a module whose backend is not installed raises an `ImportError` naming the extra to install.

## Dependencies

Base (`pip install ezmsg-learn`) — `ezmsg`, `ezmsg-baseproc`, `ezmsg-sigproc`, `numpy`, `scipy`, `array-api-compat`.

| Extra | Adds | Covers |
| --- | --- | --- |
| _(none)_ | — | `process.ssr`, `process.flatten`, `process.seqseqsampler`, `process.refit_kalman`, `model.cca`, `model.refit_kalman` |
| `sklearn` | `pandas`, `river`, `scikit-learn` | `process.adaptive_linear_regressor`, `process.linear_regressor`, `process.sgd`, `process.slda`, `process.sklearn`, `dim_reduce.*` |
| `torch` | `torch` | `process.base`, `process.torch`, `process.rnn`, `process.transformer`, `process.mlp_old`, `model.mlp`, `model.rnn`, `model.transformer` |
| `all` | both of the above | everything, including all `collection.sample_adapt_regressor` backends |

`collection.sample_adapt_regressor` imports its backend lazily, so it needs only the extra for the `model_type` in use — and none at all for `model_type="kalman"`.


## Development

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork this repository and clone your fork locally.
3. Open a terminal and `cd` to the cloned folder.
4. Run `uv sync` to create a `.venv` and install dependencies.
5. (Optional) Install pre-commit hooks: `uv run pre-commit install`
6. After making changes, run the test suite: `uv run pytest tests`

## License

MIT License - see [LICENSE](LICENSE) for details.
