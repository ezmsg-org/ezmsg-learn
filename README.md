# ezmsg-learn

This repository contains a Python package with modules for machine learning (ML)-related processing in the [`ezmsg`](https://www.ezmsg.org) framework. As ezmsg is intended primarily for processing unbounded streaming signals, so are the modules in this repo.

> If you are only interested in offline analysis without concern for reproducibility in online applications, then you should probably look elsewhere.

Processing units include dimensionality reduction, linear regression, and classification that can be initialized with known weights, or adapted on-the-fly with incoming (labeled) data. Machine-learning code depends on `river`, `scikit-learn`, `numpy`, and `torch`.

## Installation

Install from PyPI:

```bash
pip install ezmsg-learn
```

Or install the latest development version:

```bash
pip install git+https://github.com/ezmsg-org/ezmsg-learn@dev
```

## Dependencies

- `ezmsg`
- `ezmsg-baseproc`
- `ezmsg-sigproc`
- `numpy`
- `scipy`
- `scikit-learn`
- `river`


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
