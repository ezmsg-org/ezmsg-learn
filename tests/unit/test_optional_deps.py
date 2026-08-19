"""Guards for the optional-backend split.

``torch`` and the sklearn stack ship as extras (``ezmsg-learn[torch]`` /
``ezmsg-learn[sklearn]``) so that the pure-numpy processors can be installed on
constrained hosts. Two properties keep that promise:

1. The base-install modules must not reach a backend, even transitively. This is
   checked in a subprocess, since the test session itself imports everything.
2. A module belonging to an extra must fail with a message naming that extra.
   Only checkable on an install *without* the extra, so those tests skip in the
   full dev environment and run in CI's minimal-install job.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BACKEND_MODULES = ("torch", "sklearn", "river", "pandas")

SRC = Path(__file__).resolve().parents[2] / "src"

#: Importable with no extras installed.
LIGHT_MODULES = [
    "ezmsg.learn.util",
    "ezmsg.learn.process.ssr",
    "ezmsg.learn.process.flatten",
    "ezmsg.learn.process.seqseqsampler",
    "ezmsg.learn.process.refit_kalman",
    "ezmsg.learn.model.cca",
    "ezmsg.learn.model.refit_kalman",
    "ezmsg.learn.linear_model.cca",
    "ezmsg.learn.collection.sample_adapt_regressor",
]

TORCH_MODULES = [
    "ezmsg.learn.model.mlp",
    "ezmsg.learn.model.mlp_old",
    "ezmsg.learn.model.rnn",
    "ezmsg.learn.model.transformer",
    "ezmsg.learn.nlin_model.mlp",
    "ezmsg.learn.process.base",
    "ezmsg.learn.process.torch",
    "ezmsg.learn.process.rnn",
    "ezmsg.learn.process.transformer",
    "ezmsg.learn.process.mlp_old",
]

SKLEARN_MODULES = [
    "ezmsg.learn.dim_reduce.adaptive_decomp",
    "ezmsg.learn.dim_reduce.incremental_decomp",
    "ezmsg.learn.process.adaptive_linear_regressor",
    "ezmsg.learn.process.linear_regressor",
    "ezmsg.learn.process.sgd",
    "ezmsg.learn.process.slda",
    "ezmsg.learn.process.sklearn",
    "ezmsg.learn.linear_model.adaptive_linear_regressor",
    "ezmsg.learn.linear_model.linear_regressor",
    "ezmsg.learn.linear_model.sgd",
    "ezmsg.learn.linear_model.slda",
]

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _backends_loaded_by(source: str) -> list[str]:
    """Run ``source`` in a fresh interpreter; return the backends it left in ``sys.modules``."""
    script = textwrap.dedent(source) + textwrap.dedent(f"""
        import sys
        print(" ".join(m for m in {BACKEND_MODULES!r} if m in sys.modules))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return proc.stdout.split()


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_module_loads_no_backend(module: str) -> None:
    """A base-install module must not pull in torch or the sklearn stack."""
    loaded = _backends_loaded_by(f"import importlib; importlib.import_module({module!r})")
    assert not loaded, f"{module} transitively imported {loaded}"


def test_kalman_collection_loads_no_backend() -> None:
    """``model_type="kalman"`` needs neither extra, so building it must load neither."""
    loaded = _backends_loaded_by("""
        from ezmsg.learn.collection.sample_adapt_regressor import (
            SampleAdaptRegressorSettings,
            build_sample_adapt_regressor,
        )

        build_sample_adapt_regressor(SampleAdaptRegressorSettings(model_type="kalman"))
    """)
    assert not loaded, f"the kalman collection transitively imported {loaded}"


@pytest.mark.skipif(HAS_TORCH, reason="torch installed; the missing-extra path is unreachable")
@pytest.mark.parametrize("module", TORCH_MODULES)
def test_torch_module_names_its_extra(module: str) -> None:
    with pytest.raises(ImportError, match=r"ezmsg-learn\[torch\]"):
        importlib.import_module(module)


@pytest.mark.skipif(HAS_SKLEARN, reason="scikit-learn installed; the missing-extra path is unreachable")
@pytest.mark.parametrize("module", SKLEARN_MODULES)
def test_sklearn_module_names_its_extra(module: str) -> None:
    with pytest.raises(ImportError, match=r"ezmsg-learn\[sklearn\]"):
        importlib.import_module(module)


@pytest.mark.skipif(HAS_SKLEARN, reason="scikit-learn installed; the missing-extra path is unreachable")
def test_regressor_registry_names_its_extra() -> None:
    """``util`` itself stays importable; only the registries need the extra."""
    from ezmsg.learn import util

    with pytest.raises(ImportError, match=r"ezmsg-learn\[sklearn\]"):
        util.get_regressor("adaptive", "linear")

    with pytest.raises(ImportError, match=r"ezmsg-learn\[sklearn\]"):
        util.ADAPTIVE_REGRESSORS


def test_util_module_getattr_rejects_unknown_names() -> None:
    from ezmsg.learn import util

    with pytest.raises(AttributeError):
        util.NOT_A_REGISTRY
