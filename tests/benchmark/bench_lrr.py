"""Performance benchmarks for LRRTransformer.

Run with:
    .venv/bin/python tests/benchmark/bench_lrr.py

Benchmarks:
    1. _process (inference) — numpy, varying chunk sizes
    2. partial_fit (training) — numpy, varying chunk sizes
    3. _process — torch MPS (Apple Silicon GPU)
    4. partial_fit — torch MPS (Apple Silicon GPU)
"""

import time

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.learn.process.ssr import LRRSettings, LRRTransformer

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_CH = 512
N_CLUSTERS = 8
CLUSTER_SIZE = N_CH // N_CLUSTERS  # 64
FS = 30_000.0
CHUNK_SIZES = [20, 50, 100, 150, 200, 300]
WARMUP_ITERS = 20
BENCH_ITERS = 200

CLUSTERS = [list(range(i * CLUSTER_SIZE, (i + 1) * CLUSTER_SIZE)) for i in range(N_CLUSTERS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(data, key: str = "bench") -> AxisArray:
    return AxisArray(
        data=data,
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS, offset=0.0)},
        key=key,
    )


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_row(chunk: int, median_us: float, throughput_khz: float) -> None:
    print(f"  chunk={chunk:>4d}  |  {median_us:8.1f} us/call  |  {throughput_khz:8.1f} kHz effective")


def _bench_loop(fn, n_warmup: int, n_iters: int) -> list[float]:
    """Run fn() for warmup + measured iterations, return list of elapsed times."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _bench_loop_sync(fn, sync_fn, n_warmup: int, n_iters: int) -> list[float]:
    """Like _bench_loop but calls sync_fn() before each timing measurement."""
    for _ in range(n_warmup):
        fn()
    sync_fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        sync_fn()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# NumPy benchmarks
# ---------------------------------------------------------------------------


def bench_process_numpy() -> None:
    _print_header("_process (inference) — NumPy")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    print()

    rng = np.random.default_rng(0)

    # Fit via partial_fit so the message hash is primed for send()
    fit_data = rng.standard_normal((2000, N_CH))
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))
    proc.partial_fit(_make_msg(fit_data))

    for chunk in CHUNK_SIZES:
        data = rng.standard_normal((chunk, N_CH))
        msg = _make_msg(data)
        # Prime — first send triggers the affine's _reset_state
        proc.send(msg)

        times = _bench_loop(lambda: proc.send(msg), WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)  # samples/s
        _print_row(chunk, median_us, throughput / 1e3)


def bench_partial_fit_numpy() -> None:
    _print_header("partial_fit (training) — NumPy")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    print()

    rng = np.random.default_rng(1)
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))

    for chunk in CHUNK_SIZES:
        data = rng.standard_normal((chunk, N_CH))
        msg = _make_msg(data)
        # Prime
        proc.partial_fit(msg)

        times = _bench_loop(lambda: proc.partial_fit(msg), WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)
        _print_row(chunk, median_us, throughput / 1e3)


# ---------------------------------------------------------------------------
# Torch MPS benchmarks
# ---------------------------------------------------------------------------


def bench_process_mps() -> None:
    import torch

    if not torch.backends.mps.is_available():
        print("\n  [SKIPPED] MPS not available")
        return

    _print_header("_process (inference) — Torch MPS")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    print()

    rng = np.random.default_rng(0)
    device = torch.device("mps")

    # Fit on CPU (numpy), then send MPS data to trigger device conversion
    fit_data = rng.standard_normal((2000, N_CH))
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))
    proc.partial_fit(_make_msg(fit_data))

    def sync():
        torch.mps.synchronize()

    for chunk in CHUNK_SIZES:
        data_mps = torch.randn(chunk, N_CH, device=device, dtype=torch.float32)
        msg = _make_msg(data_mps)
        # Prime — first send triggers affine's _reset_state with device conversion
        proc.send(msg)

        times = _bench_loop_sync(lambda: proc.send(msg), sync, WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)
        _print_row(chunk, median_us, throughput / 1e3)


def bench_partial_fit_mps() -> None:
    import torch

    if not torch.backends.mps.is_available():
        print("\n  [SKIPPED] MPS not available")
        return

    _print_header("partial_fit (training) — Torch MPS")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    print()

    _ = np.random.default_rng(1)
    device = torch.device("mps")
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))

    def sync():
        torch.mps.synchronize()

    for chunk in CHUNK_SIZES:
        data_mps = torch.randn(chunk, N_CH, device=device, dtype=torch.float32)
        msg = _make_msg(data_mps)
        # Prime
        proc.partial_fit(msg)

        times = _bench_loop_sync(lambda: proc.partial_fit(msg), sync, WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)
        _print_row(chunk, median_us, throughput / 1e3)


# ---------------------------------------------------------------------------
# MLX benchmarks
# ---------------------------------------------------------------------------


def bench_process_mlx() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        print("\n  [SKIPPED] MLX not installed")
        return

    _print_header("_process (inference) — MLX")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    print()

    rng = np.random.default_rng(0)

    # Fit on CPU (numpy), then send MLX data
    fit_data = rng.standard_normal((2000, N_CH))
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))
    proc.partial_fit(_make_msg(fit_data))

    def sync():
        mx.eval()

    for chunk in CHUNK_SIZES:
        data_mlx = mx.random.normal(shape=(chunk, N_CH))
        msg = _make_msg(data_mlx)
        # Prime — first send triggers affine's _reset_state with MLX conversion
        out = proc.send(msg)
        mx.eval(out.data)

        def run():
            out = proc.send(msg)
            mx.eval(out.data)

        times = _bench_loop(run, WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)
        _print_row(chunk, median_us, throughput / 1e3)


def bench_partial_fit_mlx() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        print("\n  [SKIPPED] MLX not installed")
        return

    _print_header("partial_fit (training) — MLX")
    print(f"  {N_CH} channels, {N_CLUSTERS}x{CLUSTER_SIZE} clusters, {WARMUP_ITERS} warmup, {BENCH_ITERS} iters")
    # MLX linalg.inv doesn't support GPU yet; run inv on CPU stream
    print("  NOTE: linalg.inv runs on mx.cpu stream (GPU not supported)")
    print()

    import mlx.core as mx

    _ = np.random.default_rng(1)
    proc = LRRTransformer(LRRSettings(channel_clusters=CLUSTERS, min_cluster_size=1))

    # Monkey-patch _solve_weights to use mx.cpu stream for inv
    original_solve = proc._solve_weights

    def _solve_weights_cpu_inv(cxx):
        from array_api_compat import get_namespace

        xp = get_namespace(cxx)
        # If this is MLX, we need to override linalg.inv
        if xp.__name__ == "mlx.core":
            orig_inv = mx.linalg.inv
            mx.linalg.inv = lambda a: orig_inv(a, stream=mx.cpu)
            try:
                return original_solve(cxx)
            finally:
                mx.linalg.inv = orig_inv
        return original_solve(cxx)

    proc._solve_weights = _solve_weights_cpu_inv

    for chunk in CHUNK_SIZES:
        data_mlx = mx.random.normal(shape=(chunk, N_CH))
        msg = _make_msg(data_mlx)
        # Prime
        proc.partial_fit(msg)

        def run():
            proc.partial_fit(msg)
            mx.eval()

        times = _bench_loop(run, WARMUP_ITERS, BENCH_ITERS)
        median_us = np.median(times) * 1e6
        throughput = chunk / np.median(times)
        _print_row(chunk, median_us, throughput / 1e3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"LRRTransformer benchmark: {N_CH} channels, {N_CLUSTERS} clusters of {CLUSTER_SIZE}, fs={FS / 1e3:.0f} kHz")

    bench_process_numpy()
    bench_partial_fit_numpy()
    bench_process_mps()
    bench_partial_fit_mps()
    bench_process_mlx()
    bench_partial_fit_mlx()

    print()
    realtime_budget_us = {c: c / FS * 1e6 for c in CHUNK_SIZES}
    print("Real-time budgets at 30 kHz:")
    for chunk, budget in realtime_budget_us.items():
        print(f"  chunk={chunk:>4d}  ->  {budget:8.1f} us")
