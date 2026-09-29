"""Profile the CAMINO pixel fit from camino_pixel_fit_final.ipynb with jax.profiler.

Captures an XProf/TensorBoard trace of (a) repeated stage-2 value_and_grad calls
and (b) a short L-BFGS-B run, so device compute and SciPy/host overhead can be
compared. Compilation happens before tracing starts so it does not swamp the
timeline.

    pip install -e ".[profile]"            # installs xprof
    python notebooks/profile_pixel_fit.py   # from the repo root
    xprof --port 8791 notebooks/profiles    # open http://localhost:8791 -> trace_viewer

Options: --psf-npixels 512 to compare against the old 2048^2 propagation grid,
--perfetto to also print a ui.perfetto.dev link, --dump-hlo to write the
optimised XLA HLO for inspection.
"""

import argparse
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--date", default="2025-12-18")
parser.add_argument("--psf-npixels", type=int, default=None)  # None: fit_npix
parser.add_argument("--n-evals", type=int, default=5)
parser.add_argument("--lbfgs-iters", type=int, default=5)
parser.add_argument(
    "--sgd-steps",
    type=int,
    default=None,
    help="override stage-1 steps (default: reuse checkpoint if present)",
)
parser.add_argument("--log-dir", default=None)
parser.add_argument("--perfetto", action="store_true")
parser.add_argument("--dump-hlo", action="store_true")
args = parser.parse_args()

here = Path(__file__).resolve().parent
run_dir = here / f"{args.date.replace('-', '')}_pixel"
log_dir = Path(args.log_dir or here / "profiles").resolve()
if args.dump_hlo:
    # Must be set before jax is imported.
    hlo_dir = log_dir / f"hlo_psf{args.psf_npixels or 'default'}"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_dump_to={hlo_dir} --xla_dump_hlo_as_text"
    )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from camino.data_utils import download_wlp8_wlm8  # noqa: E402
from camino.fitting import (  # noqa: E402
    FitConfig,
    _make_final_objective,
    _run_optimizer,
    _run_pixel_initial_stage,
    load_data,
)

wlp8, wlm8 = download_wlp8_wlm8(args.date, download_dir=run_dir / "data")
overrides = dict(show_plots=False, progress=False)
if args.psf_npixels is not None:
    overrides["psf_npixels"] = args.psf_npixels
if args.sgd_steps is not None:
    overrides["stage1_sgd_steps"] = args.sgd_steps
problem = load_data(wlp8, wlm8, config=FitConfig.for_mode("pixel", **overrides))

with problem.filter_context():
    checkpoint = run_dir / "stage1_checkpoint.npz"
    if args.sgd_steps is None and checkpoint.is_file():
        saved = np.load(checkpoint)
        params = {k: jnp.asarray(saved[k]) for k in problem.initial_params}
        print(f"Stage-1 parameters from {checkpoint}")
    else:
        params = _run_pixel_initial_stage(problem, [])

    x0, unpack, evaluate = _make_final_objective(problem, params)
    x = jnp.asarray(x0)

    t = time.perf_counter()
    jax.block_until_ready(evaluate(x))
    print(f"compile + first call: {time.perf_counter() - t:.2f} s")

    log_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(log_dir), create_perfetto_link=args.perfetto):
        for step in range(args.n_evals):
            with jax.profiler.StepTraceAnnotation("value_and_grad", step_num=step):
                t = time.perf_counter()
                jax.block_until_ready(evaluate(x))
                print(f"value_and_grad {step}: {time.perf_counter() - t:.3f} s")

        if args.lbfgs_iters:
            with jax.profiler.TraceAnnotation("scipy L-BFGS-B"):
                t = time.perf_counter()
                result = _run_optimizer(
                    evaluate,
                    x0,
                    name="profiled L-BFGS-B",
                    method="L-BFGS-B",
                    options=problem.config.stage2_options(maxiter=args.lbfgs_iters),
                    config=problem.config,
                    histories=[],
                )
                wall = time.perf_counter() - t
            print(
                f"L-BFGS-B: {result.nit} iterations, {result.nfev} evaluations, "
                f"{wall:.2f} s ({wall / max(result.nit, 1):.2f} s/iter, "
                f"{wall / max(result.nfev, 1):.2f} s/eval)"
            )

print(
    f"\nTrace written to {log_dir}\nView with:  xprof --port 8791 {log_dir}",
    file=sys.stderr,
)
