---
title: MixSIAPy
emoji: 🌊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
fullWidth: true
header: mini
suggested_hardware: cpu-upgrade
short_description: Bayesian stable-isotope mixing models in Python
---

# MixSIAPy

MixSIAPy is a native Python implementation of Bayesian tracer mixing models inspired by MixSIAR. It uses PyMC for inference and ArviZ for posterior storage and diagnostics; JAGS is not required at runtime.

> **Project status: research preview.** The principal model structures and the official example workflows have been ported, but full numerical parity with every R/JAGS configuration is still being validated. Do not describe this release as a drop-in or 100% verified replacement for MixSIAR.

## Implemented model structures

- summary (mean/SD/sample size) and raw source data;
- source data stratified by a factor and concentration dependence;
- fixed, random, nested, and continuous effects in ILR coordinates;
- generalized ILR compositional regression with multiple numeric/categorical
  predictors and two-way interactions (`crop_water_uptake.py`);
- process × residual, residual-only, and process-only error structures;
- posterior summaries, convergence diagnostics, model comparison, source aggregation, continuous-effect prediction, and plotting;
- explicit Python ports of the example scripts distributed with MixSIAR 3.1.12.

## Installation

Python 3.9 or newer is required. Create an isolated environment and install the Bayesian and testing dependencies:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[bayes,test]"
python -m pytest
```

For a lightweight installation that only loads and checks data:

```powershell
python -m pip install -e .
```

## Graphical interface

Install the GUI together with the Bayesian dependencies and launch it from a
terminal. It opens in the default browser, but all computation remains local.

```powershell
python -m pip install -e ".[bayes,gui]"
mixsiapy-gui
```

The GUI validates uploaded mixture/source/discrimination CSV files, configures
fixed/random/continuous effects and generalized compositional regression,
selects the inference backend and CPU/GPU device, displays diagnostics and
figures, and downloads a ZIP containing the complete posterior plus
`analysis_config.json` and an editable `reproduce_analysis.py` script.

### Windows desktop release

The Windows release provides the same local graphical interface without requiring
users to install Python. Extract the complete release archive and double-click
`MixSIAPy.exe`; the application starts a service on the local computer and opens the
interface in the default browser. The controller window must remain open while the
application is in use. The portable desktop build includes CPU sampling with the
default PyMC backend. Optional Nutpie, JAX, NumPyro, BlackJAX, and GPU environments
remain available through the Python installation.

## Quick start

Each example is self-contained and follows the corresponding R script without a hidden common configuration module. The installed examples are available through `get_resource_path("examples")`.

```powershell
python -c "from mixsiapy import get_resource_path; print(get_resource_path('examples','wolves.py'))"
python C:\path\printed\above\wolves.py --run test
python C:\path\printed\above\wolves.py --build-only
```

Running `wolves.py` directly performs a short test sample and writes posterior NetCDF, CSV summaries, diagnostics, and figures to `outputs/wolves/`. `--build-only` skips sampling. Use `--output <directory>` to select another destination. On Windows, chains are run sequentially by default to avoid multiprocessing import problems in older PyMC/ArviZ installations.

### Spyder

Spyder must use the same Python environment in which MixSIAPy is installed. Locate that interpreter with `python -c "import sys; print(sys.executable)"`, select it in Spyder, and restart the kernel after installing or updating the package. Open the installed example returned by `get_resource_path("examples", "wolves.py")` and click **Run**. The variables `fit`, `model`, `mix`, `source`, and `discr` remain visible in Variable Explorer, while complete files are saved under `outputs/wolves` relative to Spyder's working directory.

To verify which installation Spyder imports, run in its console:

```python
import mixsiapy
print(mixsiapy.__file__)
print(mixsiapy.get_resource_path("data"))
```

Both paths should point to `mixsiapy` inside the active Python environment's `site-packages` directory.

The wheel installs `data/`, `examples/`, `docs/`, and `reference_r/` directly inside the `mixsiapy` package directory (for example, `Lib/site-packages/mixsiapy/data`). Locate them without assuming an environment directory:

```python
from mixsiapy import get_resource_path

wolves_data = get_resource_path("data", "wolves_consumer.csv")
wolves_example = get_resource_path("examples", "wolves.py")
```

Minimal library usage:

```python
from pathlib import Path
from mixsiapy import load_mix_data, load_source_data, load_discr_data, run_model

data = Path("data")
mix = load_mix_data(
    data / "wolves_consumer.csv",
    ["d13C", "d15N"],
    ["Region", "Pack"],
    [True, True],
    [False, True],
)
source = load_source_data(data / "wolves_sources.csv", "Region", False, "means", mix)
discr = load_discr_data(data / "wolves_discrimination.csv", mix)
fit = run_model("test", mix, source, discr, random_seed=42, target_accept=0.95)
```

## Repository layout

- `.github/`: automated testing and deployment workflows;
- `mixsiapy/`: supported API plus installed `data/`, `examples/`, `docs/`, `gui/`, and `reference_r/` resources;
- `tests/`: automated Python unit and integration tests;
- `benchmarks/`: reproducible numerical-agreement, backend-performance, and GPU-scaling experiments.

Generated files in `outputs/` are ignored by Git. The supported API is the `mixsiapy` package; code under `reference_r/` is not imported at runtime.

## Validation and reproducibility

Run the fast test suite with `python -m pytest`. The validation status and the planned R-versus-Python comparison matrix are documented in [`mixsiapy/docs/VALIDATION.md`](mixsiapy/docs/VALIDATION.md). Posterior agreement must be evaluated with sufficiently converged chains; results from runs with large R-hat values or many divergences are not valid reference targets.

## Contributing and citation

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Citation metadata will be added before the first archived release, after the author list, repository URL, and release DOI are confirmed.

## License and attribution

This project is distributed under the GNU General Public License v3.0. MixSIAPy is an independent Python port and is not presented as an official release of the R MixSIAR authors. The frozen reference code retains its original authorship and GPL-3 licensing information.
