# Sequential Multiple Importance Sampling for Bayesian Inference — Python

## Overview
Constructs adaptive intermediate distributions in Nataf standard-normal space, then estimates log Bayesian evidence using multiple importance sampling (MIS). The driver also reports SIS and SuS reference estimates. The default sampler is ES. Optional kernels do not mean every combination was evaluated in the cited paper.

## Structure
- `drive_SuS_SeMIS.py` — repeated-run example driver.
- `semis_bayes/SeMCS.py` — sequential sampling loop.
- `semis_bayes/SeMIS_Bay_Nataf.py` and `semis_bayes/SuS_Bay_Nataf.py` — intermediate distributions, seeds and stopping conditions.
- `semis_bayes/MIS.py` — evidence and posterior post-processing.
- `semis_bayes/likelihood_functions/` — bundled likelihood/prior models.
- `semis_bayes/mcmc_samplers/` — four kernels and their shared `MCMCBase` protocol.
- `tests/` and `validation/` — regression and cross-language checks.

## Quick start
Run from this repository's root. Each checkout contains its own helpers and does not require a sibling repository.

```bash
python -m pip install -r requirements.txt
python drive_SuS_SeMIS.py --nrun 1 --seed 1 --sampler ES
```

Inspect both the returned loop flag and `intBay.FlgCvg` (the intermediate model's terminal condition). A zero-seed exit can precede the terminal check; a finite evidence estimate alone is not proof of convergence. Likelihood calls are reported by `Ncal`, not by the number of posterior resamples.

## Examples and samplers
| Example | Model / dimension | Selector |
|---|---|---|
| Eggbox | Two-dimensional multimodal likelihood | `LKF_2D_Eggbox()` |
| GaussianShells | Dimension supplied by the caller | `LKF_nD_GaussianShells(Ndim)` |
| GaussianLogGamma | Dimension supplied by the caller, at least 2 | `LKF_nD_GaussianLogGamma(Ndim)` |

The existing driver uses 2D GaussianLogGamma, 1700 SeMIS samples and 1000 SuS samples per level, with 100 repetitions. The default conditional probability is 0.1. The quick start is a small usage example, not a paper-reproduction experiment.

Available samplers are `CW_aMH`, `aCS`, `HMC`, and `ES`. Use `--sampler aCS` (or another listed kernel).

All kernels implement `SamGen(Nchains, Nsteps)` or `SamGen(U_seed, Y_seed, Nsteps)` and return samples, packed outputs, evaluation count and updated Bayesian state. Arrays use `Ndim × Nchains × Nsteps`; packed output rows are the intermediate log-likelihood factor, Gaussian log density and cached model outputs. Sampler dimensions are model-defined, not hard-coded to two.

## Dependencies
Python with NumPy and SciPy (see `requirements.txt`); Matplotlib is optional for plots. Tests use the standard-library `unittest` runner. Python 3.12 was used for the focused checks; this is not a claim that every earlier Python version was tested.

## Method & validation notes
- **Implementation:** Intermediate thresholds, evidence/posterior formulas and model definitions are separate from the MCMC kernels. Numeric threshold vectors are stored in `G.parameters` (Python `G["parameters"]`); adaptation state occupies separate fields. The aCS `lambda` is a sampler scale, not the SeMIS intermediate-distribution threshold.
- **Sampler policy:** aCS uses initial coordinate scale 1 and initial lambda 0.6, passing the final lambda to the next level. ES uses coordinatewise initial angles in `[0, pi]`. HMC uses analytic Gaussian-reference rotations with likelihood acceptance in Nataf space, not generic physical-space gradient integration.
- **Outputs:** `MIS.Z` is natural-log evidence. Driver columns based on `100*(logZ/logZ_ref-1)` are its empirical mean and SD, not CoV of evidence. `Npos/Ncal` is the rounded Kish-weight-count ratio used to set the posterior resample count; it is not a correlation-adjusted posterior-mean ESS. Resampled posterior draws may repeat or remain dependent.
- **Validation scope:** Interface, same-seed trajectory and evaluation-accounting checks are not proof of agreement with all paper statistics. Historical aCS uses seed-based scales/reset adaptation, so the current policy can change termination and results. Stopping rules have not been relaxed to hide unsuccessful runs.
 The bundled suite includes one expected failure for the inherited all-negative signed-log reduction limitation.
- **History:** The Python port retains necessary signed-log, distribution-wrapper and reference-grid parity fixes. Nataf follows the original MATLAB CDF-to-inverse-CDF composition without tail clipping and its original positive-only nonstandard-correlation search interval; extreme normal/lognormal tails may therefore produce infinities, and negative nonstandard correlations are not correctly fitted. Arbitrary model dimensions and trailing sample axes are retained. The public GaussianShells model has no matching example row in the final SeMIS journal paper. The estimator, stopping rules, examples and driver defaults are unchanged.


- **Correlation scope:** Bundled examples use independent priors. The inherited MATLAB nonstandard-correlated-marginal branch references `GauIntPot`, which is not bundled; that branch is not independently runnable without the original helper. Python has its own quadrature implementation, but preserves the original positive-only search interval rather than extending negative-correlation support.

## Reference & license
[Paper DOI: 10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788)
```bibtex
@article{li2026sequential,
  title   = {Sequential multiple importance sampling for multi-modal Bayesian inference},
  author  = {Li, Binbin and He, Xiao and Liao, Zihan},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {245},
  pages   = {113788},
  year    = {2026},
  doi     = {10.1016/j.ymssp.2025.113788}
}
```

Licensed under the [MIT License](LICENSE); the original copyright notice is retained.
