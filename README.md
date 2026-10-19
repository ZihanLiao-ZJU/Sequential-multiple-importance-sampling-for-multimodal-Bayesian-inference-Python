# Sequential Multiple Importance Sampling for Multimodal Bayesian Inference (Python)

Python implementation of the **Sequential Multiple Importance Sampling (SeMIS)** algorithm for multimodal Bayesian inference, with subset simulation (SuS) included for comparison. This is a line-by-line port of the MATLAB code in [Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference), which accompanies the paper:

> **Sequential multiple importance sampling for multi-modal Bayesian inference.**
> *Mechanical Systems and Signal Processing*, 245 (2026) 113788. doi: [10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Benchmark likelihoods](#benchmark-likelihoods)
- [Custom likelihood](#custom-likelihood)
- [Outputs](#outputs)
- [Validation against MATLAB](#validation-against-matlab)
- [Differences from the MATLAB version](#differences-from-the-matlab-version)
- [Citation](#citation)
- [License](#license)

## Overview

SeMIS is a multiple importance sampling (MIS) based algorithm for Bayesian inference with multimodal and high-dimensional posterior distributions. It builds a sequence of **softly truncated, prior-based proposal distributions** that evolve smoothly from the prior to the posterior, which improves mixing across separated modes while still allowing efficient estimation of the Bayesian evidence.

At each iteration, SeMIS:

1. constructs a proposal distribution by softly truncating the prior using likelihood information;
2. adaptively determines the proposal parameter based on a target acceptance rate;
3. generates samples with parallel MCMC chains started from accepted seeds (elliptical slice sampling by default);
4. accumulates all samples through multiple importance sampling with the balance heuristic.

After the last iteration, the Bayesian evidence is estimated and posterior samples are obtained by resampling. All weights are computed in the log domain for numerical stability.

The class structure, method names and variable names follow the MATLAB code, so the two versions can be read side by side.

## Requirements

- Python 3.9 or later
- NumPy and SciPy
- Matplotlib (optional, only for plotting)

```bash
pip install -r requirements.txt
```

## Repository structure

```
.
├── drive_SuS_SeMIS.py                   # driver: runs SeMIS and SuS on a benchmark problem
├── validate.py                          # comparison with the MATLAB implementation
├── semis_bayes/
│   ├── SeMIS_Bay_Nataf.py               # SeMIS intermediate distributions (soft truncation) in standard normal space
│   ├── SuS_Bay_Nataf.py                 # subset simulation for Bayesian updating (comparison)
│   ├── SeMCS.py                         # sequential Monte Carlo loop shared by SeMIS and SuS
│   ├── MIS.py                           # MIS estimator and posterior resampling
│   ├── Nataf.py                         # Nataf transformation between physical and standard normal space
│   ├── distributions.py                 # `makedist`: MATLAB-style distribution objects on top of scipy.stats
│   ├── mcmc_samplers/                   # elliptical slice sampling (ES.py) and adaptive conditional sampling (aCS.py)
│   ├── likelihood_functions/            # benchmark likelihoods with reference log-evidence
│   ├── support_functions/               # weights, seed selection and other helpers
│   └── log_operations.py                # log-domain arithmetic (logsum, logplus, logmean, ...)
├── validation/                          # MATLAB random streams and results used by validate.py
├── LICENSE
└── README.md
```

## Quick start

Run the driver from the repository root:

```bash
python drive_SuS_SeMIS.py
```

The driver defines the benchmark problem (by default `LKF_nD_GaussianLogGamma` with `Ndim = 2`), runs SeMIS and SuS for 100 independent runs, and prints the evidence and efficiency statistics described below. `--nrun` changes the number of runs, `--seed` fixes the random seed and `--quiet` hides the progress of each iteration. To change the problem, edit the `Tfun = ...` line and `Ndim` at the top of `drive_SuS_SeMIS.py`. The sample size per level is set by `smc_SeMIS.NumSam` (default 1700) and `smc_SuS.NumSam` (default 1000).

The same steps from Python code:

```python
from semis_bayes import LKF_nD_GaussianLogGamma, SeMIS_Bay_Nataf, SeMCS, ES, MIS

Tfun = LKF_nD_GaussianLogGamma(2)
smc_SeMIS = SeMCS(ES(SeMIS_Bay_Nataf(Tfun)))
smc_SeMIS.NumSam = 1700

out_sm = smc_SeMIS.RunIte()          # SMC iterations
out_mis = MIS(out_sm)                # MIS post-processing
print(out_mis.Z)                     # log-evidence (MIS estimator)
u_pos = out_mis.PosSim()[0]          # posterior samples in standard normal space
x_pos = Tfun.Pdis.U2X(u_pos)         # posterior samples in the parameter space
```

## Benchmark likelihoods

| File | Description | Reference log-evidence |
|---|---|---|
| `LKF_2D_Eggbox.py` | 2D eggbox likelihood, uniform prior on [0, 10π]² | 235.856 |
| `LKF_nD_GaussianLogGamma.py` | Gaussian + LogGamma, n-dimensional | −8.19 (n = 2), −20.47 (n = 5), −40.94 (n = 10), −81.887 (n = 20) |
| `LKF_nD_GaussianShells.py` | two Gaussian shells, n-dimensional | −1.75 (n = 2) to −255.62 (n = 100); see the class docstring |

Each likelihood class provides the prior (`Pdis`), the log-likelihood and the reference value `logz_ref` used by the driver. `LKF_2D_Eggbox` and `LKF_nD_GaussianLogGamma` also provide the reference marginals, `KStest` (Kolmogorov–Smirnov distance of posterior samples to the reference marginal) and `PltFun`. To solve your own problem, see [Custom likelihood](#custom-likelihood).

## Custom likelihood

Any user-defined Bayesian updating problem can be solved by writing a likelihood class with the same interface as the files in `semis_bayes/likelihood_functions/` and passing it to the driver in place of the benchmark. The algorithm only accesses the following members:

| Member | Type | Description |
|---|---|---|
| `Ndim` | int | number of uncertain parameters |
| `Pdis` | `Nataf` object | prior distribution, built as `Nataf(Marg_X, Rho_X)` from a list of distribution objects (`makedist`) and their correlation matrix |
| `EvlLKF(self, theta)` | method | log-likelihood of the samples `theta` (`Ndim × Nsam` array, in the physical parameter space); must return an array of length `Nsam` |
| `logz_ref` | float | reference log-evidence; `drive_SuS_SeMIS.py` uses it only to compute the relative bias, so set it to `np.nan` if unknown |

The sampler works in standard normal space and maps samples to the parameter space through `Pdis.U2X` before calling `EvlLKF`, so the likelihood is written directly in terms of the physical parameters. `EvlLKF` is called with many samples at once and must return the **logarithm** of the likelihood.

`makedist` takes MATLAB's distribution names and parameter names (`"Normal"` with `mu`, `sigma`; `"Lognormal"`; `"Uniform"` with `lower`, `upper`; `"GeneralizedExtremeValue"`; `"Gamma"`; `"Weibull"`; ...). Any frozen `scipy.stats` distribution can also be used through `Distribution("Name", frozen)`.

Minimal example: two parameters with independent standard normal priors and a Gaussian likelihood for two noisy observations.

```python
import numpy as np
from semis_bayes import Nataf, makedist


class MyLikelihood:
    def __init__(self):
        self.Ndim = 2
        self.Pdis = Nataf([makedist("Normal", mu=0, sigma=1)] * 2, np.eye(2))
        self.logz_ref = np.nan   # reference log-evidence, used only for the bias statistics (NaN if unknown)

    def EvlLKF(self, theta):
        # theta : [Ndim x Nsam] parameter samples
        # logL  : [Nsam] log-likelihood
        y_obs = np.array([[1.0], [-0.5]])   # observed data
        sigma = 0.2                         # measurement noise standard deviation
        return -0.5 * np.sum((theta - y_obs) ** 2, axis=0) / sigma ** 2 - y_obs.size * np.log(np.sqrt(2 * np.pi) * sigma)
```

Then, in `drive_SuS_SeMIS.py`, replace the benchmark with your class:

```python
Tfun = MyLikelihood()
```

The rest of the driver is unchanged.

## Outputs

During the run, each iteration prints the sample size, the number of chains, the samples per chain and the number of likelihood evaluations. After all runs, the driver prints a summary table and keeps the following variables:

| Variable | Meaning |
|---|---|
| `logz_SeMIS_MIS_bias`, `logz_SeMIS_MIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, MIS estimator |
| `logz_SeMIS_SIS_bias`, `logz_SeMIS_SIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, SIS estimator |
| `logz_SuS_bias`, `logz_SuS_cv` | relative bias and CoV (%) of the SuS log-evidence |
| `Npos_Ncal_SeMIS_mu`, `Npos_Ncal_SuS_mu` | posterior samples per likelihood evaluation (%) |
| `Ncal_SeMIS_mu`, `Ncal_SuS_mu` | average number of likelihood evaluations per run |
| `x_pos_SeMIS`, `x_pos_SuS` | posterior samples in the physical parameter space (last run) |

## Validation against MATLAB

```bash
python validate.py
```

1. **Exact replay.** `validation/reference/` contains, for each recorded case, every random number drawn by one SeMIS run and one SuS run of the original MATLAB code (one iteration of `drive_SuS_SeMIS.m`), together with its results. `validate.py` feeds the same random numbers to the Python code, which then reproduces the MATLAB run: the same proposal parameters, likelihood-call counts, log-evidence (MIS, SIS and SuS estimators) and resampled posterior samples, with differences at round-off level.
2. **Statistics.** 50 independent Python runs per benchmark (`--nrun` to change); the mean log-evidence of SeMIS (MIS estimator) and SuS is compared with the reference value.

Results of `python validate.py` (50 runs per benchmark):

| Case | Replay vs MATLAB | SeMIS mean logZ (std) | SuS mean logZ (std) | Reference |
|---|---|---|---|---|
| Gaussian + LogGamma, n = 2 | identical (2 seeds, see note) | −8.193 (0.074) | −8.200 (0.130) | −8.189 |
| Gaussian + LogGamma, n = 5 | identical | −20.465 (0.196) | −20.533 (0.280) | −20.472 |
| Gaussian shells, n = 2 | identical | −1.751 (0.074) | −1.774 (0.112) | −1.75 |
| Gaussian shells, n = 5 | identical | −5.677 (0.148) | −5.709 (0.166) | −5.67 |
| Eggbox, n = 2 | identical | 235.856 (0.123) | 235.838 (0.160) | 235.856 |

All means are within 2 standard errors of the reference value.

**Note on the second Gaussian + LogGamma replay.** At the last level, the proposal parameter `r_i` is searched with `fminbnd` on 100 sub-intervals, and in this run the objective is exactly flat between the two candidates (the same value, 2.225933576697242, at both). Which one is kept is then decided by round-off in the objective, and Python and MATLAB keep different ones (−1.17e-4 and −5.03e-5); since only the second is above the convergence threshold −1e-4, the Python run takes one more level. `validate.py` detects this, checks that both values are equally good, and replays the run again with MATLAB's `[r_i, r_max]` fixed (`SeMCS.G`): everything else is then identical. The same flat objective can make the MATLAB code itself take a different path on another machine.

The reference streams were recorded by running the original MATLAB code under GNU Octave 8.4 (no MATLAB licence was available) with the scripts in `validation/matlab/`: they shadow `randn`, `rand`, `randperm`, `randi` and `randsample` to log every draw, and `validation/matlab/octave/` provides Octave stand-ins for `makedist`, `pagemtimes` and the `"all"` option of `max`, `min` and `sum`, as well as an `fminbnd` with MATLAB's default tolerance (Octave's own default differs). Octave compares complex numbers by their modulus, so for the recording the sign tests in `logsum`, `logplus` and `logminus` were applied to `real(...)`, which is what MATLAB compares. MATLAB does not need any of these changes. To record them with MATLAB instead:

```matlab
>> cd validation/matlab
>> record_reference('/path/to/Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference', 'loggamma', 2, 1, '../reference')
```

## Differences from the MATLAB version

- `RunIte` and `SuS_Bay_Nataf.BayInf` return dictionaries instead of structs, and `MIS.PosSim` returns a tuple `(u_pos, y_pos, u_all, y_all)`.
- Arrays keep the MATLAB layout (`Ndim × Nc × Ns`, flattened in column-major order); Python indices start at 0.
- `fminbnd` is `scipy.optimize.minimize_scalar(method="bounded")` with MATLAB's default tolerance (`TolX = 1e-4`); it implements the same golden-section and parabolic-interpolation search.
- Random numbers are drawn through `semis_bayes._utils` (`randn`, `rand`, `randperm`, `randi`, `randsample`), backed by NumPy; `np.random.seed` fixes a run.
- `MIS.CCDF` and `MIS.KLDEst` are not ported: they call `IndPriWgt`, `IndPosWgt` and `EpyWgt`, which are not part of the MATLAB repository, and the driver does not use them. `LKF_2D_5Gaussian`, mentioned in a comment of `drive_SuS_SeMIS.m`, is not in the MATLAB repository either.
- The reference marginals of `LKF_2D_Eggbox` (used only by `KStest`) are computed on first use rather than in the constructor.

## Citation

If you use this code in your research, please cite:

> Li, B., He, X., & Liao, Z. (2026). Sequential multiple importance sampling for multi-modal Bayesian inference. *Mechanical Systems and Signal Processing*, 245, 113788. https://doi.org/10.1016/j.ymssp.2025.113788

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

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
