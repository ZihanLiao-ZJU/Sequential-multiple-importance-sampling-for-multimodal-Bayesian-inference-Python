# Sequential Multiple Importance Sampling for Multimodal Bayesian Inference (Python)

Python implementation of the **Sequential Multiple Importance Sampling (SeMIS)** algorithm for multimodal Bayesian inference, with subset simulation (SuS) included for comparison. This is a Python port of the MATLAB code in [Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference), which accompanies the paper:

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
- [Tests](#tests)
- [Implementation notes](#implementation-notes)
- [Citation](#citation)
- [License](#license)

## Overview

Given observed data $D$ and uncertain parameters $\theta$, Bayesian updating follows Bayes' theorem:

$$
p(\theta \mid D)
= \frac{p(D \mid \theta)\,p(\theta)}{p(D)},
\qquad
Z = p(D) = \int p(D \mid \theta)\,p(\theta)\,\mathrm{d}\theta.
$$

Here, $p(\theta)$ is the **prior distribution**, $p(D \mid \theta)$ is the **likelihood**, $p(\theta \mid D)$ is the **posterior distribution**, and $Z = p(D)$ is the **Bayesian evidence (marginal likelihood)** that normalizes the posterior.

The toolbox provides two main outputs:

1. an **evidence (marginal likelihood) estimate** $\widehat{Z}$, computed and reported in the natural-log domain as $\log \widehat{Z}$ for numerical stability;
2. **posterior distribution samples**, which give a Monte Carlo approximation to $p(\theta \mid D)$.

The evidence is an estimate, not an exact integral. Posterior samples are obtained by resampling MCMC-generated samples; they may be repeated or dependent and should not be treated as exact independent draws.

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
├── validate.py                          # recorded-reference and statistical checks
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
│   ├── operations_on_logarithmic_values/  # one .py file per MATLAB log function
│   └── log_operations.py                # backward-compatible imports
├── validation/                          # recorded fixtures used by validate.py
├── LICENSE
└── README.md
```

The logarithmic helpers use matching MATLAB/Python filenames: `logsum`,
`logmean`, `logplus`, `logminus`, `logmultiply`, `logcumsum`, and `logstd`.
Their Python implementations are in `operations_on_logarithmic_values/`;
existing imports from `log_operations` remain supported.


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

The sampler works in standard normal space and maps samples to the parameter space through `Pdis.U2X` before calling `EvlLKF`, so the likelihood is written directly in terms of the physical parameters. `EvlLKF` is called with many samples at once and must return the **logarithm** of the likelihood. Retain the likelihood normalization constants when estimating absolute evidence: omitting a parameter-independent factor leaves the normalized posterior unchanged but rescales the evidence.

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

The post-processing object `out_mis = MIS(out_sm)` provides both outputs from a single SeMIS run:

- **Evidence:** `out_mis.Z` is $\log \widehat{Z}$ from the MIS estimator, not $\widehat{Z}$ itself. `np.exp(out_mis.Z)` converts it to the evidence estimate when representable; keeping the log value avoids underflow or overflow.
- **Posterior samples:** `u_pos = out_mis.PosSim()[0]` returns samples in standard normal space. `x_pos = Tfun.Pdis.U2X(u_pos)` maps them to the physical parameter space, with one posterior sample per column.

During the run, each iteration prints the sample size, the number of chains, the samples per chain and the number of likelihood evaluations. After all runs, the driver prints a summary table and keeps the following variables:

| Variable | Meaning |
|---|---|
| `logz_SeMIS_MIS`, `logz_SeMIS_SIS`, `logz_SuS` | per-run estimates of the natural log-evidence, $\log \widehat{Z}$ |
| `logz_SeMIS_MIS_bias`, `logz_SeMIS_MIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, MIS estimator |
| `logz_SeMIS_SIS_bias`, `logz_SeMIS_SIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, SIS estimator |
| `logz_SuS_bias`, `logz_SuS_cv` | relative bias and CoV (%) of the SuS log-evidence |
| `Npos_Ncal_SeMIS_mu`, `Npos_Ncal_SuS_mu` | posterior samples per likelihood evaluation (%) |
| `Ncal_SeMIS_mu`, `Ncal_SuS_mu` | average number of likelihood evaluations per run |
| `x_pos_SeMIS`, `x_pos_SuS` | posterior samples in the physical parameter space (last run) |

## Tests

Run the unit tests and recorded-reference checks from the repository root:

```bash
python -m unittest discover -s tests -v
python validate.py --replay
```

The bundled replay records the historical sampler policy, not a current-kernel equivalence guarantee. The replay command requires all six recording/result pairs in `validation/reference/`. Individual fixtures can be selected with `--case`, for example:

```bash
python validate.py --replay --case eggbox_2_1 --case shells_2_1
```

## Implementation notes

- `RunIte` and `SuS_Bay_Nataf.BayInf` return dictionaries; `MIS.PosSim` returns `(u_pos, y_pos, u_all, y_all)`.
- Arrays use the `Ndim × Nc × Ns` layout and column-major flattening. Python indices start at 0.
- `fminbnd` uses `scipy.optimize.minimize_scalar(method="bounded")` with tolerance `1e-4`.
- Random draws use NumPy through `semis_bayes._utils`; `np.random.seed` sets the random seed.
- `MIS.CCDF` and `MIS.KLDEst` are not available in this package. The drivers do not use these methods.
- Eggbox reference marginals for `KStest` are computed on first use.
- `ResRat` returns the `[Nc, Ns]` chain layout. Use `ratio.reshape(-1, 1, order="F")` for a column vector.
- Parameterized Normal, Lognormal and GEV distributions use direct transforms when the required attributes exist; generic frozen SciPy wrappers use inverse-CDF mapping.
- Nataf correlation conversion supports positive, negative and zero target correlations.
- Signed-log arithmetic does not support slices containing only negative terms, including all-negative cumulative prefixes. The unit suite records this case as an expected failure. The benchmark importance weights are nonnegative.

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


## Unified MCMC interface

Each checkout contains all four interchangeable kernels: `CW_aMH`, `aCS`,
`HMC`, and `ES`. Set `SamplerClass` in the MATLAB driver, or pass
`--sampler CW_aMH|aCS|HMC|ES` to the Python driver. Existing default choices
are retained (aCS for the standalone SuS toolbox, ES for SeMIS).

All kernels use `BayStc` and return `(u, y, Ncal, BayStc)` from
`SamGen(Nchains, Nsteps)` or `SamGen(u_seed, y_seed, Nsteps)`. Samples have
shape `Ndim x Nchains x Nsteps`; packed outputs contain the intermediate
log-likelihood factor, Gaussian log density, then cached model outputs.
There is no two-dimensional restriction on the sampler interface.

Intermediate state `G` is now a MATLAB struct / Python dict. The former
numeric threshold vector is unchanged in order and stored in `G.parameters`
(Python `G["parameters"]`). The aCS adaptation starts with sigma=1 and
lambda=0.6; its final `G.lambda` is inherited by the next level, rather than
reset. Prescribed legacy numeric threshold sequences are wrapped on input.

ES uses coordinatewise initial angular brackets `[0, pi]`. HMC uses an
analytic Gaussian rotation in Gaussian/Nataf coordinates, with a
Metropolis correction for the intermediate likelihood factor. It is not
a generic physical-space gradient HMC implementation.

The unified kernels change some random trajectories, particularly the
requested aCS initialization/state propagation. Smoke tests alone do not
establish reproduction of all numerical results in the paper.
