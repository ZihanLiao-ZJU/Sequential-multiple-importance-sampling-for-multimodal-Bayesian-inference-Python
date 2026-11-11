# Sequential Multiple Importance Sampling for Multimodal Bayesian Inference (Python)

Python implementation of the **Sequential Multiple Importance Sampling (SeMIS)** algorithm for multimodal Bayesian inference, with subset simulation (SuS) included for comparison. This is a line-by-line port of the MATLAB code in [Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference](https://github.com/ZihanLiao-ZJU/Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference), which accompanies the paper:

> Li, B., He, X., & Liao, Z. (2026). Sequential multiple importance sampling for multi-modal Bayesian inference. *Mechanical Systems and Signal Processing, 245*, Article 113788. [https://doi.org/10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788)

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Built-in examples](#built-in-examples)
- [Using a custom likelihood](#using-a-custom-likelihood)
- [Outputs](#outputs)
- [Differences from the MATLAB version](#differences-from-the-matlab-version)
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

```text
.
├── LICENSE
├── README.md
├── drive_SuS_SeMIS.py
├── requirements.txt
└── semis_bayes/
    ├── MIS.py
    ├── Nataf.py
    ├── SeMCS.py
    ├── SeMIS_Bay_Nataf.py
    ├── SuS_Bay_Nataf.py
    ├── __init__.py
    ├── _utils.py
    ├── distributions.py
    ├── likelihood_functions/
    │   ├── LKF_2D_Eggbox.py
    │   ├── LKF_nD_GaussianLogGamma.py
    │   ├── LKF_nD_GaussianShells.py
    │   ├── __init__.py
    │   └── _marginals.py
    ├── mcmc_samplers/
    │   ├── CW_aMH.py
    │   ├── ES.py
    │   ├── HMC.py
    │   ├── LICENSE
    │   ├── MCMCBase.py
    │   ├── __init__.py
    │   └── aCS.py
    ├── operations_on_logarithmic_values/
    │   └── __init__.py
    └── support_functions/
        ├── DspRst.py
        ├── MisInt.py
        ├── ResRat.py
        ├── SedSlt.py
        ├── VabCmb.py
        ├── WgtMis_Blc.py
        ├── WgtPos.py
        ├── __init__.py
        └── logGauss.py
```

## Quick start

Run the driver once from the repository root with the `ES` sampler:

```bash
python drive_SuS_SeMIS.py --nrun 1 --sampler ES
```

The driver runs SeMIS and SuS on `LKF_nD_GaussianLogGamma(2)`. In `drive_SuS_SeMIS.py`, change `Tfun` and `Ndim` to choose the problem, and set the scalar sample sizes at `smc_SeMIS.NumSam = 1700` and `smc_SuS.NumSam = 1000`. Use `--sampler` to choose the MCMC sampler; `--seed` fixes the random seed and `--quiet` hides iteration progress.

To run only SeMIS once from Python:

```python
from semis_bayes import LKF_nD_GaussianLogGamma, SeMIS_Bay_Nataf, SeMCS, ES, MIS

Tfun = LKF_nD_GaussianLogGamma(2)
smc_SeMIS = SeMCS(ES(SeMIS_Bay_Nataf(Tfun)))
smc_SeMIS.NumSam = 1700

out_sm = smc_SeMIS.RunIte()
out_mis = MIS(out_sm)
print(out_mis.Z)                     # log-evidence (MIS estimator)
u_pos = out_mis.PosSim()[0]          # posterior samples in standard normal space
x_pos = Tfun.Pdis.U2X(u_pos)         # posterior samples in the parameter space
```

## Built-in examples

| File | Description | Reference log-evidence |
|---|---|---|
| `LKF_2D_Eggbox.py` | 2D eggbox likelihood, uniform prior on [0, 10π]² | 235.856 |
| `LKF_nD_GaussianLogGamma.py` | Gaussian + LogGamma, n-dimensional | −8.19 (n = 2), −20.47 (n = 5), −40.94 (n = 10), −81.887 (n = 20) |
| `LKF_nD_GaussianShells.py` | two Gaussian shells, n-dimensional | −1.75 (n = 2) to −255.62 (n = 100); see the class docstring |

Each likelihood class provides the prior (`Pdis`), the log-likelihood and the reference value `logz_ref` used by the driver. `LKF_2D_Eggbox` and `LKF_nD_GaussianLogGamma` also provide the reference marginals, `KStest` (Kolmogorov–Smirnov distance of posterior samples to the reference marginal) and `PltFun`. To solve your own problem, see [Custom likelihood](#using-a-custom-likelihood).

## Using a custom likelihood

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

The raw sampling history `g` contains `Nite` sampled states followed by one final candidate state. `Nsze` records the chain and step counts at each sampled level.

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

Signed-log arithmetic retains an inherited limitation: an all-negative signed-log slice, including a negative first cumulative prefix, is not handled correctly by `logsum`. Preserving complex `logcumsum` outputs retains mixed-sign prefixes but does not change that reduction algorithm.

The proposal-parameter search uses `fminbnd` on 100 sub-intervals. For a flat objective, floating-point arithmetic can select different equally good minimizers and change a stopping decision near the convergence threshold.

## Differences from the MATLAB version

- `RunIte` and `SuS_Bay_Nataf.BayInf` return dictionaries instead of structs, and `MIS.PosSim` returns a tuple `(u_pos, y_pos, u_all, y_all)`.
- Arrays keep the MATLAB layout (`Ndim × Nc × Ns`, flattened in column-major order); Python indices start at 0.
- `fminbnd` is `scipy.optimize.minimize_scalar(method="bounded")` with MATLAB's default tolerance (`TolX = 1e-4`); it implements the same golden-section and parabolic-interpolation search.
- Random numbers are drawn through `semis_bayes._utils` (`randn`, `rand`, `randperm`, `randi`, `randsample`), backed by NumPy; `np.random.seed` fixes a run.
- `MIS.CCDF` and `MIS.KLDEst` are not ported: they call `IndPriWgt`, `IndPosWgt` and `EpyWgt`, which are not part of the MATLAB repository, and the driver does not use them. `LKF_2D_5Gaussian`, mentioned in a comment of `drive_SuS_SeMIS.m`, is not in the MATLAB repository either.
- The reference marginals of `LKF_2D_Eggbox` (used only by `KStest`) are computed on first use rather than in the constructor.
- Eggbox reference grids follow MATLAB's `0:step:10*pi` endpoints; no point beyond `10*pi` is added.
- `ResRat` intentionally keeps the Python `[Nc, Ns]` chain layout, whereas MATLAB flattens it to `[Nc*Ns, 1]`. Use `ratio.reshape(-1, 1, order="F")` for the MATLAB layout; flattened values and the main estimators agree.
- Normal/Lognormal/GEV `U2X` shortcuts are used only when their parameter attributes exist; generic frozen SciPy wrappers fall back to `icdf(norm.cdf(...))`.
- The numerical Nataf correlation search supports negative non-Normal/Lognormal correlations and skips independent pairs. The MATLAB source's numerical search is positive-only; this is an intentional correctness fix, not bit-for-bit behavior preservation.

## Citation

If you use this toolbox in your research, please cite:

> Li, B., He, X., & Liao, Z. (2026). Sequential multiple importance sampling for multi-modal Bayesian inference. *Mechanical Systems and Signal Processing, 245*, Article 113788. [https://doi.org/10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788)

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

