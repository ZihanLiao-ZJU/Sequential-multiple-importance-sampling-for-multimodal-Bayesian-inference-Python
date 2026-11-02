# Shared-core Bayesian validation (2026-11-02)

This release preserves the native SeMIS MIS evidence estimator and the ES sampling policy. For each of three configurations, the previous public implementation and this shared-core implementation were run at the same 100 independent seeds. All 300 pairs have exactly equal samples, packed values, readable scientific thresholds, MIS log evidence, likelihood-call counts, sample counts and sampling RNG state. All scientific terminal conditions were reached. The common Engine did not improve or degrade those histories. Full serialized G hashes differ with the new state representation and are not claimed identical; the 100-run recorder does not include a paired SIS diagnostic.

Original-paper agreement is a separate check. The local uncorrected proof reports rounded relative SD(log evidence) and mean likelihood calls from 500 repeats. It does not provide an evidence CoV from which CoV(Z)² × Ncal can be certified. The source comparison therefore uses E_log = mean(Ncal) × [sample SD(log Zhat) / |paper log Z reference|]². Lower is better. Source CoV(Z) and evidence-scale efficiency remain unavailable; no lognormal conversion is used.

| Case | Paper log Z reference (rounded) | Mean log Zhat (100) | SD(log Zhat) | Mean Ncal | E_log | Paper E_log | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| eggbox (2D) | 235.86 | 235.865456 | 0.1291937 | 38446.87 | 0.01153547 | 0.011196 | 1.03032 |
| loggamma (2D) | -8.19 | -8.18425959 | 0.09192406 | 19983.15 | 2.517413 | 2.459367 | 1.02360 |
| loggamma (5D) | -20.47 | -20.461664 | 0.1850445 | 85320.30 | 6.972185 | 7.337456 | 0.95022 |

All three source efficiency ratios are within the predeclared 10% point allowance. Eggbox and 2D LogGamma are 3.03% and 2.36% higher; 5D LogGamma is 4.98% lower. These paper-versus-current differences already occur in the previous public implementation and cannot be attributed to the shared Engine.

Each tested configuration uses N=1700 per level. The exact paper-generation per-level N is not printed; total calls differ, so this is an efficiency comparison at the displayed actual costs, rather than a claim of exact paper-budget reproduction. GaussianShells and the optional SuS/other-kernel paths are outside this 100-run source table; separate smaller regression checks cover their interfaces.

The seed cohort is 2026110200–2026110299. Numerical environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, one numerical-library thread. Native model completion and the engine loop flag are retained separately; an empty seed pool can end the driver after the model terminal condition is reached.

Source: Li, He and Liao, *Sequential multiple importance sampling for multi-modal Bayesian inference*, local uncorrected proof Table 2, DOI [10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788). The final publisher table-generation data have not been authenticated, so the source and rounding limitations remain explicit.
