"""Driver: SeMIS and SuS on a benchmark problem.

Python port of drive_SuS_SeMIS.m. Run from the repository root:

    python drive_SuS_SeMIS.py              # 100 runs, Gaussian-LogGamma, Ndim = 2 (as in MATLAB)
    python drive_SuS_SeMIS.py --nrun 20    # fewer runs
"""
import argparse

import numpy as np

from semis_bayes import (MIS, ES, LKF_2D_Eggbox, LKF_nD_GaussianLogGamma,  # noqa: F401
                         LKF_nD_GaussianShells, SeMCS, SeMIS_Bay_Nataf, SuS_Bay_Nataf)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nrun", type=int, default=100, help="number of independent runs (default 100)")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--quiet", action="store_true", help="do not print the progress of each iteration")
args = parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)

# %% Problem definition and target density
Ndim = 2                                      # parameter dimension
# Tfun = LKF_2D_Eggbox()                      # Eggbox benchmark
# Tfun = LKF_nD_GaussianShells(Ndim)          # Gaussian shells benchmark
Tfun = LKF_nD_GaussianLogGamma(Ndim)          # Gaussian-LogGamma benchmark
logz_ref = Tfun.logz_ref                      # reference log-evidence

# Bayesian inference objects in Nataf space
IntBay_SuS = SuS_Bay_Nataf(Tfun)              # SuS-based inference
IntBay_SeMIS = SeMIS_Bay_Nataf(Tfun)          # SeMIS-based inference

# Sequential Monte Carlo samplers
smc_SuS = SeMCS(ES(IntBay_SuS))
smc_SeMIS = SeMCS(ES(IntBay_SeMIS))

# Sampling budget per level
smc_SeMIS.NumSam = 1700
smc_SuS.NumSam = 1000
smc_SeMIS.Verbose = smc_SuS.Verbose = not args.quiet

# %% Monte Carlo experiment settings
Nrun = args.nrun                              # number of independent runs

# Log-evidence estimates
logz_SeMIS_MIS = np.zeros(Nrun)               # SeMIS + MIS
logz_SeMIS_SIS = np.zeros(Nrun)               # SeMIS + SIS
logz_SuS = np.zeros(Nrun)                     # SuS

# Computational cost and posterior size
Ncal_SeMIS = np.zeros(Nrun)                   # likelihood evaluations
Ncal_SuS = np.zeros(Nrun)
Npos_SeMIS = np.zeros(Nrun)                   # posterior samples
Npos_SuS = np.zeros(Nrun)

# %% Main Monte Carlo loop
for irun in range(Nrun):

    # ----- SeMIS -----
    out_sm = smc_SeMIS.RunIte()               # SMC iterations
    Ncal_SeMIS[irun] = np.sum(out_sm["Ncal"])

    out_mis = MIS(out_sm)                     # MIS post-processing
    logz_SeMIS_MIS[irun] = out_mis.Z          # MIS evidence

    u_pos = out_mis.PosSim()[0]               # posterior samples (U-space)
    Npos_SeMIS[irun] = u_pos.shape[1]
    logz_SeMIS_SIS[irun] = out_mis.C[-1] + out_sm["g"][-1][1]   # SIS evidence

    x_pos_SeMIS = Tfun.Pdis.U2X(u_pos)        # map to physical space

    # ----- SuS -----
    out_sm = smc_SuS.RunIte()
    Ncal_SuS[irun] = np.sum(out_sm["Ncal"])

    out_sus = IntBay_SuS.BayInf(out_sm)
    logz_SuS[irun] = out_sus["z"]

    u_pos = out_sus["x_pos"]
    Npos_SuS[irun] = u_pos.shape[1]
    x_pos_SuS = Tfun.Pdis.U2X(u_pos)

# %% Performance and efficiency statistics
ddof = 1 if Nrun > 1 else 0
# relative bias and coefficient of variation of evidence (%)
logz_SeMIS_MIS_bias = np.mean(logz_SeMIS_MIS / logz_ref - 1) * 100
logz_SeMIS_MIS_cv = np.std(logz_SeMIS_MIS / logz_ref - 1, ddof=ddof) * 100

logz_SeMIS_SIS_bias = np.mean(logz_SeMIS_SIS / logz_ref - 1) * 100
logz_SeMIS_SIS_cv = np.std(logz_SeMIS_SIS / logz_ref - 1, ddof=ddof) * 100

logz_SuS_bias = np.mean(logz_SuS / logz_ref - 1) * 100
logz_SuS_cv = np.std(logz_SuS / logz_ref - 1, ddof=ddof) * 100

# posterior efficiency and average cost
Npos_Ncal_SeMIS_mu = np.mean(Npos_SeMIS / Ncal_SeMIS) * 100
Npos_Ncal_SuS_mu = np.mean(Npos_SuS / Ncal_SuS) * 100

Ncal_SeMIS_mu = np.mean(Ncal_SeMIS)
Ncal_SuS_mu = np.mean(Ncal_SuS)

print("================================================")
print(f"Results over {Nrun} runs (reference log-evidence {logz_ref:.4f})")
print(f"{'':10s} {'mean logz':>10s} {'bias (%)':>9s} {'CoV (%)':>8s} {'Ncal':>8s} {'Npos/Ncal (%)':>14s}")
print(f"{'SeMIS-MIS':10s} {np.mean(logz_SeMIS_MIS):10.4f} {logz_SeMIS_MIS_bias:9.3f} {logz_SeMIS_MIS_cv:8.3f} "
      f"{Ncal_SeMIS_mu:8.0f} {Npos_Ncal_SeMIS_mu:14.2f}")
print(f"{'SeMIS-SIS':10s} {np.mean(logz_SeMIS_SIS):10.4f} {logz_SeMIS_SIS_bias:9.3f} {logz_SeMIS_SIS_cv:8.3f}")
print(f"{'SuS':10s} {np.mean(logz_SuS):10.4f} {logz_SuS_bias:9.3f} {logz_SuS_cv:8.3f} "
      f"{Ncal_SuS_mu:8.0f} {Npos_Ncal_SuS_mu:14.2f}")
print("================================================")
