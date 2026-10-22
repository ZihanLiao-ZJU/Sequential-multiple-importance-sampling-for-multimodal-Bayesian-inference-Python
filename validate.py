"""Validate Python against recorded Octave runs of the MATLAB source.

Part 1 - replay (exact): validation/reference/ holds random number streams and
results recorded by running the MATLAB source under Octave once per benchmark (one
SeMIS run followed by one SuS run, as in one iteration of drive_SuS_SeMIS.m;
validation/matlab/record_reference.m). The Python code is driven by the same
random draws/selected indices and compares the recorded proposal parameters, likelihood
call counts, log-evidence (MIS, SIS and SuS) and posterior samples to round-off.
PASS* is conditional on prescribed proposal parameters, not a direct replay.
Legacy fixtures do not record probabilities; Python weight invariants are separate.

Part 2 - statistics: repeated Python runs on the bundled benchmarks; the mean
log-evidence of SeMIS (MIS estimator) and SuS is compared with the reference.

    python validate.py              # both parts
    python validate.py --replay     # part 1 only
    python validate.py --nrun 50    # part 2 with 50 runs per benchmark
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "validation"))

import semis_bayes as sb  # noqa: E402
from semis_bayes import _utils  # noqa: E402
from replay import ReplayRNG  # noqa: E402
from semis_bayes.log_operations import logmean  # noqa: E402
from semis_bayes.support_functions.ResRat import ResRat  # noqa: E402
from weights import semis_probabilities, sus_probabilities  # noqa: E402

EXPECTED_CASES = ("eggbox_2_1", "loggamma_2_1", "loggamma_2_2", "loggamma_5_1",
                  "shells_2_1", "shells_5_1")


def reference_files(cases=None):
    """Require both files for every explicitly selected case (all six by default)."""
    cases = EXPECTED_CASES if cases is None else tuple(cases)
    if not cases or any(case not in EXPECTED_CASES for case in cases):
        raise ValueError("select at least one known replay case")
    folder = Path(HERE) / "validation" / "reference"
    missing = [str(folder / f"{prefix}_{case}.mat") for case in cases
               for prefix in ("rec", "res") if not (folder / f"{prefix}_{case}.mat").is_file()]
    if missing:
        raise FileNotFoundError("missing required replay fixtures:\n" + "\n".join(missing))
    return [str(folder / f"rec_{case}.mat") for case in cases]


def make_lkf(case, ndim):
    return {"shells": lambda: sb.LKF_nD_GaussianShells(ndim),
            "loggamma": lambda: sb.LKF_nD_GaussianLogGamma(ndim),
            "eggbox": lambda: sb.LKF_2D_Eggbox()}[case]()


def run_once(Tfun, G_semis=None, check_weights=None):
    """One SeMIS run and one SuS run, as in one iteration of drive_SuS_SeMIS.

    G_semis fixes the SeMIS proposal parameters [r_i, r_max] of every level
    (SeMCS.G) instead of searching them with fminbnd.
    """
    IntBay_SuS = sb.SuS_Bay_Nataf(Tfun)
    smc_SuS = sb.SeMCS(sb.ES(IntBay_SuS))
    smc_SeMIS = sb.SeMCS(sb.ES(sb.SeMIS_Bay_Nataf(Tfun)))
    smc_SeMIS.NumSam = 1700
    smc_SuS.NumSam = 1000
    smc_SeMIS.Verbose = smc_SuS.Verbose = False
    if G_semis is not None:
        smc_SeMIS.G = [np.array(gi) for gi in G_semis]
    out_sm = smc_SeMIS.RunIte()
    out_mis = sb.MIS(out_sm)
    r = dict(z_mis=out_mis.Z, u_pos_semis=out_mis.PosSim()[0],
             z_sis=out_mis.C[-1] + out_sm["g"][-1][1], G_semis=np.vstack(out_sm["g"]),
             Ncal_semis=np.asarray(out_sm["Ncal"]), Nite_semis=out_sm["Nite"], out_semis=out_sm)
    if check_weights is not None:
        check_weights(semis_probabilities(out_sm, out_mis))
    out_sm = smc_SuS.RunIte()
    out_sus = IntBay_SuS.BayInf(out_sm)
    if check_weights is not None:
        check_weights(sus_probabilities(out_sm))
    r.update(z_sus=out_sus["z"], u_pos_sus=out_sus["x_pos"], G_sus=np.vstack(out_sm["g"]),
             Ncal_sus=np.asarray(out_sm["Ncal"]), Nite_sus=out_sm["Nite"])
    return r


def maxdiff(a, b):
    # Expected benchmark outputs are real. Reject imaginary corruption before
    # float conversion can silently discard it (including imaginary NaN/Inf).
    for value in (a, b):
        value = np.asarray(value)
        if np.iscomplexobj(value) and np.any(np.imag(value) != 0):
            return np.inf
    a, b = np.asarray(np.real(a), float), np.asarray(np.real(b), float)
    if (a.shape != b.shape or np.any(np.isnan(a)) or np.any(np.isnan(b))
            or not np.array_equal(np.isposinf(a), np.isposinf(b))
            or not np.array_equal(np.isneginf(a), np.isneginf(b))):
        return np.inf
    f = np.isfinite(a)
    return float(np.max(np.abs(a[f] - b[f]), initial=0.0))


def compare(p, m, rng):
    finite_keys = ("z_mis", "z_sis", "z_sus", "u_pos_semis", "u_pos_sus",
                   "Ncal_semis", "Ncal_sus", "Nite_semis", "Nite_sus")
    count_keys = ("Ncal_semis", "Ncal_sus", "Nite_semis", "Nite_sus")
    for result in (p, m):
        for key in finite_keys + ("G_semis", "G_sus"):
            value = np.asarray(result[key])
            if value.size == 0 or (np.iscomplexobj(value) and np.any(np.imag(value) != 0)):
                return False, np.inf
            if key in finite_keys and not np.all(np.isfinite(value)):
                return False, np.inf
            if key in count_keys and (np.any(value < 0) or np.any(value != np.floor(np.real(value)))):
                return False, np.inf
    d = max(maxdiff(p[k], m[k].reshape(np.shape(p[k])) if np.size(p[k]) == m[k].size else m[k])
            for k in ("z_mis", "z_sis", "z_sus", "G_semis", "G_sus", "u_pos_semis", "u_pos_sus"))
    same = all(np.array_equal(np.ravel(p[k]), m[k].ravel())
               for k in count_keys)
    return same and d < 1e-8 and rng.done, d


def replay_case(fn, Tfun, G_semis=None):
    rng = ReplayRNG(fn)
    _utils.set_rng(rng)
    try:
        return run_once(Tfun, G_semis, rng.check_last_probabilities), rng
    except RuntimeError:            # the random stream no longer matches: the runs diverged
        return None, rng
    finally:
        _utils.set_rng(None)


def search_tie(Tfun, out, G_m, G_p):
    """At the first level where the r_i found by Python and MATLAB differ, evaluate
    the fminbnd objective |logmean(beta_i) - log(p)| at both values (same samples)."""
    k = next(i for i in range(1, min(len(G_m), len(G_p)))
             if not np.allclose(G_m[i], G_p[i], rtol=1e-10, atol=1e-12))
    obj = sb.SeMIS_Bay_Nataf(Tfun)
    obj.G = np.array(G_m[k - 1])
    y, p = out["y"][k - 1], sb.SeMCS(sb.ES(obj)).ConPrb
    f = [abs(logmean(ResRat(obj, y, np.array([r, G_m[k][1]]))[0], None) - np.log(p))
         for r in (G_p[k][0], G_m[k][0])]
    return k, f


def replay(cases=None):
    files = reference_files(cases)
    print(f"Part 1: {len(files)}/{len(EXPECTED_CASES)} selected Octave reference replays")
    print("Legacy fixtures replay selected indices, not recorded weights; separate Python weight invariants are checked.")
    print(f"{'case':14s} {'logZ MIS (Py)':>14s} {'logZ MIS (ref)':>17s} {'logZ SuS (Py)':>14s} "
          f"{'logZ SuS (ref)':>17s} {'max|d|':>8s}  result")
    ok_all = True
    notes = []
    for fn in files:
        tag = os.path.basename(fn)[4:-4]
        case, ndim, _ = tag.split("_")
        T = make_lkf(case, int(ndim))
        m = loadmat(os.path.join(HERE, "validation", "reference", f"res_{tag}.mat"))
        p, rng = replay_case(fn, T)
        ok, d = compare(p, m, rng) if p is not None else (False, np.inf)
        label = "PASS"
        if not ok:
            # The SeMIS proposal parameter r_i is found by fminbnd on 100 sub-intervals.
            # Where the objective is flat, round-off decides which minimiser is kept.
            # Replay again with the MATLAB values of [r_i, r_max] fixed: everything
            # else must then agree exactly, and the two r_i must be equally good.
            pc, rngc = replay_case(fn, T, m["G_semis"])
            okc, d = compare(pc, m, rngc) if pc is not None else (False, np.inf)
            G_p = list(p["G_semis"]) if p is not None else None
            if okc and G_p is None:
                # diverged before RunIte returned: recover Python's r_i from a free run
                # stopped where the parameters are compared
                G_p = free_G(fn, T, len(m["G_semis"]))
            k, f = search_tie(T, pc["out_semis"], m["G_semis"], G_p) if okc else (None, [np.inf, 0])
            ok = okc and abs(f[0] - f[1]) < 1e-10
            label = "PASS*" if ok else "FAIL"
            if ok:
                notes.append(f"* {case} n={ndim}: at level {k} Python takes r_i = {G_p[k][0]:.6e} and MATLAB "
                             f"r_i = {m['G_semis'][k][0]:.6e}; the fminbnd objective is {f[0]:.15f} at both "
                             f"(difference {abs(f[0] - f[1]):.1e}), so the choice is decided by round-off. With "
                             f"MATLAB's r_i the replay is identical.")
            p = pc if okc else p
        ok_all &= ok
        zm = p["z_mis"] if p is not None else np.nan
        zs = p["z_sus"] if p is not None else np.nan
        print(f"{case + ' n=' + ndim:14s} {zm:14.6f} {m['z_mis'].item():17.6f} {zs:14.6f} "
              f"{m['z_sus'].item():17.6f} {d:8.1e}  {label}")
    for n in notes:
        print(n)
    return ok_all


def free_G(fn, Tfun, nlev):
    """Proposal parameters of a free Python SeMIS replay (stopped after nlev - 1 levels)."""
    rng = ReplayRNG(fn)
    _utils.set_rng(rng)
    try:
        smc = sb.SeMCS(sb.ES(sb.SeMIS_Bay_Nataf(Tfun)))
        smc.NumSam = 1700
        smc.Verbose = False
        smc.MaxIte = nlev
        return list(smc.RunIte()["g"])
    finally:
        _utils.set_rng(None)


def statistics(nrun, seed):
    print(f"\nPart 2: {nrun} independent Python runs per benchmark (SeMIS 1700, SuS 1000 samples per level)")
    print(f"{'case':14s} {'method':6s} {'mean logZ':>10s} {'std':>7s} {'reference':>10s} {'|bias|/s.e.':>11s} "
          f"{'Ncal':>7s}  result")
    np.random.seed(seed)
    ok_all = True
    for case, ndim in [("loggamma", 2), ("loggamma", 5), ("shells", 2), ("shells", 5), ("eggbox", 2)]:
        T = make_lkf(case, ndim)
        res = [run_once(T) for _ in range(nrun)]
        for meth in ("mis", "sus"):
            z = np.array([r[f"z_{meth}"] for r in res])
            ncal = np.mean([np.sum(r["Ncal_semis" if meth == "mis" else "Ncal_sus"]) for r in res])
            se = np.std(z, ddof=1) / np.sqrt(nrun)
            t = abs(np.mean(z) - T.logz_ref) / se
            # pass: within 4 standard errors, or relative bias of logZ below 1 %
            ok = t < 4 or abs(np.mean(z) - T.logz_ref) < 0.01 * abs(T.logz_ref)
            ok_all &= ok
            print(f"{case + ' n=' + str(ndim):14s} {'SeMIS' if meth == 'mis' else 'SuS':6s} {np.mean(z):10.3f} "
                  f"{np.std(z, ddof=1):7.3f} {T.logz_ref:10.3f} {t:11.2f} {ncal:7.0f}  {'PASS' if ok else 'CHECK'}")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="store_true", help="only run the replay comparison")
    ap.add_argument("--nrun", type=int, default=50, help="runs per benchmark in part 2 (default 50)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--case", choices=EXPECTED_CASES, action="append",
                    help="explicit replay subset; repeat to select multiple cases (default: all six)")
    a = ap.parse_args()
    try:
        ok = replay(a.case)
    except (FileNotFoundError, ValueError) as exc:
        ap.exit(1, f"Replay validation failed: {exc}\n")
    if not a.replay:
        ok &= statistics(a.nrun, a.seed)
    print("\nSelected checks passed (PASS* is conditional; no native MATLAB or recorded-weight parity claim)."
          if ok else "\nSome checks did not pass.")
    sys.exit(0 if ok else 1)
