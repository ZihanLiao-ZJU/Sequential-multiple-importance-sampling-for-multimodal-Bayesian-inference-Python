"""Deterministic regressions; these do not require a MATLAB runtime."""
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import stats
from scipy.io import savemat

import validate
from semis_bayes import Distribution, Nataf, makedist
from semis_bayes.likelihood_functions.LKF_2D_Eggbox import _matlab_grid
from semis_bayes.log_operations import logcumsum, logsum
from semis_bayes.support_functions.ResRat import ResRat
from replay import ReplayRNG
from weights import semis_probabilities
from test_core_alignment import Model
import semis_bayes as sb


class RegressionTests(unittest.TestCase):
    def test_frozen_distribution_fallbacks(self):
        u = np.array([[-2., 0., 2.]])
        for name, frozen in [("Normal", stats.norm(loc=2, scale=3)),
                             ("Lognormal", stats.lognorm(s=.5, scale=2)),
                             ("Generalized Extreme Value", stats.genextreme(c=-.2, loc=1, scale=2))]:
            with self.subTest(name=name):
                d = Distribution(name, frozen)
                x = Nataf([d], np.eye(1)).U2X(u)
                np.testing.assert_allclose(x, frozen.ppf(stats.norm.cdf(u)))

    def test_parameterized_fast_paths_keep_extreme_tails(self):
        u = np.array([[-10., 10.]])
        for d in [makedist("Normal", mu=2, sigma=3), makedist("Lognormal", mu=.2, sigma=.3)]:
            self.assertTrue(np.all(np.isfinite(Nataf([d], np.eye(1)).U2X(u))))

    def test_negative_uniform_correlation(self):
        obj = Nataf([makedist("Uniform", lower=-1, upper=1)] * 2,
                    np.array([[1., -.5], [-.5, 1.]]))
        # Gaussian-copula uniform correlation: rho_X = 6/pi * asin(rho_U/2).
        self.assertAlmostEqual(obj.Rho_U[0, 1], 2 * np.sin(-np.pi / 12), delta=1e-4)

    def test_zero_pair_in_correlated_matrix_skips_quadrature(self):
        # Pairs (0,2) and (1,2) are independent despite another correlated pair.
        R = np.array([[1., .2, 0.], [.2, 1., 0.], [0., 0., 1.]])
        ds = [makedist("Normal"), makedist("Normal"), makedist("Uniform")]
        with patch("semis_bayes.Nataf.GauIntPot", side_effect=AssertionError("unexpected quadrature")):
            np.testing.assert_array_equal(Nataf(ds, R).Rho_U, R)

    def test_complex_cumsum_both_axes_preserves_sign(self):
        a = np.array([[0, np.log(2) + 1j * np.pi, np.log(3)]])
        for values, axis in [(a, 1), (a.T, 0)]:
            with np.errstate(all="raise"):
                actual = logcumsum(values, axis)
            np.testing.assert_allclose(np.exp(actual), np.cumsum(np.exp(values), axis=axis), atol=1e-14)
            self.assertTrue(np.iscomplexobj(actual))

    def test_real_cumsum_stays_real(self):
        a = np.log([[1., 2., 3.]])
        out = logcumsum(a, 1)
        self.assertFalse(np.iscomplexobj(out))
        np.testing.assert_allclose(np.exp(out), [[1., 3., 6.]])

    @unittest.expectedFailure
    def test_known_inherited_all_negative_logsum(self):
        # Both original implementations mishandle reductions containing only
        # negative signed-log terms. Track without hiding it behind the fix for
        # logcumsum's independent real-dtype truncation bug.
        a = np.log([2., 3.]) + 1j * np.pi
        np.testing.assert_allclose(np.exp(logsum(a)), -5., atol=1e-14)

    def test_eggbox_matlab_endpoints(self):
        for step, count, last in [(.001, 31416, 31.415), (.0005, 62832, 31.4155)]:
            grid = _matlab_grid(10 * np.pi, step)
            self.assertEqual(len(grid), count)
            self.assertEqual(grid[0], 0)
            self.assertAlmostEqual(grid[-1], last)
            self.assertLessEqual(grid[-1], 10 * np.pi)

    def test_resrat_preserves_chain_shape_and_flat_values(self):
        obj = sb.SeMIS_Bay_Nataf(Model())
        y = np.zeros((4, 2, 3)); y[2] = np.arange(-6, 0).reshape(2, 3)
        ratio, c = ResRat(obj, y, np.array([-2., 0.]))
        flat, cf = ResRat(obj, y.reshape(4, -1, order="F"), np.array([-2., 0.]))
        self.assertEqual(ratio.shape, (2, 3))
        np.testing.assert_array_equal(ratio.reshape(-1, 1, order="F"), flat)
        self.assertEqual(c, cf)

    def test_nonfinite_comparison_fails_closed(self):
        for a, b in [([np.inf], [-np.inf]), ([np.nan], [np.inf]),
                     ([np.nan], [np.nan]), ([0], [np.inf])]:
            self.assertEqual(validate.maxdiff(a, b), np.inf)
        self.assertEqual(validate.maxdiff([np.inf, -np.inf], [np.inf, -np.inf]), 0.)

    def test_complex_output_corruption_cannot_pass(self):
        for bad in ([1 + 2j], [complex(1, np.nan)], [complex(1, np.inf)]):
            self.assertEqual(validate.maxdiff(bad, [1.]), np.inf)
            self.assertEqual(validate.maxdiff([1.], bad), np.inf)

    def test_compare_rejects_nonfinite_or_empty_results(self):
        keys = ("z_mis", "z_sis", "z_sus", "u_pos_semis", "u_pos_sus",
                "Ncal_semis", "Ncal_sus", "Nite_semis", "Nite_sus")
        base = {key: np.array([1.]) for key in keys}
        base.update(G_semis=np.array([np.inf, -np.inf]), G_sus=np.array([np.inf, -np.inf]))
        done = SimpleNamespace(done=True)
        self.assertEqual(validate.compare(base, base, done), (True, 0.))
        for key in keys:
            for value in (np.array([np.inf]), np.array([-np.inf]), np.array([np.nan]), np.array([])):
                bad = dict(base, **{key: value})
                self.assertFalse(validate.compare(bad, bad, done)[0], (key, value))
        for key in ("G_semis", "G_sus"):
            self.assertFalse(validate.compare(dict(base, **{key: np.array([])}), base, done)[0])

    def test_missing_all_fixtures_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d, patch.object(validate, "HERE", d):
            with self.assertRaisesRegex(FileNotFoundError, "missing required replay fixtures"):
                validate.replay()

    def test_default_manifest_requires_all_six_cases(self):
        with tempfile.TemporaryDirectory() as d, patch.object(validate, "HERE", d):
            folder = pathlib.Path(d) / "validation" / "reference"
            folder.mkdir(parents=True)
            for prefix in ("rec", "res"):
                (folder / f"{prefix}_eggbox_2_1.mat").touch()
            self.assertEqual(len(validate.reference_files(["eggbox_2_1"])), 1)
            with self.assertRaises(FileNotFoundError):
                validate.reference_files()
            with self.assertRaises(ValueError):
                validate.reference_files([])

    def _rng(self, directory, indices=(1, 2)):
        fname = pathlib.Path(directory) / "stream.mat"
        savemat(fname, dict(kinds=[5], lens=[len(indices)], vals=indices))
        return ReplayRNG(fname)

    def test_replay_checks_actual_resampling_probabilities(self):
        with tempfile.TemporaryDirectory() as d:
            rng = self._rng(d)
            np.testing.assert_array_equal(rng.randsample(2, 2, [.25, .75]), [0, 1])
            rng.check_last_probabilities([.25, .75])
            with self.assertRaises(AssertionError):
                rng.check_last_probabilities([.75, .25])

    def test_replay_rejects_invalid_probabilities(self):
        with tempfile.TemporaryDirectory() as d:
            for w in ([0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1]):
                with self.assertRaises(ValueError):
                    self._rng(d).randsample(2, 2, w)
            with self.assertRaisesRegex(RuntimeError, "zero supplied probability"):
                self._rng(d).randsample(2, 2, [1, 0])
            with self.assertRaisesRegex(RuntimeError, "outside"):
                self._rng(d, indices=(1, 3)).randsample(2, 2, [1, 1])
            for indices in [(1, 1.5), (1, np.nan), (1, np.inf)]:
                with self.assertRaisesRegex(RuntimeError, "finite integers"):
                    self._rng(d, indices=indices).randsample(2, 2, [1, 1])

    def test_replay_exhaustion_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            rng = self._rng(d)
            rng.randsample(2, 2, [1, 1])
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                rng.randsample(2, 2, [1, 1])

    def test_independent_semis_weight_invariant_detects_corruption(self):
        np.random.seed(9)
        smc = sb.SeMCS(sb.ES(sb.SeMIS_Bay_Nataf(Model())))
        smc.NumSam = 40; smc.Verbose = False
        out = smc.RunIte(); mis = sb.MIS(out)
        p = semis_probabilities(out, mis)
        self.assertAlmostEqual(p.sum(), 1.)
        mis.Wpos[0] = mis.Wpos[0] + .1
        with self.assertRaises(AssertionError):
            semis_probabilities(out, mis)


if __name__ == "__main__":
    unittest.main()
