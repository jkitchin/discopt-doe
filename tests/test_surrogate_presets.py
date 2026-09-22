"""Calibration and behaviour of the ``"gp"`` preset and the BO driver additions.

The preset used to be a maximum-likelihood GP whose noise collapsed to ~0 on
small unreplicated designs, so its 95% intervals covered ~0.79. These tests
pin the properties that matter for honest error bars and for reproducible,
constrained active learning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from discopt.doe.acquisition import expected_improvement, max_variance  # noqa: E402
from discopt.doe.optimize import optimize_round  # noqa: E402
from discopt.doe.surrogate import (  # noqa: E402
    GPSurrogate,
    coerce_surrogate,
    gp_surrogate,
)
from discopt.doe.workbook import InputSpec, Workbook  # noqa: E402

SIGMA = 0.5


def _truth2(U: np.ndarray) -> np.ndarray:
    u1, u2 = U[:, 0], U[:, 1]
    bump = 12 * np.exp(-((u1 - 0.65) ** 2 / 0.08 + (u2 - 0.4) ** 2 / 0.15))
    return 60 + bump + 6 * u2 - 4 * u1 * u2


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = (np.arange(n)[:, None] + rng.uniform(size=(n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


# ──────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────


def test_gp_preset_intervals_are_calibrated_on_unreplicated_noisy_data():
    """20 unreplicated noisy runs: 95% predictive intervals should cover new
    observations close to 95% of the time, and the noise must not collapse."""
    rng = np.random.default_rng(0)
    cover, noise = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(12):
            U = _lhs(20, 2, rng)
            y = _truth2(U) + rng.normal(0, SIGMA, 20)
            Ut = rng.uniform(size=(300, 2))
            yt = _truth2(Ut) + rng.normal(0, SIGMA, 300)
            s = coerce_surrogate("gp").fit(U, y)
            m, sd = s.predict(Ut)
            cover.append(np.mean(np.abs(yt - m) <= 1.96 * sd))
            noise.append(s.noise_sd_)
    assert np.mean(cover) >= 0.88
    assert np.median(noise) > 0.1  # the old preset's median was ~0.0003


def test_noise_from_replicates_is_pure_error():
    rng = np.random.default_rng(1)
    U = np.repeat(_lhs(8, 2, rng), 3, axis=0)  # every point run three times
    y = _truth2(U) + rng.normal(0, SIGMA, len(U))
    s = gp_surrogate().fit(U, y)
    groups = y.reshape(8, 3)
    pure = np.sqrt(np.sum((groups - groups.mean(axis=1, keepdims=True)) ** 2) / (8 * 2))
    assert s.noise_source_ == "replicates"
    assert s.noise_sd_ == pytest.approx(pure, rel=1e-6)


def test_noise_zero_interpolates_and_fixed_noise_is_respected():
    rng = np.random.default_rng(2)
    U = _lhs(15, 2, rng)
    y = _truth2(U)
    s = gp_surrogate(noise=0).fit(U, y)
    assert s.noise_source_ == "none" and s.noise_sd_ == 0.0
    m, _ = s.predict(U)
    assert np.max(np.abs(m - y)) < 1e-3
    f = gp_surrogate(noise=0.3).fit(U, y + rng.normal(0, 0.3, len(y)))
    assert f.noise_source_ == "fixed"
    assert f.noise_sd_ == pytest.approx(0.3, rel=1e-6)


def test_warns_when_fitted_noise_sits_on_its_floor():
    rng = np.random.default_rng(3)
    U = _lhs(15, 2, rng)
    with pytest.warns(UserWarning, match="floor"):
        s = gp_surrogate().fit(U, _truth2(U))  # noiseless data, noise fitted
    assert s.noise_at_floor_


def test_latent_sigma_excludes_noise():
    rng = np.random.default_rng(4)
    U = _lhs(20, 2, rng)
    s = gp_surrogate(noise=0.5).fit(U, _truth2(U) + rng.normal(0, 0.5, 20))
    _, sd = s.predict(U[:3])
    _, lat = s.predict_latent(U[:3])
    assert np.allclose(sd**2 - lat**2, 0.25, rtol=1e-6)


# ──────────────────────────────────────────────────────────────────
# ARD, reproducibility, describe
# ──────────────────────────────────────────────────────────────────


def test_ard_auto_recovers_an_inert_input():
    rng = np.random.default_rng(5)
    U = _lhs(30, 3, rng)
    y = _truth2(U[:, :2]) + rng.normal(0, 0.2, 30)  # third input does nothing
    s = gp_surrogate().fit(U, y)
    assert s.ard_ is True
    ls = s.length_scales_
    assert ls[2] > 10 * max(ls[0], ls[1])


def test_fits_are_reproducible_for_a_seed():
    rng = np.random.default_rng(6)
    U = _lhs(15, 2, rng)
    y = _truth2(U) + rng.normal(0, 0.5, 15)
    a = gp_surrogate(random_state=11).fit(U, y).predict(U)[0]
    b = gp_surrogate(random_state=11).fit(U, y).predict(U)[0]
    assert np.array_equal(a, b)
    d = gp_surrogate().fit(U, y).describe()
    assert set(d) >= {"ard", "length_scales", "noise_sd", "noise_source", "signal_sd"}
    assert isinstance(coerce_surrogate("gp"), GPSurrogate)


# ──────────────────────────────────────────────────────────────────
# Acquisitions
# ──────────────────────────────────────────────────────────────────


def test_ei_decays_with_latent_sigma_but_not_predictive():
    """Replicate the neighbourhood of the optimum heavily: the mean there is
    known, so EI should be small. With the predictive σ (which never drops
    below the noise) it stays large; with the latent σ (the default) it decays."""
    rng = np.random.default_rng(7)
    X = np.concatenate([np.linspace(0, 1, 9), np.repeat(0.5, 30)])[:, None]
    y = -((X[:, 0] - 0.5) ** 2) + rng.normal(0, 0.2, len(X))
    s = gp_surrogate().fit(X, y)
    at_opt = np.array([[0.5]])
    best = float(np.max(s.predict(X)[0]))
    ei_latent = expected_improvement(s, at_opt, y_best=best, direction=1)[0]
    ei_pred = expected_improvement(s, at_opt, y_best=best, direction=1, latent=False)[0]
    assert ei_latent < 0.25 * ei_pred


def test_max_variance_picks_the_most_uncertain_candidate():
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 0.5, size=(10, 1))
    s = gp_surrogate(noise=0.05).fit(X, np.sin(6 * X[:, 0]))
    cands = np.linspace(0, 1, 41)[:, None]
    score = max_variance(s, cands)
    _, lat = s.predict_latent(cands)
    assert int(np.argmax(score)) == int(np.argmax(lat))
    assert cands[int(np.argmax(score)), 0] > 0.9  # far from the data


def test_bootstrap_adapter_can_include_noise():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(9)
    X = rng.uniform(size=(40, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.5, 40)
    plain = coerce_surrogate(LinearRegression()).fit(X, y)
    noisy = coerce_surrogate(LinearRegression(), include_noise=True).fit(X, y)
    assert plain.mode == "bootstrap"
    _, s0 = plain.predict(X[:5])
    _, s1 = noisy.predict(X[:5])
    _, lat = noisy.predict_latent(X[:5])
    assert np.allclose(lat, s0)
    assert np.all(s1 > 0.4) and np.all(s0 < 0.3)


def test_sklearn_gp_with_white_kernel_gets_latent_sigma():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(10)
    X = rng.uniform(size=(20, 1))
    y = np.sin(5 * X[:, 0]) + rng.normal(0, 0.3, 20)
    s = coerce_surrogate(GaussianProcessRegressor(RBF() + WhiteKernel(), normalize_y=True))
    s.fit(X, y)
    _, sd = s.predict(X[:4])
    _, lat = s.predict_latent(X[:4])
    assert np.all(lat < sd)


# ──────────────────────────────────────────────────────────────────
# optimize_round: pools, seeds, feasibility; Workbook.record_responses
# ──────────────────────────────────────────────────────────────────


def _workbook(path, xs, ys, extra_columns=None):
    Workbook.create(
        path,
        template=None,
        template_args={},
        input_specs=[InputSpec("x1", 0.0, 1.0), InputSpec("x2", 0.0, 1.0)],
        criterion="custom",
        measurement_error=0.5,
        seed=0,
        response_name="y",
        extra_columns=extra_columns or (),
    )
    wb = Workbook.open(path)
    ids = wb.append_runs(0, [{"x1": float(a), "x2": float(b)} for a, b in xs])
    wb.record_responses({i: v for i, v in zip(ids, ys) if v is not None})
    wb.save()
    return ids


def test_record_responses_round_trip(tmp_path):
    path = tmp_path / "r.xlsx"
    ids = _workbook(path, [(0.1, 0.2), (0.3, 0.4)], [None, None], extra_columns=["ok"])
    wb = Workbook.open(path)
    wb.record_responses({ids[0]: 1.5}, extra={ids[1]: {"ok": 0}})
    wb.save()
    wb = Workbook.open(path)
    done = wb.completed_runs()
    assert [r["run_id"] for r in done] == [ids[0]] and done[0]["y"] == 1.5
    assert done[0]["measured_at"]
    assert wb.pending_runs()[0]["ok"] == 0
    with pytest.raises(ValueError, match="run_id"):
        wb.record_responses({999: 1.0})
    with pytest.raises(ValueError, match="cannot set"):
        wb.record_responses({ids[0]: 1.0}, extra={ids[0]: {"batch": 3}})


def test_candidate_pool_and_sampler_are_respected(tmp_path):
    rng = np.random.default_rng(12)
    xs = _lhs(10, 2, rng)
    path = tmp_path / "c.xlsx"
    _workbook(path, xs, list(_truth2(xs)))
    pool = [{"x1": 0.2, "x2": 0.8}, {"x1": 0.65, "x2": 0.4}, {"x1": 0.9, "x2": 0.1}]
    res = optimize_round(path, candidates=pool, batch_size=2, seed=0)
    assert all(d in pool for d in res.next_designs)

    def on_diagonal(n, rng):
        t = rng.uniform(size=n)
        return np.column_stack([t, t])

    res = optimize_round(path, candidate_sampler=on_diagonal, n_candidates=64, seed=0)
    d = res.next_designs[0]
    assert d["x1"] == pytest.approx(d["x2"])


def test_optimize_round_is_reproducible_with_seed(tmp_path):
    rng = np.random.default_rng(13)
    xs = _lhs(10, 2, rng)
    ys = list(_truth2(xs) + rng.normal(0, 0.5, 10))
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _workbook(a, xs, ys)
    _workbook(b, xs, ys)
    ra = optimize_round(a, batch_size=3, seed=4)
    rb = optimize_round(b, batch_size=3, seed=4)
    assert ra.next_designs == rb.next_designs


def test_failed_runs_steer_away_from_the_infeasible_region(tmp_path):
    """Runs with x1 > 0.6 fail. Marked as infeasible, they train a classifier
    and the recommendations stay in the region that works."""
    rng = np.random.default_rng(14)
    xs = _lhs(24, 2, rng)
    ok = xs[:, 0] <= 0.6
    ys = [float(v) if good else None for v, good in zip(_truth2(xs), ok)]
    path = tmp_path / "f.xlsx"
    ids = _workbook(path, xs, ys, extra_columns=["feasible"])
    wb = Workbook.open(path)
    wb.record_responses({}, extra={i: {"feasible": int(g)} for i, g in zip(ids, ok)})
    wb.save()
    res = optimize_round(path, batch_size=3, seed=0, feasibility_column="feasible")
    assert res.feasibility is not None and min(res.feasibility) > 0.5
    assert all(d["x1"] <= 0.7 for d in res.next_designs)
    # the same through explicit run_ids
    failed = [i for i, g in zip(ids, ok) if not g]
    res2 = optimize_round(path, batch_size=1, seed=0, infeasible_runs=failed)
    assert res2.next_designs[0]["x1"] <= 0.7
