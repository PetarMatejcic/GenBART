import numpy as np
import pytest

try:
    import genbart as gb
    RegBart = gb.RegBart
except Exception:  # pragma: no cover
    from genbart.reg_bart import RegBart


def _make_regression_data(n=80, p=3, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 1] ** 2 + 0.25 * rng.normal(size=n)
    return X, y


def _friedman(X):
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


def _fit_small_model(X, y, random_state=123):
    return RegBart(
        m=8,
        n_burn=30,
        n_samples=40,
        random_state=random_state,
    ).fit(X, y)


def test_predict_before_fit_raises():
    """Raise a clear error when predict is called before the model has been fitted."""
    model = RegBart(m=5, n_burn=5, n_samples=5, random_state=0)

    with pytest.raises(RuntimeError, match="Model not fitted"):
        model.predict(np.array([[0.0, 0.0]]))


def test_fit_rejects_three_dimensional_X():
    """Reject feature arrays with more than two dimensions."""
    X = np.zeros((5, 2, 2), dtype=float)
    y = np.zeros(5, dtype=float)

    model = RegBart(m=5, n_burn=5, n_samples=5, random_state=0)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_accepts_one_dimensional_X_and_predicts():
    """Accept a one-dimensional predictor array by reshaping it internally and return finite predictions."""
    rng = np.random.default_rng(1)
    X = np.linspace(0.0, 1.0, 60)
    y = np.sin(2.0 * np.pi * X) + 0.1 * rng.normal(size=X.shape[0])

    model = RegBart(m=6, n_burn=20, n_samples=30, random_state=1).fit(X, y)
    out = model.predict(np.array([0.25, 0.75]))

    assert out["prediction"].shape == (2,)
    assert out["conf_int_low"].shape == (2,)
    assert out["conf_int_high"].shape == (2,)
    assert np.all(np.isfinite(out["prediction"]))
    assert np.all(out["conf_int_low"] <= out["prediction"])
    assert np.all(out["prediction"] <= out["conf_int_high"])


def test_predict_matrix_output_shapes_and_interval_ordering():
    """Return correctly shaped matrix predictions with finite values and ordered intervals."""
    X, y = _make_regression_data()
    model = _fit_small_model(X, y, random_state=10)

    out = model.predict(X[:7])

    assert set(out.keys()) == {"prediction", "conf_int_low", "conf_int_high"}
    assert out["prediction"].shape == (7,)
    assert out["conf_int_low"].shape == (7,)
    assert out["conf_int_high"].shape == (7,)
    assert np.all(np.isfinite(out["prediction"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))
    assert np.all(out["conf_int_low"] <= out["prediction"])
    assert np.all(out["prediction"] <= out["conf_int_high"])


def test_predict_single_row_multivariate_returns_scalars():
    """Return scalar outputs when predicting a single row for a multivariate model."""
    X, y = _make_regression_data()
    model = _fit_small_model(X, y, random_state=11)

    out = model.predict(X[0])

    assert np.isscalar(out["prediction"])
    assert np.isscalar(out["conf_int_low"])
    assert np.isscalar(out["conf_int_high"])
    assert np.isfinite(out["prediction"])
    assert out["conf_int_low"] <= out["prediction"] <= out["conf_int_high"]


def test_predict_without_confidence_intervals_omits_interval_keys():
    """Omit confidence interval keys when conf_int is disabled."""
    X, y = _make_regression_data()
    model = _fit_small_model(X, y, random_state=12)

    out = model.predict(X[:5], conf_int=False)

    assert set(out.keys()) == {"prediction"}
    assert out["prediction"].shape == (5,)


def test_predict_invalid_central_measure_raises():
    """Raise ValueError for unsupported central tendency choices."""
    X, y = _make_regression_data()
    model = _fit_small_model(X, y, random_state=13)

    with pytest.raises(ValueError):
        model.predict(X[:3], central_measure="mode")


def test_variable_importance_has_correct_shape_and_finite_values():
    """Return one finite nonnegative importance value per predictor after fitting."""
    X, y = _make_regression_data(n=100, p=4, seed=14)
    model = _fit_small_model(X, y, random_state=14)

    vi = model.variable_importance()

    assert vi.shape == (4,)
    assert np.all(np.isfinite(vi))
    assert np.all(vi >= 0.0)


def test_marginalize_returns_expected_shapes_and_finite_values():
    """Return finite marginal predictions and intervals for every grid point."""
    X, y = _make_regression_data(n=90, p=3, seed=15)
    model = _fit_small_model(X, y, random_state=15)

    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 9)
    out = model.marginalize(variable=0, grid=grid, sampling_size=40, level=0.9)

    assert set(out.keys()) == {"prediction", "conf_int_low", "conf_int_high"}
    assert out["prediction"].shape == grid.shape
    assert out["conf_int_low"].shape == grid.shape
    assert out["conf_int_high"].shape == grid.shape
    assert np.all(np.isfinite(out["prediction"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))
    assert np.all(out["conf_int_low"] <= out["prediction"])
    assert np.all(out["prediction"] <= out["conf_int_high"])


def test_fit_with_constant_y_stays_finite():
    """Handle constant responses without producing NaNs or crashing."""
    rng = np.random.default_rng(16)
    X = rng.normal(size=(50, 3))
    y = np.full(50, 7.5, dtype=float)

    model = RegBart(m=6, n_burn=20, n_samples=30, random_state=16).fit(X, y)
    out = model.predict(X[:4])

    assert np.all(np.isfinite(out["prediction"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))


def test_fit_and_predict_are_reproducible_with_fixed_seed():
    """Produce identical fitted predictions and variable importance when the same seed is reused."""
    X, y = _make_regression_data(n=90, p=4, seed=17)

    model1 = RegBart(m=8, n_burn=25, n_samples=35, random_state=999).fit(X, y)
    model2 = RegBart(m=8, n_burn=25, n_samples=35, random_state=999).fit(X, y)

    pred1 = model1.predict(X[:10])
    pred2 = model2.predict(X[:10])

    np.testing.assert_allclose(pred1["prediction"], pred2["prediction"])
    np.testing.assert_allclose(pred1["conf_int_low"], pred2["conf_int_low"])
    np.testing.assert_allclose(pred1["conf_int_high"], pred2["conf_int_high"])
    np.testing.assert_allclose(model1.variable_importance(), model2.variable_importance())


def test_regbart_fit_continuous_random_friedman_does_not_crash():
    """Fit RegBart on continuous random Friedman data without triggering the split-value regression bug."""
    rng = np.random.default_rng(12345)

    n_obs = 500
    X = rng.random((n_obs, 50))
    true_y = _friedman(X)
    y = true_y + 0.5 * rng.normal(size=n_obs)

    model = RegBart(
        m=10,
        n_burn=200,
        n_samples=500,
        random_state=12345,
    ).fit(X, y)

    vi = model.variable_importance()
    preds = model.predict(X[:8], conf_int=False)["prediction"]

    assert vi.shape == (50,)
    assert np.all(np.isfinite(vi))
    assert preds.shape == (8,)
    assert np.all(np.isfinite(preds))
