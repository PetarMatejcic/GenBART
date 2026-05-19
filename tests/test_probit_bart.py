import numpy as np
import pytest
from scipy.stats import norm

try:
    import genbart as gb
    ProbitBart = gb.ProbitBart
except Exception:  # pragma: no cover
    from genbart.probit_bart import ProbitBart


def _make_classification_data(n=120, p=4, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    latent = 1.25 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2] ** 2 - 0.25 * X[:, 3]
    probs = norm.cdf(latent)
    y = rng.binomial(1, probs).astype(int)
    return X, y


def _friedman_latent(X):
    return (
        1.5 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2.0 * (X[:, 2] - 0.5) ** 2
        + 1.0 * X[:, 3]
        + 0.5 * X[:, 4]
        - 1.5
    )


def _fit_small_model(X, y, random_state=123):
    return ProbitBart(
        m=8,
        n_burn=30,
        n_samples=40,
        random_state=random_state,
    ).fit(X, y)


def test_fit_rejects_three_dimensional_X():
    """Reject feature arrays with more than two dimensions."""
    X = np.zeros((5, 2, 2), dtype=float)
    y = np.zeros(5, dtype=int)

    model = ProbitBart(m=5, n_burn=5, n_samples=5, random_state=0)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_accepts_one_dimensional_X_and_predicts_probs():
    """Accept a one-dimensional predictor array by reshaping it internally and return finite probabilities."""
    rng = np.random.default_rng(1)
    X = np.linspace(-2.0, 2.0, 80)
    latent = 1.5 * X
    probs = norm.cdf(latent)
    y = rng.binomial(1, probs).astype(int)

    model = ProbitBart(m=6, n_burn=20, n_samples=30, random_state=1).fit(X, y)
    out = model.predict_probs(np.array([-1.0, 0.0, 1.0]))

    assert out["probs"].shape == (3,)
    assert out["conf_int_low"].shape == (3,)
    assert out["conf_int_high"].shape == (3,)
    assert np.all(np.isfinite(out["probs"]))
    assert np.all(out["conf_int_low"] <= out["probs"])
    assert np.all(out["probs"] <= out["conf_int_high"])
    assert np.all((0.0 <= out["probs"]) & (out["probs"] <= 1.0))


def test_predict_probs_matrix_output_shapes_and_interval_ordering():
    """Return correctly shaped matrix probabilities with finite values and ordered intervals."""
    X, y = _make_classification_data()
    model = _fit_small_model(X, y, random_state=10)

    out = model.predict_probs(X[:9], level=0.9)

    assert set(out.keys()) == {"probs", "conf_int_low", "conf_int_high"}
    assert out["probs"].shape == (9,)
    assert out["conf_int_low"].shape == (9,)
    assert out["conf_int_high"].shape == (9,)
    assert np.all(np.isfinite(out["probs"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))
    assert np.all(out["conf_int_low"] <= out["probs"])
    assert np.all(out["probs"] <= out["conf_int_high"])
    assert np.all((0.0 <= out["probs"]) & (out["probs"] <= 1.0))
    assert np.all((0.0 <= out["conf_int_low"]) & (out["conf_int_low"] <= 1.0))
    assert np.all((0.0 <= out["conf_int_high"]) & (out["conf_int_high"] <= 1.0))


def test_predict_probs_single_row_multivariate_returns_scalars():
    """Return scalar probability outputs when predicting a single row for a multivariate model."""
    X, y = _make_classification_data()
    model = _fit_small_model(X, y, random_state=11)

    out = model.predict_probs(X[0], level=0.9)

    assert np.isscalar(out["probs"])
    assert np.isscalar(out["conf_int_low"])
    assert np.isscalar(out["conf_int_high"])
    assert np.isfinite(out["probs"])
    assert 0.0 <= out["conf_int_low"] <= out["probs"] <= out["conf_int_high"] <= 1.0


def test_predict_returns_boolean_outputs_for_matrix():
    """Return a boolean class decision for each row when predicting a matrix."""
    X, y = _make_classification_data()
    model = _fit_small_model(X, y, random_state=12)

    pred = model.predict(X[:8], threshold=0.4)

    assert pred.shape == (8,)
    assert pred.dtype == np.bool_
    assert np.all(np.isin(pred, [False, True]))


def test_predict_returns_boolean_for_single_row():
    """Return a single boolean class decision when predicting one multivariate row."""
    X, y = _make_classification_data()
    model = _fit_small_model(X, y, random_state=13)

    pred = model.predict(X[0], threshold=0.5)

    assert isinstance(pred, (bool, np.bool_))


def test_variable_inclusion_has_correct_shape_and_finite_values():
    """Return one finite nonnegative inclusion value per predictor after fitting."""
    X, y = _make_classification_data(n=140, p=5, seed=14)
    model = _fit_small_model(X, y, random_state=14)

    vi = model.variable_inclusion()

    assert vi.shape == (5,)
    assert np.all(np.isfinite(vi))
    assert np.all(vi >= 0.0)


def test_fit_handles_all_one_labels_without_crashing():
    """Handle the degenerate all-positive-label case without producing NaNs or crashing."""
    rng = np.random.default_rng(15)
    X = rng.normal(size=(60, 3))
    y = np.ones(60, dtype=int)

    model = ProbitBart(m=6, n_burn=20, n_samples=30, random_state=15).fit(X, y)
    out = model.predict_probs(X[:5])

    assert np.all(np.isfinite(out["probs"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))
    assert np.all((0.0 <= out["probs"]) & (out["probs"] <= 1.0))


def test_fit_handles_all_zero_labels_without_crashing():
    """Handle the degenerate all-negative-label case without producing NaNs or crashing."""
    rng = np.random.default_rng(16)
    X = rng.normal(size=(60, 3))
    y = np.zeros(60, dtype=int)

    model = ProbitBart(m=6, n_burn=20, n_samples=30, random_state=16).fit(X, y)
    out = model.predict_probs(X[:5])

    assert np.all(np.isfinite(out["probs"]))
    assert np.all(np.isfinite(out["conf_int_low"]))
    assert np.all(np.isfinite(out["conf_int_high"]))
    assert np.all((0.0 <= out["probs"]) & (out["probs"] <= 1.0))


def test_fit_and_predict_probs_are_reproducible_with_fixed_seed():
    """Produce identical fitted probabilities and variable inclusion when the same seed is reused."""
    X, y = _make_classification_data(n=130, p=5, seed=17)

    model1 = ProbitBart(m=8, n_burn=25, n_samples=35, random_state=999).fit(X, y)
    model2 = ProbitBart(m=8, n_burn=25, n_samples=35, random_state=999).fit(X, y)

    pred1 = model1.predict_probs(X[:10], level=0.9)
    pred2 = model2.predict_probs(X[:10], level=0.9)

    np.testing.assert_allclose(pred1["probs"], pred2["probs"])
    np.testing.assert_allclose(pred1["conf_int_low"], pred2["conf_int_low"])
    np.testing.assert_allclose(pred1["conf_int_high"], pred2["conf_int_high"])
    np.testing.assert_allclose(model1.variable_inclusion(), model2.variable_inclusion())


def test_model_assigns_higher_average_probability_to_positive_training_examples():
    """Assign larger average fitted probabilities to positive than negative training examples on a structured signal."""
    X, y = _make_classification_data(n=180, p=4, seed=18)
    model = ProbitBart(m=8, n_burn=35, n_samples=45, random_state=18).fit(X, y)

    probs = model.predict_probs(X, level=0.9)["probs"]

    pos_mean = probs[y == 1].mean()
    neg_mean = probs[y == 0].mean()

    assert pos_mean > neg_mean


def test_probitbart_fit_continuous_random_high_dimensional_does_not_crash():
    """Fit ProbitBart on continuous random high-dimensional data without crashing and return finite probabilities."""
    rng = np.random.default_rng(12345)

    n_obs = 400
    X = rng.random((n_obs, 50))
    latent = _friedman_latent(X)
    probs = norm.cdf(latent)
    y = rng.binomial(1, probs).astype(int)

    model = ProbitBart(
        m=10,
        n_burn=150,
        n_samples=300,
        random_state=12345,
    ).fit(X, y)

    vi = model.variable_inclusion()
    pred = model.predict_probs(X[:8], level=0.9)

    assert vi.shape == (50,)
    assert np.all(np.isfinite(vi))
    assert pred["probs"].shape == (8,)
    assert np.all(np.isfinite(pred["probs"]))
    assert np.all(np.isfinite(pred["conf_int_low"]))
    assert np.all(np.isfinite(pred["conf_int_high"]))
    assert np.all((0.0 <= pred["probs"]) & (pred["probs"] <= 1.0))
