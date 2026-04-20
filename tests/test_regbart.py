import numpy as np
import pytest

from genbart.reg_bart import RegBart


def make_step_data(n=60):
    x = np.linspace(0.0, 1.0, n)
    X = x.reshape(-1, 1)
    y = np.where(x <= 0.5, -1.0, 1.0)
    return X, y


def make_two_feature_signal_data(n=80, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = np.where(X[:, 0] <= 0.5, -1.0, 1.0) + rng.normal(0.0, 0.03, size=n)
    return X, y


def make_model(**kwargs):
    params = dict(
        m=8,
        n_burn=20,
        n_samples=20,
        alpha=0.95,
        beta=2.0,
        k=2.0,
        nu=3.0,
        q=0.90,
        random_state=123,
    )
    params.update(kwargs)
    return RegBart(**params)


def test_predict_before_fit_raises():
    model = make_model()
    X = np.array([[0.1], [0.2]], dtype=float)

    with pytest.raises(RuntimeError):
        model.predict(X)


def test_fit_returns_self_and_builds_engine_state():
    X, y = make_step_data(n=40)

    model = make_model(m=6, n_burn=10, n_samples=10)
    out = model.fit(X, y)

    assert out is model
    assert model.engine is not None
    assert model.packed_forest is not None

    assert model.training_predictions.shape == (model.m, X.shape[0])
    assert model.fitted_sums.shape == (X.shape[0],)
    assert model.residuals.shape == (X.shape[0],)

    assert np.isfinite(model.training_predictions).all()
    assert np.isfinite(model.fitted_sums).all()
    assert np.isfinite(model.residuals).all()


def test_predict_shapes_for_single_row_and_matrix():
    X, y = make_two_feature_signal_data(n=60, seed=11)

    model = make_model(m=6, n_burn=10, n_samples=12, random_state=11).fit(X, y)

    single = model.predict(X[0], level=0.80)
    assert np.isscalar(single["prediction"])
    assert np.isscalar(single["conf_int_low"])
    assert np.isscalar(single["conf_int_high"])
    assert single["conf_int_low"] <= single["prediction"] <= single["conf_int_high"]

    batch = model.predict(X[:7], level=0.80)
    assert batch["prediction"].shape == (7,)
    assert batch["conf_int_low"].shape == (7,)
    assert batch["conf_int_high"].shape == (7,)
    assert np.all(batch["conf_int_low"] <= batch["prediction"])
    assert np.all(batch["prediction"] <= batch["conf_int_high"])


def test_predict_without_conf_int_returns_only_prediction():
    X, y = make_step_data(n=40)

    model = make_model(m=6, n_burn=10, n_samples=10).fit(X, y)
    out = model.predict(X, conf_int=False)

    assert set(out.keys()) == {"prediction"}
    assert out["prediction"].shape == (X.shape[0],)


def test_predict_supports_mean_and_median():
    X, y = make_two_feature_signal_data(n=50, seed=21)

    model = make_model(m=6, n_burn=10, n_samples=12, random_state=21).fit(X, y)

    mean_pred = model.predict(X[:10], central_measure="mean", conf_int=False)["prediction"]
    median_pred = model.predict(X[:10], central_measure="median", conf_int=False)["prediction"]

    assert mean_pred.shape == (10,)
    assert median_pred.shape == (10,)
    assert np.isfinite(mean_pred).all()
    assert np.isfinite(median_pred).all()


def test_step_function_training_rmse_is_reasonable():
    X, y = make_step_data(n=60)

    model = make_model(
        m=8,
        n_burn=25,
        n_samples=25,
        random_state=7,
    ).fit(X, y)

    pred = model.predict(X, conf_int=False)["prediction"]
    rmse = np.sqrt(np.mean((pred - y) ** 2))

    assert rmse < 0.35


def test_variable_importance_prefers_signal_variable():
    X, y = make_two_feature_signal_data(n=100, seed=33)

    model = make_model(
        m=5,          # smaller m makes the importance competition sharper
        n_burn=25,
        n_samples=25,
        random_state=33,
    ).fit(X, y)

    vi = model.variable_importance()

    assert vi.shape == (2,)
    assert np.isfinite(vi).all()
    assert np.isclose(vi.sum(), 1.0, atol=0.15)
    assert vi[0] > vi[1]


def test_marginalize_returns_expected_shapes():
    X, y = make_two_feature_signal_data(n=80, seed=44)

    model = make_model(m=6, n_burn=12, n_samples=12, random_state=44).fit(X, y)

    grid = np.linspace(0.0, 1.0, 9)
    out = model.marginalize(variable=0, grid=grid, sampling_size=40, level=0.80)

    assert out["prediction"].shape == (9,)
    assert out["conf_int_low"].shape == (9,)
    assert out["conf_int_high"].shape == (9,)
    assert np.all(out["conf_int_low"] <= out["prediction"])
    assert np.all(out["prediction"] <= out["conf_int_high"])


def test_single_feature_vector_input_is_treated_as_many_rows():
    X, y = make_step_data(n=50)

    model = make_model(m=6, n_burn=10, n_samples=10, random_state=55).fit(X, y)

    xvec = X[:8, 0]  # shape (8,), should be reshaped to (8, 1)
    out = model.predict(xvec, conf_int=False)

    assert out["prediction"].shape == (8,)
    assert np.isfinite(out["prediction"]).all()


def test_extreme_values_are_populated_for_marginalization():
    X, y = make_two_feature_signal_data(n=40, seed=66)

    model = make_model(m=4, n_burn=8, n_samples=8, random_state=66).fit(X, y)

    assert len(model.extreme_values) == X.shape[1]
    for lo, hi in model.extreme_values:
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo <= hi
