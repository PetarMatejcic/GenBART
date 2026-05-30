import numpy as np
import pytest

from genbart import RegBart, ProbitBart
from genbart.feature_selection import (
    BartPredictiveSelector,
    BartVariableSelector,
    PredictiveSelectionResult,
    ThresholdResult,
    VariableSelectionResult,
    build_threshold_results,
    global_max_threshold,
    global_se_threshold,
    local_threshold,
)
from genbart.feature_selection.utils import (
    VALID_IMPORTANCE_KINDS,
    get_feature_names,
    permute_column,
    permute_response,
    validate_xy,
)


def make_regression_data(seed=0, n=60, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 2] + rng.normal(scale=0.25, size=n)
    return X, y


def make_classification_data(seed=1, n=60, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    latent = 2.0 * X[:, 0] - 1.5 * X[:, 2]
    y = (latent > np.median(latent)).astype(int)
    return X, y


def small_regbart_params():
    return {
        "m": 8,
        "n_burn": 4,
        "n_samples": 8,
    }


def small_probitbart_params():
    return {
        "m": 8,
        "n_burn": 4,
        "n_samples": 8,
    }


def test_valid_importance_kinds_use_logml_name():
    assert VALID_IMPORTANCE_KINDS == ("raw", "logml")


def test_validate_xy_accepts_1d_x_as_single_feature():
    X = np.arange(5)
    y = np.arange(5)

    X_out, y_out = validate_xy(X, y)

    assert X_out.shape == (5, 1)
    assert y_out.shape == (5,)


def test_validate_xy_rejects_bad_shapes():
    with pytest.raises(ValueError, match="X must be"):
        validate_xy(np.zeros((2, 2, 2)), np.zeros(2))

    with pytest.raises(ValueError, match="y must be"):
        validate_xy(np.zeros((2, 2)), np.zeros((2, 1)))

    with pytest.raises(ValueError, match="same number of rows"):
        validate_xy(np.zeros((3, 2)), np.zeros(2))


def test_get_feature_names_from_explicit_names():
    X = np.zeros((4, 3))

    names = get_feature_names(X, ["a", "b", "c"], 3)

    assert names == ["a", "b", "c"]


def test_get_feature_names_default_names():
    X = np.zeros((4, 3))

    names = get_feature_names(X, None, 3)

    assert names == ["x0", "x1", "x2"]


def test_get_feature_names_rejects_wrong_length():
    X = np.zeros((4, 3))

    with pytest.raises(ValueError, match="feature_names"):
        get_feature_names(X, ["a", "b"], 3)


def test_permute_response_is_reproducible():
    y = np.arange(10)

    p1 = permute_response(y, seed=123)
    p2 = permute_response(y, seed=123)

    assert np.array_equal(p1, p2)
    assert sorted(p1.tolist()) == sorted(y.tolist())


def test_permute_column_only_changes_requested_column():
    rng = np.random.default_rng(123)
    X = np.arange(20).reshape(10, 2)

    X_perm = permute_column(X, column=1, rng=rng)

    assert np.array_equal(X_perm[:, 0], X[:, 0])
    assert sorted(X_perm[:, 1].tolist()) == sorted(X[:, 1].tolist())
    assert X_perm.shape == X.shape


def test_threshold_result_basic_methods():
    result = ThresholdResult(
        method="local",
        feature_names=["a", "b", "c"],
        importance=np.array([0.3, 0.1, 0.5]),
        thresholds=np.array([0.2, 0.2, 0.4]),
        selected=np.array([True, False, True]),
        quantile=0.95,
    )

    assert result.selected_features() == ["a", "c"]
    assert np.array_equal(result.selected_indices(), np.array([0, 2]))
    assert np.array_equal(result.selected_mask(), np.array([True, False, True]))
    assert result.n_selected() == 2
    assert result.threshold_for("a") == pytest.approx(0.2)
    assert result.importance_for("c") == pytest.approx(0.5)


def test_local_threshold():
    importance = np.array([0.4, 0.1, 0.7])
    null_importance = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.4],
            [0.3, 0.2, 0.5],
        ]
    )

    result = local_threshold(
        feature_names=["a", "b", "c"],
        importance=importance,
        null_importance=null_importance,
        alpha=0.5,
    )

    assert result.method == "local"
    assert result.thresholds.shape == (3,)
    assert np.array_equal(result.selected, np.array([True, False, True]))


def test_global_max_threshold():
    importance = np.array([0.4, 0.1, 0.7])
    null_importance = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.4],
            [0.3, 0.2, 0.5],
        ]
    )

    result = global_max_threshold(
        feature_names=["a", "b", "c"],
        importance=importance,
        null_importance=null_importance,
        alpha=0.5,
    )

    assert result.method == "global_max"
    assert np.allclose(result.thresholds, result.thresholds[0])
    assert "global_threshold" in result.extra
    assert "max_null" in result.extra


def test_global_se_threshold():
    importance = np.array([0.4, 0.1, 0.7])
    null_importance = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.4],
            [0.3, 0.2, 0.5],
        ]
    )

    result = global_se_threshold(
        feature_names=["a", "b", "c"],
        importance=importance,
        null_importance=null_importance,
        alpha=0.5,
    )

    assert result.method == "global_se"
    assert result.thresholds.shape == (3,)
    assert "global_se" in result.extra
    assert "null_mean" in result.extra
    assert "null_sd" in result.extra


def test_build_threshold_results_contains_all_methods():
    methods = build_threshold_results(
        feature_names=["a", "b", "c"],
        importance=np.array([0.4, 0.1, 0.7]),
        null_importance=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.2, 0.4],
                [0.3, 0.2, 0.5],
            ]
        ),
        alpha=0.5,
    )

    assert set(methods) == {"local", "global_max", "global_se"}


def test_variable_selection_result_methods():
    feature_names = ["a", "b", "c"]
    importance = np.array([0.4, 0.1, 0.7])
    null_importance = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.4],
            [0.3, 0.2, 0.5],
        ]
    )

    methods = build_threshold_results(
        feature_names=feature_names,
        importance=importance,
        null_importance=null_importance,
        alpha=0.5,
    )

    result = VariableSelectionResult(
        feature_names=feature_names,
        importance=importance,
        importance_repeats=np.vstack([importance, importance]),
        importance_sd=np.array([0.01, 0.02, 0.03]),
        null_importance=null_importance,
        methods=methods,
        default_method="local",
        importance_kind="raw",
    )

    assert result.selected_features() == ["a", "c"]
    assert np.array_equal(result.ranking(), np.array([2, 0, 1]))
    assert result.to_frame()[0]["feature"] == "c"
    assert result.summary()["top_feature"] == "c"
    assert len(result.compare_methods()) == 3


def test_predictive_selection_result_methods():
    result = PredictiveSelectionResult(
        feature_names=["a", "b", "c"],
        importance_mean=np.array([0.3, -0.1, 0.2]),
        importance_sd=np.array([0.01, 0.02, 0.03]),
        prob_positive=np.array([1.0, 0.0, 0.8]),
        degradation_samples=np.array(
            [
                [0.3, -0.1, 0.2],
                [0.4, -0.2, 0.1],
            ]
        ),
        selected=np.array([True, False, True]),
        loss_metric="brier",
        task="classification",
        selection_probability=0.75,
        min_mean_degradation=0.0,
        baseline_loss_mean=0.2,
        n_repeats=2,
        n_permutations=3,
        use_posterior_draws=False,
    )

    assert result.selected_features() == ["a", "c"]
    assert np.array_equal(result.selected_indices(), np.array([0, 2]))
    assert np.array_equal(result.ranking(), np.array([0, 2, 1]))
    assert result.to_frame()[0]["feature"] == "a"
    assert result.summary()["n_permutations"] == 3


def test_regbart_predict_and_posterior_sample_draws_api():
    X, y = make_regression_data()

    model = RegBart(**small_regbart_params(), random_state=10)
    model.fit(X, y)

    pred = model.predict(X[:4], conf_int=False)["prediction"]
    draws = model.posterior_sample_draws(X[:4])

    assert np.asarray(pred).shape == (4,)
    assert draws.shape == (small_regbart_params()["n_samples"], 4)
    assert np.all(np.isfinite(draws))


def test_probitbart_predict_and_posterior_sample_draws_api():
    X, y = make_classification_data()

    model = ProbitBart(**small_probitbart_params(), random_state=11)
    model.fit(X, y)

    prob = model.predict_probs(X[:4])["prediction"]
    label = model.predict(X[:4])
    draws = model.posterior_sample_draws(X[:4])

    assert np.asarray(prob).shape == (4,)
    assert np.asarray(label).shape == (4,)
    assert draws.shape == (small_probitbart_params()["n_samples"], 4)
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_basebart_variable_inclusion_supports_raw_and_logml():
    X, y = make_regression_data()

    model = RegBart(**small_regbart_params(), random_state=12)
    model.fit(X, y)

    raw = model.variable_inclusion(kind="raw")
    logml = model.variable_inclusion(kind="logml")

    assert raw.shape == (X.shape[1],)
    assert logml.shape == (X.shape[1],)
    assert np.all(raw >= 0.0)
    assert np.all(logml >= 0.0)

    with pytest.raises(ValueError, match="raw.*logml|logml.*raw"):
        model.variable_inclusion(kind="logml_weighted")


def test_bart_variable_selector_classification_raw():
    X, y = make_classification_data()

    selector = BartVariableSelector(
        ProbitBart,
        model_params=small_probitbart_params(),
        n_permutations=2,
        n_repeats=2,
        alpha=0.2,
        method="global_se",
        importance_kind="raw",
        random_state=20,
    )

    result = selector.fit(
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    assert isinstance(result, VariableSelectionResult)
    assert result.importance.shape == (X.shape[1],)
    assert result.importance_repeats.shape == (2, X.shape[1])
    assert result.null_importance.shape == (2, X.shape[1])
    assert result.selected_mask().shape == (X.shape[1],)
    assert result.importance_kind == "raw"
    assert set(result.methods) == {"local", "global_max", "global_se"}


def test_bart_variable_selector_regression_logml():
    X, y = make_regression_data()

    selector = BartVariableSelector(
        RegBart,
        model_params=small_regbart_params(),
        n_permutations=2,
        n_repeats=2,
        alpha=0.2,
        method="local",
        importance_kind="logml",
        random_state=21,
    )

    result = selector.fit(
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    assert isinstance(result, VariableSelectionResult)
    assert result.importance.shape == (X.shape[1],)
    assert result.null_importance.shape == (2, X.shape[1])
    assert result.importance_kind == "logml"
    assert result.summary()["method"] == "local"


def test_bart_predictive_selector_classification_point_predictions():
    X, y = make_classification_data()

    selector = BartPredictiveSelector(
        ProbitBart,
        model_params=small_probitbart_params(),
        task="classification",
        loss_metric="brier",
        n_repeats=2,
        n_permutations=2,
        selection_probability=0.5,
        min_mean_degradation=0.0,
        use_posterior_draws=False,
        random_state=30,
    )

    result = selector.fit(
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    assert isinstance(result, PredictiveSelectionResult)
    assert result.importance_mean.shape == (X.shape[1],)
    assert result.importance_sd.shape == (X.shape[1],)
    assert result.prob_positive.shape == (X.shape[1],)
    assert result.selected_mask().shape == (X.shape[1],)
    assert result.n_permutations == 2
    assert result.use_posterior_draws is False


def test_bart_predictive_selector_regression_point_predictions():
    X, y = make_regression_data()

    selector = BartPredictiveSelector(
        RegBart,
        model_params=small_regbart_params(),
        task="regression",
        loss_metric="mse",
        n_repeats=2,
        n_permutations=2,
        selection_probability=0.5,
        min_mean_degradation=0.0,
        use_posterior_draws=False,
        random_state=31,
    )

    result = selector.fit(
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    assert isinstance(result, PredictiveSelectionResult)
    assert result.importance_mean.shape == (X.shape[1],)
    assert result.importance_sd.shape == (X.shape[1],)
    assert result.prob_positive.shape == (X.shape[1],)
    assert result.selected_mask().shape == (X.shape[1],)
    assert result.n_permutations == 2
    assert result.use_posterior_draws is False


def test_bart_predictive_selector_with_posterior_draws():
    X, y = make_classification_data()

    selector = BartPredictiveSelector(
        ProbitBart,
        model_params=small_probitbart_params(),
        task="classification",
        loss_metric="brier",
        n_repeats=1,
        n_permutations=2,
        selection_probability=0.5,
        min_mean_degradation=0.0,
        use_posterior_draws=True,
        random_state=32,
    )

    result = selector.fit(
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    expected_rows = (
        selector.n_repeats
        * selector.n_permutations
        * small_probitbart_params()["n_samples"]
    )

    assert isinstance(result, PredictiveSelectionResult)
    assert result.degradation_samples.shape == (expected_rows, X.shape[1])
    assert result.use_posterior_draws is True


def test_bart_predictive_selector_accepts_eval_data():
    X, y = make_regression_data(n=70)
    X_train, y_train = X[:50], y[:50]
    X_eval, y_eval = X[50:], y[50:]

    selector = BartPredictiveSelector(
        RegBart,
        model_params=small_regbart_params(),
        task="regression",
        loss_metric="mse",
        n_repeats=1,
        n_permutations=2,
        selection_probability=0.5,
        min_mean_degradation=0.0,
        use_posterior_draws=False,
        random_state=33,
    )

    result = selector.fit(
        X_train,
        y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        feature_names=["x0", "x1", "x2", "x3", "x4"],
    )

    assert isinstance(result, PredictiveSelectionResult)
    assert result.degradation_samples.shape[1] == X.shape[1]


def test_selector_from_model_constructors():
    reg_model = RegBart(**small_regbart_params(), random_state=100)
    probit_model = ProbitBart(**small_probitbart_params(), random_state=101)

    variable_selector = BartVariableSelector.from_model(
        reg_model,
        n_permutations=2,
        n_repeats=1,
        random_state=40,
    )
    predictive_selector = BartPredictiveSelector.from_model(
        probit_model,
        task="classification",
        loss_metric="brier",
        n_repeats=1,
        n_permutations=2,
        random_state=41,
    )

    assert variable_selector.model_cls is RegBart
    assert predictive_selector.model_cls is ProbitBart