import numpy as np
import pytest

import genbart.variable_selection as vs


BartVariableSelector = getattr(
    vs,
    "BartVariableSelector",
    getattr(vs, "BartVariableSelection", None),
)


class FakeBart:
    """
    Fast stand-in for RegBart / ProbitBart.

    It exposes the minimal interface needed by BartVariableSelector:
        __init__(random_state=...)
        fit(X, y)
        variable_inclusion()

    The returned variable inclusion proportions are based on absolute
    marginal association between each column of X and y, plus tiny
    deterministic jitter controlled by random_state.
    """

    def __init__(self, random_state=0, **kwargs):
        self.random_state = int(random_state)
        self.kwargs = dict(kwargs)

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def variable_inclusion(self):
        X = self.X_
        y = self.y_

        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()

        scores = np.abs(X_centered.T @ y_centered)

        rng = np.random.default_rng(self.random_state)
        scores = scores + rng.uniform(0.0, 1e-9, size=X.shape[1])

        total = scores.sum()
        if total <= 0:
            return np.full(X.shape[1], 1.0 / X.shape[1])

        return scores / total


class FakeBartWithParams:
    def __init__(
        self,
        m=20,
        n_burn=100,
        n_samples=200,
        random_state=0,
    ):
        self.m = m
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.random_state = random_state

    def get_params(self):
        return {
            "m": self.m,
            "n_burn": self.n_burn,
            "n_samples": self.n_samples,
            "random_state": self.random_state,
        }

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def variable_inclusion(self):
        p = self.X_.shape[1]
        out = np.zeros(p)
        out[0] = 1.0
        return out

@pytest.fixture
def signal_data():
    rng = np.random.default_rng(123)

    X = rng.normal(size=(80, 5))
    y = 3.0 * X[:, 0] + 0.25 * X[:, 1] + rng.normal(scale=0.1, size=80)

    feature_names = [f"x{j}" for j in range(X.shape[1])]
    return X, y, feature_names


@pytest.fixture
def selector(signal_data):
    _, _, feature_names = signal_data

    return BartVariableSelector(
        model_cls=FakeBart,
        model_params={"dummy_param": 10},
        n_permutations=6,
        n_repeats=3,
        alpha=0.05,
        method="global_se",
        random_state=42,
        verbose=False,
    )


def as_result(obj):
    """
    Accept either:
        result = selector.fit(X, y)
    or:
        selector.fit(X, y)
        result = selector.result_
    """
    return getattr(obj, "result_", obj)


def normalize_method(method):
    if method == "global":
        return "global_max"
    return method


def get_thresholds(result, method):
    method = normalize_method(method)

    if hasattr(result, "thresholds"):
        return np.asarray(result.thresholds(method), dtype=float)

    return np.asarray(result.methods[method].thresholds, dtype=float)


def get_selected_mask(result, method):
    method = normalize_method(method)

    if hasattr(result, "selected_mask"):
        return np.asarray(result.selected_mask(method), dtype=bool)

    return np.asarray(result.methods[method].selected, dtype=bool)


def get_selected_features(result, method):
    if hasattr(result, "selected_features"):
        return result.selected_features(method)

    return result.methods[normalize_method(method)].selected_features()


def result_to_rows(result, method="global_se"):
    frame = result.to_frame(method)

    if hasattr(frame, "to_dict"):
        return frame.to_dict(orient="records")

    return frame


def build_result_from_arrays(real_vips, null_vips, feature_names=None, quantile=0.95):
    """
    Builds a VariableSelectionResult through the selector's private result builder.

    This lets us test threshold calculations exactly without running model fits.
    """
    real_vips = np.asarray(real_vips, dtype=float)
    null_vips = np.asarray(null_vips, dtype=float)

    if feature_names is None:
        feature_names = [f"x{j}" for j in range(real_vips.shape[0])]

    selector = BartVariableSelector(
        model_cls=FakeBart,
        model_params={},
        n_permutations=null_vips.shape[0],
        n_repeats=1,
        alpha=1.0 - quantile,
        method="global_se",
        random_state=0,
    )

    selector.feature_names = list(feature_names)
    selector.real_vips = real_vips
    selector.real_vips_repeats = real_vips.reshape(1, -1)
    selector.real_vips_sd = np.zeros_like(real_vips)
    selector.null_vips = null_vips

    # Optional fields, in case your result builder uses cached summaries.
    selector.null_vips_mean = null_vips.mean(axis=0)
    selector.null_vips_sd = (
        null_vips.std(axis=0, ddof=1)
        if null_vips.shape[0] > 1
        else np.zeros(null_vips.shape[1])
    )

    if not hasattr(selector, "_build_result"):
        pytest.fail("BartVariableSelector needs a _build_result() method.")

    try:
        return selector._build_result(quantile=quantile)
    except TypeError:
        return selector._build_result()


def expected_global_se_thresholds(null_vips, quantile):
    null_vips = np.asarray(null_vips, dtype=float)

    null_mean = null_vips.mean(axis=0)
    null_sd = null_vips.std(axis=0, ddof=1)

    safe_sd = null_sd.copy()
    safe_sd[safe_sd == 0.0] = 1.0

    standardized = (null_vips - null_mean) / safe_sd
    max_standardized = np.max(standardized, axis=1)

    global_se = np.quantile(max_standardized, quantile)
    thresholds = null_mean + global_se * null_sd

    return thresholds, global_se


def test_selector_class_is_available():
    assert BartVariableSelector is not None


def test_constructor_rejects_non_callable_model():
    with pytest.raises(TypeError):
        BartVariableSelector(model_cls=None)


def test_constructor_rejects_invalid_repeats():
    with pytest.raises(ValueError):
        BartVariableSelector(model_cls=FakeBart, n_repeats=0)


def test_constructor_rejects_invalid_permutations():
    with pytest.raises(ValueError):
        BartVariableSelector(model_cls=FakeBart, n_permutations=0)


def test_validate_xy_rejects_bad_shapes(selector):
    X = np.ones((10, 2))
    y = np.ones(9)

    with pytest.raises(ValueError):
        selector._validate_xy(X, y)

    with pytest.raises(ValueError):
        selector._validate_xy(np.ones((2, 3, 4)), np.ones(2))

    with pytest.raises(ValueError):
        selector._validate_xy(np.ones((10, 2)), np.ones((10, 1)))


def test_get_feature_names_default(selector):
    X = np.ones((10, 3))
    names = selector._get_feature_names(X, feature_names=None, p=3)

    assert names == ["x0", "x1", "x2"]


def test_get_feature_names_rejects_wrong_length(selector):
    X = np.ones((10, 3))

    with pytest.raises(ValueError):
        selector._get_feature_names(X, feature_names=["a", "b"], p=3)


def test_make_model_creates_fresh_models_with_seed(selector):
    model_1 = selector._make_model(seed=11)
    model_2 = selector._make_model(seed=12)

    assert isinstance(model_1, FakeBart)
    assert isinstance(model_2, FakeBart)
    assert model_1 is not model_2
    assert model_1.random_state == 11
    assert model_2.random_state == 12
    assert model_1.kwargs["dummy_param"] == 10


def test_fit_returns_result_with_expected_shapes(selector, signal_data):
    X, y, feature_names = signal_data

    result = as_result(selector.fit(X, y, feature_names=feature_names))

    assert np.asarray(result.real_vips).shape == (5,)
    assert np.asarray(result.real_vips_repeats).shape == (3, 5)
    assert np.asarray(result.real_vips_sd).shape == (5,)
    assert np.asarray(result.null_vips).shape == (6, 5)

    assert list(result.feature_names) == feature_names


def test_fit_builds_all_threshold_methods(selector, signal_data):
    X, y, feature_names = signal_data

    result = as_result(selector.fit(X, y, feature_names=feature_names))

    assert hasattr(result, "methods")

    methods = set(result.methods)
    assert {"local", "global_max", "global_se"}.issubset(methods)


def test_fit_is_reproducible(signal_data):
    X, y, feature_names = signal_data

    selector_1 = BartVariableSelector(
        model_cls=FakeBart,
        model_params={},
        n_permutations=5,
        n_repeats=2,
        alpha=0.05,
        method="global_se",
        random_state=777,
    )

    selector_2 = BartVariableSelector(
        model_cls=FakeBart,
        model_params={},
        n_permutations=5,
        n_repeats=2,
        alpha=0.05,
        method="global_se",
        random_state=777,
    )

    result_1 = as_result(selector_1.fit(X, y, feature_names=feature_names))
    result_2 = as_result(selector_2.fit(X, y, feature_names=feature_names))

    np.testing.assert_allclose(result_1.real_vips, result_2.real_vips)
    np.testing.assert_allclose(result_1.real_vips_repeats, result_2.real_vips_repeats)
    np.testing.assert_allclose(result_1.null_vips, result_2.null_vips)


def test_signal_variable_is_ranked_first(selector, signal_data):
    X, y, feature_names = signal_data

    result = as_result(selector.fit(X, y, feature_names=feature_names))
    top_idx = int(np.argmax(result.real_vips))

    assert feature_names[top_idx] == "x0"


def test_local_thresholds_are_variablewise_quantiles():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    expected = np.quantile(null_vips, 0.95, axis=0)
    observed = get_thresholds(result, "local")

    np.testing.assert_allclose(observed, expected)


def test_global_max_threshold_is_permutationwise_max_quantile():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    expected_scalar = np.quantile(np.max(null_vips, axis=1), 0.95)
    expected = np.full(real_vips.shape, expected_scalar)

    observed = get_thresholds(result, "global_max")

    np.testing.assert_allclose(observed, expected)
    assert np.allclose(observed, observed[0])


def test_global_se_thresholds_match_formula():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    expected, expected_global_se = expected_global_se_thresholds(null_vips, 0.95)
    observed = get_thresholds(result, "global_se")

    np.testing.assert_allclose(observed, expected)

    method_result = result.methods["global_se"]
    if hasattr(method_result, "extra") and "global_se" in method_result.extra:
        assert method_result.extra["global_se"] == pytest.approx(expected_global_se)


def test_global_alias_matches_global_max():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    np.testing.assert_allclose(
        get_thresholds(result, "global"),
        get_thresholds(result, "global_max"),
    )


def test_selected_mask_matches_real_vips_greater_than_thresholds():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    for method in ["local", "global_max", "global_se"]:
        thresholds = get_thresholds(result, method)
        selected = get_selected_mask(result, method)

        np.testing.assert_array_equal(selected, real_vips > thresholds)


def test_selected_features_returns_feature_names():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])
    feature_names = ["signal", "weak", "noise"]

    result = build_result_from_arrays(
        real_vips,
        null_vips,
        feature_names=feature_names,
        quantile=0.95,
    )

    selected = get_selected_features(result, "global_se")

    assert isinstance(selected, list)
    assert "signal" in selected


def test_to_frame_contains_expected_columns():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    rows = result_to_rows(result, method="global_se")
    assert len(rows) == 3

    required = {
        "rank",
        "feature",
        "vip",
        "vip_sd",
        "null_mean",
        "null_sd",
        "threshold",
        "selected",
        "method",
    }

    assert required.issubset(set(rows[0].keys()))


def test_to_frame_is_sorted_by_decreasing_vip():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.05, 0.20, 0.015])
    feature_names = ["weak", "signal", "noise"]

    result = build_result_from_arrays(
        real_vips,
        null_vips,
        feature_names=feature_names,
        quantile=0.95,
    )

    rows = result_to_rows(result, method="global_se")

    assert rows[0]["feature"] == "signal"
    assert rows[0]["rank"] == 1


def test_summary_has_expected_keys():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    summary = result.summary("global_se")

    assert summary["method"] == "global_se"
    assert summary["n_features"] == 3
    assert "n_selected" in summary
    assert "selected_features" in summary
    assert "top_feature" in summary
    assert "top_vip" in summary


def test_compare_methods_contains_all_methods():
    null_vips = np.array([
        [0.10, 0.05, 0.01],
        [0.15, 0.04, 0.02],
        [0.08, 0.07, 0.03],
        [0.12, 0.06, 0.02],
    ])
    real_vips = np.array([0.20, 0.05, 0.015])

    result = build_result_from_arrays(real_vips, null_vips, quantile=0.95)

    comparison = result.compare_methods()

    if hasattr(comparison, "to_dict"):
        rows = comparison.to_dict(orient="records")
    else:
        rows = comparison

    methods = {row["method"] for row in rows}

    assert {"local", "global_max", "global_se"}.issubset(methods)


def test_from_model_uses_model_class_and_params():
    model = FakeBartWithParams(
        m=30,
        n_burn=50,
        n_samples=75,
        random_state=999,
    )

    selector = BartVariableSelector.from_model(
        model,
        n_permutations=5,
        n_repeats=2,
        random_state=123,
    )

    assert selector.model_cls is FakeBartWithParams
    assert selector.model_params["m"] == 30
    assert selector.model_params["n_burn"] == 50
    assert selector.model_params["n_samples"] == 75
    assert selector.model_params["random_state"] == 999

    assert selector.n_permutations == 5
    assert selector.n_repeats == 2
    assert selector.random_state == 123


def test_from_model_make_model_overrides_random_state():
    model = FakeBartWithParams(random_state=999)

    selector = BartVariableSelector.from_model(
        model,
        random_state=123,
    )

    new_model = selector._make_model(seed=42)

    assert isinstance(new_model, FakeBartWithParams)
    assert new_model.random_state == 42


def test_from_model_rejects_model_class():
    with pytest.raises(TypeError):
        BartVariableSelector.from_model(FakeBartWithParams)


class FakeNoGetParams:
    pass
def test_from_model_rejects_model_without_get_params():
    with pytest.raises(TypeError):
        BartVariableSelector.from_model(FakeNoGetParams())


class FakeBadGetParams:
    def get_params(self):
        return ["not", "a", "dict"]
def test_from_model_rejects_non_dict_get_params():
    with pytest.raises(TypeError):
        BartVariableSelector.from_model(FakeBadGetParams())