# tests/test_model_selection.py

import numpy as np
import pytest

from genbart.model_selection import (
    cross_validate_reg_bart,
    cross_validate_probit_bart,
)
from genbart.reg_bart import RegBart
from genbart.probit_bart import ProbitBart


def make_regression_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(20, 3))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=20)
    return X, y


def make_classification_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(24, 3))
    logits = 2.0 * X[:, 0] - X[:, 1]
    y = (logits > np.median(logits)).astype(int)
    return X, y


def test_cross_validate_reg_bart_returns_expected_result():
    X, y = make_regression_data()

    result = cross_validate_reg_bart(
        X,
        y,
        param_grid={
            "m": [2],
            "k": [1.0, 2.0],
            "nu": [3.0],
            "q": [0.90],
            "n_burn": [1],
            "n_samples": [2],
        },
        cv=2,
        random_state=123,
        refit=True,
    )

    assert set(result) == {
        "best_params",
        "best_score",
        "best_model",
        "cv_results",
    }

    assert isinstance(result["best_params"], dict)
    assert np.isfinite(result["best_score"])
    assert isinstance(result["best_model"], RegBart)
    assert len(result["cv_results"]) == 2

    pred = result["best_model"].predict(X[:5], conf_int=False)["prediction"]
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


def test_cross_validate_reg_bart_refit_false():
    X, y = make_regression_data()

    result = cross_validate_reg_bart(
        X,
        y,
        param_grid={
            "m": [2],
            "k": [2.0],
            "nu": [3.0],
            "q": [0.90],
            "n_burn": [1],
            "n_samples": [2],
        },
        cv=2,
        random_state=123,
        refit=False,
    )

    assert result["best_model"] is None
    assert isinstance(result["best_params"], dict)
    assert np.isfinite(result["best_score"])


def test_cross_validate_reg_bart_invalid_scoring_raises():
    X, y = make_regression_data()

    with pytest.raises(ValueError):
        cross_validate_reg_bart(
            X,
            y,
            param_grid={
                "m": [2],
                "k": [2.0],
                "nu": [3.0],
                "q": [0.90],
                "n_burn": [1],
                "n_samples": [2],
            },
            cv=2,
            scoring="bad_score",
            random_state=123,
        )


def test_cross_validate_probit_bart_returns_expected_result():
    X, y = make_classification_data()

    result = cross_validate_probit_bart(
        X,
        y,
        param_grid={
            "m": [2],
            "k": [1.0, 2.0],
            "n_burn": [1],
            "n_samples": [2],
        },
        cv=2,
        scoring="auc",
        random_state=123,
        refit=True,
    )

    assert set(result) == {
        "best_params",
        "best_score",
        "best_model",
        "cv_results",
    }

    assert isinstance(result["best_params"], dict)
    assert np.isfinite(result["best_score"])
    assert isinstance(result["best_model"], ProbitBart)
    assert len(result["cv_results"]) == 2

    probs = result["best_model"].predict_probs(X[:5])["probs"]
    assert probs.shape == (5,)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_cross_validate_probit_bart_accuracy_scoring():
    X, y = make_classification_data()

    result = cross_validate_probit_bart(
        X,
        y,
        param_grid={
            "m": [2],
            "k": [2.0],
            "n_burn": [1],
            "n_samples": [2],
        },
        cv=2,
        scoring="accuracy",
        random_state=123,
        refit=True,
    )

    assert 0.0 <= result["best_score"] <= 1.0
    assert isinstance(result["best_model"], ProbitBart)


def test_cross_validate_probit_bart_invalid_scoring_raises():
    X, y = make_classification_data()

    with pytest.raises(ValueError):
        cross_validate_probit_bart(
            X,
            y,
            param_grid={
                "m": [2],
                "k": [2.0],
                "n_burn": [1],
                "n_samples": [2],
            },
            cv=2,
            scoring="bad_score",
            random_state=123,
        )