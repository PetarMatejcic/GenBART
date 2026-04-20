import numpy as np
import pytest

from genbart._backend import _BackfittingEngine
from genbart.tree import Tree as PyTree


def _root_rows_by_var(X: np.ndarray):
    return [np.argsort(X[:, j], kind="mergesort") for j in range(X.shape[1])]


def test_initialize_root_forest_and_validate():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )

    eng = _BackfittingEngine(X, m=3, seed=123)
    eng.initialize_root_forest()

    assert eng.n() == 4
    assert eng.p() == 2
    assert eng.m() == 3

    eng.validate_forest()


def test_serialize_tree_matches_python_root_tree():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )

    eng = _BackfittingEngine(X, m=1, seed=123)
    eng.initialize_root_forest()

    var_cpp, val_cpp, left_cpp, right_cpp, mu_cpp = eng.serialize_tree(0)

    py_tree = PyTree(data=X, rows_by_var=_root_rows_by_var(X))
    st = py_tree.serialize()

    np.testing.assert_array_equal(var_cpp, st.variable)
    np.testing.assert_allclose(val_cpp, st.value)
    np.testing.assert_array_equal(left_cpp, st.left)
    np.testing.assert_array_equal(right_cpp, st.right)
    np.testing.assert_allclose(mu_cpp, st.mu)


def test_draw_mu_root_tree_is_deterministic_when_sigma2_zero():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )
    residuals = np.array([1.0, 3.0, -2.0, 2.0], dtype=float)

    eng = _BackfittingEngine(X, m=1, seed=123)
    eng.initialize_root_forest()

    eng.draw_mu(
        0,
        residuals,
        sigma2=0.0,
        sigma_mu2=1.0,
    )

    _, _, _, _, mu = eng.serialize_tree(0)

    expected_mu = residuals.mean()
    np.testing.assert_allclose(mu, np.array([expected_mu], dtype=np.float32))


def test_refresh_tree_training_predictions_root_tree():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )
    residuals = np.array([1.0, 3.0, -2.0, 2.0], dtype=float)

    eng = _BackfittingEngine(X, m=1, seed=123)
    eng.initialize_root_forest()

    eng.draw_mu(
        0,
        residuals,
        sigma2=0.0,
        sigma_mu2=1.0,
    )

    training_predictions = np.zeros((1, X.shape[0]), dtype=float)
    fitted_sums = np.zeros(X.shape[0], dtype=float)

    eng.refresh_tree_training_predictions(
        0,
        training_predictions,
        fitted_sums,
    )

    expected_mu = residuals.mean()
    expected = np.full(X.shape[0], expected_mu, dtype=float)

    np.testing.assert_allclose(training_predictions[0], expected)
    np.testing.assert_allclose(fitted_sums, expected)


def test_bad_shapes_raise():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )

    eng = _BackfittingEngine(X, m=2, seed=123)
    eng.initialize_root_forest()

    with pytest.raises(RuntimeError):
        eng.draw_mu(0, np.zeros((4, 1), dtype=float), sigma2=1.0, sigma_mu2=1.0)

    with pytest.raises(RuntimeError):
        eng.refresh_tree_training_predictions(
            0,
            np.zeros((4, 2), dtype=float),   # wrong shape
            np.zeros(X.shape[0], dtype=float),
        )

    with pytest.raises(RuntimeError):
        eng.refresh_tree_training_predictions(
            0,
            np.zeros((2, X.shape[0]), dtype=float),
            np.zeros((X.shape[0], 1), dtype=float),  # wrong ndim
        )
