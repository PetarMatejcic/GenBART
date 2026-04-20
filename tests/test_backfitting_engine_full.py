import numpy as np
import pytest

from genbart._backend import _BackfittingEngine


GROW_WITH_REVERSE = (1.0 - 1e-12, 1e-12, 0.0, 0.0)
PRUNE_WITH_REVERSE = (1e-12, 1.0 - 1e-12, 0.0, 0.0)
CHANGE_ONLY = (0.0, 0.0, 1.0, 0.0)
SWAP_ONLY = (0.0, 0.0, 0.0, 1.0)


def make_engine(X, m=1, seed=123):
    eng = _BackfittingEngine(np.asarray(X, dtype=float), m=m, seed=seed)
    eng.initialize_root_forest()
    return eng


def test_constructor_rejects_bad_inputs():
    with pytest.raises(RuntimeError):
        _BackfittingEngine(np.array([1.0, 2.0, 3.0]), m=1, seed=1)

    with pytest.raises(RuntimeError):
        _BackfittingEngine(np.zeros((3, 2), dtype=float), m=0, seed=1)

    with pytest.raises(RuntimeError):
        _BackfittingEngine(np.zeros((0, 2), dtype=float), m=1, seed=1)

    with pytest.raises(RuntimeError):
        _BackfittingEngine(np.zeros((3, 0), dtype=float), m=1, seed=1)


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

    eng = make_engine(X, m=3, seed=7)

    assert eng.n() == 4
    assert eng.p() == 2
    assert eng.m() == 3

    eng.validate_tree(0)
    eng.validate_tree(1)
    eng.validate_tree(2)
    eng.validate_forest()


def test_root_tree_serialization_is_exact():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )

    eng = make_engine(X, m=1, seed=7)
    variable, value, left, right, mu = eng.serialize_tree(0)

    np.testing.assert_array_equal(variable, np.array([-1], dtype=np.int32))
    np.testing.assert_allclose(value, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(left, np.array([-1], dtype=np.int32))
    np.testing.assert_array_equal(right, np.array([-1], dtype=np.int32))
    np.testing.assert_allclose(mu, np.array([0.0], dtype=np.float32))


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

    eng = make_engine(X, m=1, seed=11)
    eng.draw_mu(0, residuals, sigma2=0.0, sigma_mu2=1.0)

    _, _, _, _, mu = eng.serialize_tree(0)
    expected = residuals.mean()

    np.testing.assert_allclose(mu, np.array([expected], dtype=np.float32), rtol=0, atol=1e-7)


def test_refresh_tree_training_predictions_single_tree():
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

    eng = make_engine(X, m=1, seed=11)
    eng.draw_mu(0, residuals, sigma2=0.0, sigma_mu2=1.0)

    training_predictions = np.zeros((1, X.shape[0]), dtype=float)
    fitted_sums = np.zeros(X.shape[0], dtype=float)

    eng.refresh_tree_training_predictions(0, training_predictions, fitted_sums)

    expected = np.full(X.shape[0], residuals.mean(), dtype=float)
    np.testing.assert_allclose(training_predictions[0], expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(fitted_sums, expected, rtol=0, atol=1e-7)


def test_refresh_only_updates_selected_tree():
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

    eng = make_engine(X, m=2, seed=11)
    eng.draw_mu(1, residuals, sigma2=0.0, sigma_mu2=1.0)

    training_predictions = np.zeros((2, X.shape[0]), dtype=float)
    training_predictions[0, :] = 5.0
    fitted_sums = np.full(X.shape[0], 5.0, dtype=float)

    eng.refresh_tree_training_predictions(1, training_predictions, fitted_sums)

    expected_mu = residuals.mean()
    np.testing.assert_allclose(training_predictions[0], np.full(X.shape[0], 5.0))
    np.testing.assert_allclose(training_predictions[1], np.full(X.shape[0], expected_mu))
    np.testing.assert_allclose(fitted_sums, np.full(X.shape[0], 5.0 + expected_mu))


def test_grow_accepts_deterministically_and_serializes_expected_tree():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=float)
    residuals = np.array([10.0, 10.0, -10.0, -10.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted = eng.draw_tree(
        0,
        residuals,
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=GROW_WITH_REVERSE,
    )

    assert accepted is True
    eng.validate_tree(0)

    variable, value, left, right, mu = eng.serialize_tree(0)

    np.testing.assert_array_equal(variable, np.array([0, -1, -1], dtype=np.int32))
    np.testing.assert_allclose(value, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(left, np.array([1, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(right, np.array([2, -1, -1], dtype=np.int32))
    np.testing.assert_allclose(mu, np.array([0.0, 0.0, 0.0], dtype=np.float32))


def test_prune_accepts_deterministically_after_grow():
    X = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted_grow = eng.draw_tree(
        0,
        np.array([10.0, 10.0, -10.0, -10.0], dtype=float),
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=GROW_WITH_REVERSE,
    )
    assert accepted_grow is True

    accepted_prune = eng.draw_tree(
        0,
        np.array([11.0, 11.0, 11.0, 11.0], dtype=float),
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=PRUNE_WITH_REVERSE,
    )

    assert accepted_prune is True
    eng.validate_tree(0)

    variable, value, left, right, mu = eng.serialize_tree(0)
    np.testing.assert_array_equal(variable, np.array([-1], dtype=np.int32))
    np.testing.assert_allclose(value, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(left, np.array([-1], dtype=np.int32))
    np.testing.assert_array_equal(right, np.array([-1], dtype=np.int32))
    np.testing.assert_allclose(mu, np.array([0.0], dtype=np.float32))


def test_grow_returns_false_when_no_valid_split_exists():
    X = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=float)
    residuals = np.array([1.0, -1.0, 2.0, -2.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted = eng.draw_tree(
        0,
        residuals,
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=GROW_WITH_REVERSE,
    )

    assert accepted is False
    eng.validate_tree(0)


def test_prune_returns_false_on_root_only_tree():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    residuals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted = eng.draw_tree(
        0,
        residuals,
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=PRUNE_WITH_REVERSE,
    )

    assert accepted is False
    eng.validate_tree(0)


def test_change_returns_false_on_root_only_tree():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    residuals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted = eng.draw_tree(
        0,
        residuals,
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=CHANGE_ONLY,
    )

    assert accepted is False
    eng.validate_tree(0)


def test_swap_returns_false_on_root_only_tree():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    residuals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    accepted = eng.draw_tree(
        0,
        residuals,
        sigma2=1.0,
        sigma_mu2=1.0,
        alpha=0.95,
        beta=2.0,
        move_distribution=SWAP_ONLY,
    )

    assert accepted is False
    eng.validate_tree(0)


def test_invalid_indices_and_shapes_raise():
    X = np.array(
        [
            [0.2, 1.0],
            [0.1, 3.0],
            [0.5, 2.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )

    eng = make_engine(X, m=2, seed=3)
    residuals = np.zeros(X.shape[0], dtype=float)

    with pytest.raises(RuntimeError):
        eng.validate_tree(2)

    with pytest.raises(RuntimeError):
        eng.serialize_tree(-1)

    with pytest.raises(RuntimeError):
        eng.draw_mu(0, np.zeros((X.shape[0], 1), dtype=float), sigma2=1.0, sigma_mu2=1.0)

    with pytest.raises(RuntimeError):
        eng.draw_tree(
            0,
            np.zeros((X.shape[0], 1), dtype=float),
            sigma2=1.0,
            sigma_mu2=1.0,
            alpha=0.95,
            beta=2.0,
            move_distribution=GROW_WITH_REVERSE,
        )

    with pytest.raises(RuntimeError):
        eng.refresh_tree_training_predictions(
            0,
            np.zeros((X.shape[0], 2), dtype=float),  # wrong shape
            np.zeros(X.shape[0], dtype=float),
        )

    with pytest.raises(RuntimeError):
        eng.refresh_tree_training_predictions(
            0,
            np.zeros((2, X.shape[0]), dtype=float),
            np.zeros((X.shape[0], 1), dtype=float),  # wrong ndim
        )


def test_invalid_move_distribution_raises():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    residuals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    eng = make_engine(X, m=1, seed=5)

    with pytest.raises(RuntimeError):
        eng.draw_tree(
            0,
            residuals,
            sigma2=1.0,
            sigma_mu2=1.0,
            alpha=0.95,
            beta=2.0,
            move_distribution=(0.25, 0.25, 0.25, 0.20),
        )

    with pytest.raises(RuntimeError):
        eng.draw_tree(
            0,
            residuals,
            sigma2=1.0,
            sigma_mu2=1.0,
            alpha=0.95,
            beta=2.0,
            move_distribution=(1.0, -1.0, 1.0, 0.0),
        )
