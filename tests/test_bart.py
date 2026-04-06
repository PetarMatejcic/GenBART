import copy
import numpy as np
import pytest

from genbart.bart import bart
from genbart.tree import Node, Tree


class FixedRNG:
    """Tiny RNG stub for deterministic _draw_tree tests."""
    def __init__(self, move, uniform_value=0.5):
        self.move = move
        self.uniform_value = uniform_value

    def choice(self, options, p=None):
        return self.move

    def uniform(self):
        return self.uniform_value


@pytest.fixture
def tiny_data():
    X = np.array([
        [-2.0, -1.0],
        [-1.0,  0.0],
        [ 1.0,  1.0],
        [ 2.0,  2.0],
    ])
    y = np.array([-1.0, -0.5, 0.5, 1.0])
    return X, y


@pytest.fixture
def fitted_model(tiny_data):
    X, y = tiny_data
    model = bart(m=2, n_burn=0, n_samples=0, random_state=0)
    model.fit(X, y)
    return model


def install_prunable_root_tree(model, j=0, variable=0, value=0.0, mu_left=0.1, mu_right=-0.2):
    """Replace tree j with a one-split prunable tree and rebuild only the needed caches."""
    rows = list(range(model.n))
    model.trees[j] = model.trees[j].grow(
        path=(),
        variable=variable,
        value=value,
        mu_left=mu_left,
        mu_right=mu_right,
    )

    left_rows = [r for r in rows if model.X[r, variable] <= value]
    right_rows = [r for r in rows if model.X[r, variable] > value]

    model.internals_data_cache[j] = {(): rows}
    model.terminals_data_cache[j] = {
        (0,): left_rows,
        (1,): right_rows,
    }

    model._update_training_predictions()
    model._update_fitted_sums()
    return left_rows, right_rows


def test_fit_initializes_basic_state(fitted_model):
    model = fitted_model

    assert model.n == 4
    assert model.p == 2
    assert len(model.trees) == 2
    assert model.training_predicitons.shape == (4, 2)
    assert model.fitted_sums.shape == (4,)
    assert len(model.tree_sample) == 0

    assert np.isclose(model.y.min(), -0.5)
    assert np.isclose(model.y.max(), 0.5)

    for tree in model.trees:
        assert tree.root.is_terminal()
        assert tree.root.mu == 0.0

    assert model.terminals_data_cache[0] == {(): [0, 1, 2, 3]}
    assert model.terminals_data_cache[1] == {(): [0, 1, 2, 3]}
    assert model.internals_data_cache[0] == {}
    assert model.internals_data_cache[1] == {}


def test_partial_residuals_excludes_current_tree_prediction(fitted_model):
    model = fitted_model

    model.training_predicitons[:, 0] = np.array([1.0, 2.0, 3.0, 4.0])
    model.training_predicitons[:, 1] = np.array([10.0, 20.0, 30.0, 40.0])
    model._update_fitted_sums()

    residuals = model._partial_residuals(1)
    expected = model.y - model.training_predicitons[:, 0]

    np.testing.assert_allclose(residuals, expected)


def test_update_training_predictions_matches_tree_outputs(fitted_model):
    model = fitted_model

    model.trees[0] = model.trees[0].grow(
        path=(), variable=0, value=0.0, mu_left=-1.0, mu_right=1.0
    )
    model.trees[1] = Tree(Node.terminal(0.5))

    model._update_training_predictions()

    expected_col0 = np.array([-1.0, -1.0, 1.0, 1.0])
    expected_col1 = np.array([0.5, 0.5, 0.5, 0.5])

    np.testing.assert_allclose(model.training_predicitons[:, 0], expected_col0)
    np.testing.assert_allclose(model.training_predicitons[:, 1], expected_col1)


def test_update_fitted_sums_sums_prediction_columns(fitted_model):
    model = fitted_model

    model.training_predicitons[:, 0] = np.array([1.0, 2.0, 3.0, 4.0])
    model.training_predicitons[:, 1] = np.array([10.0, 20.0, 30.0, 40.0])

    model._update_fitted_sums()

    np.testing.assert_allclose(model.fitted_sums, np.array([11.0, 22.0, 33.0, 44.0]))


def test_inverse_transform_and_p_split(fitted_model):
    model = fitted_model

    vals = np.array([-0.5, 0.0, 0.5])
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(model._inverse_transform_y(vals), expected)

    d = 3
    expected_p_split = model.alpha / (1 + d) ** model.beta
    assert model._p_split(d) == pytest.approx(expected_p_split)


def test_propose_tree_grow_returns_valid_split_from_stump(fitted_model):
    model = fitted_model

    proposed_tree, mh_ratio, old_path = model._propose_tree_grow(0)

    assert proposed_tree is not None
    assert old_path == ()
    assert np.isfinite(mh_ratio)
    assert proposed_tree.root.is_internal()
    assert len(proposed_tree.terminal_paths()) == 2


def test_propose_tree_grow_returns_none_when_no_valid_split_exists():
    X = np.ones((4, 2))
    y = np.array([0.0, 1.0, 2.0, 3.0])

    model = bart(m=2, n_burn=0, n_samples=0, random_state=0)
    model.fit(X, y)

    proposed_tree, mh_ratio, old_path = model._propose_tree_grow(0)

    assert proposed_tree is None
    assert mh_ratio is None
    assert old_path is None


def test_draw_tree_accept_grow_updates_tree_and_caches(fitted_model, monkeypatch):
    model = fitted_model

    proposed_tree = model.trees[0].grow(
        path=(), variable=0, value=0.0, mu_left=0.0, mu_right=0.0
    )

    monkeypatch.setattr(
        model,
        "_propose_tree_grow",
        lambda j: (proposed_tree, 100.0, ()),
    )
    model.rng = FixedRNG(move="grow", uniform_value=0.5)

    model._draw_tree(0)

    assert model.trees[0].root.is_internal()
    assert model.internals_data_cache[0] == {(): [0, 1, 2, 3]}
    assert model.terminals_data_cache[0] == {
        (0,): [0, 1],
        (1,): [2, 3],
    }


def test_draw_tree_reject_grow_leaves_state_unchanged(fitted_model, monkeypatch):
    model = fitted_model

    old_tree = model.trees[0]
    old_terminal_cache = copy.deepcopy(model.terminals_data_cache[0])
    old_internal_cache = copy.deepcopy(model.internals_data_cache[0])

    proposed_tree = model.trees[0].grow(
        path=(), variable=0, value=0.0, mu_left=0.0, mu_right=0.0
    )

    monkeypatch.setattr(
        model,
        "_propose_tree_grow",
        lambda j: (proposed_tree, -100.0, ()),
    )
    model.rng = FixedRNG(move="grow", uniform_value=0.99)

    model._draw_tree(0)

    assert model.trees[0] is old_tree
    assert model.terminals_data_cache[0] == old_terminal_cache
    assert model.internals_data_cache[0] == old_internal_cache
    assert model.trees[0].root.is_terminal()


def test_propose_tree_prune_returns_terminal_tree(fitted_model):
    model = fitted_model
    install_prunable_root_tree(model, j=0)

    proposed_tree, mh_ratio, old_path = model._propose_tree_prune(0)

    assert proposed_tree is not None
    assert old_path == ()
    assert np.isfinite(mh_ratio)
    assert proposed_tree.root.is_terminal()


def test_draw_tree_accept_prune_updates_tree_and_caches(fitted_model, monkeypatch):
    model = fitted_model
    install_prunable_root_tree(model, j=0)

    proposed_tree = model.trees[0].prune((), mu=0.0)

    monkeypatch.setattr(
        model,
        "_propose_tree_prune",
        lambda j: (proposed_tree, 100.0, ()),
    )
    model.rng = FixedRNG(move="prune", uniform_value=0.5)

    model._draw_tree(0)

    assert model.trees[0].root.is_terminal()
    assert model.terminals_data_cache[0] == {(): [0, 1, 2, 3]}
    assert model.internals_data_cache[0] == {}


def test_draw_tree_reject_prune_leaves_state_unchanged(fitted_model, monkeypatch):
    model = fitted_model
    install_prunable_root_tree(model, j=0)

    old_tree = model.trees[0]
    old_terminal_cache = copy.deepcopy(model.terminals_data_cache[0])
    old_internal_cache = copy.deepcopy(model.internals_data_cache[0])

    proposed_tree = model.trees[0].prune((), mu=0.0)

    monkeypatch.setattr(
        model,
        "_propose_tree_prune",
        lambda j: (proposed_tree, -100.0, ()),
    )
    model.rng = FixedRNG(move="prune", uniform_value=0.99)

    model._draw_tree(0)

    assert model.trees[0] is old_tree
    assert model.terminals_data_cache[0] == old_terminal_cache
    assert model.internals_data_cache[0] == old_internal_cache
    assert model.trees[0].root.is_internal()


def test_draw_mu_updates_terminal_node_parameters(fitted_model):
    model = fitted_model
    install_prunable_root_tree(model, j=0, mu_left=0.0, mu_right=0.0)

    old_left_mu = model.trees[0].node_at((0,)).mu
    old_right_mu = model.trees[0].node_at((1,)).mu

    model._draw_mu(0)

    new_left_mu = model.trees[0].node_at((0,)).mu
    new_right_mu = model.trees[0].node_at((1,)).mu

    assert model.trees[0].root.mu is None
    assert np.isfinite(new_left_mu)
    assert np.isfinite(new_right_mu)
    assert (new_left_mu != old_left_mu) or (new_right_mu != old_right_mu)


def test_draw_sigma_returns_positive_finite_value(fitted_model):
    model = fitted_model
    sigma = model._draw_sigma()

    assert np.isfinite(sigma)
    assert sigma > 0.0


def test_predict_averages_saved_tree_samples_for_vector_and_matrix_inputs(fitted_model):
    model = fitted_model

    split_tree_1 = Tree(Node.terminal(0.0)).grow(
        path=(), variable=0, value=0.0, mu_left=-0.5, mu_right=0.5
    )
    split_tree_2 = Tree(Node.terminal(0.0)).grow(
        path=(), variable=0, value=0.0, mu_left=-0.25, mu_right=0.25
    )

    stump0 = Tree(Node.terminal(0.0))

    model.n_samples = 2
    model.tree_sample = [
        {"sample": [split_tree_1, stump0], "sigma": 1.0},
        {"sample": [split_tree_2, stump0], "sigma": 1.0},
    ]

    x_left = np.array([-2.0, 0.0])
    X_both = np.array([
        [-2.0, 0.0],
        [ 2.0, 0.0],
    ])

    # Internal-scale means: left = (-0.5 + -0.25)/2 = -0.375, right = (0.5 + 0.25)/2 = 0.375
    # y_scale = 2 and y_shift = 0 for the fixture data, so inverse transform gives ±0.75
    assert model.predict(x_left) == pytest.approx(-0.75)
    np.testing.assert_allclose(model.predict(X_both), np.array([-0.75, 0.75]))

def test_fit_smoke_small_chain():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])

    model = bart(m=5, n_burn=2, n_samples=2, random_state=0)
    model.fit(X, y)

    assert len(model.tree_sample) == 2