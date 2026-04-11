import numpy as np

from genbart.bart import bart
from genbart.tree import Tree


def _assert_partition(tree: Tree, path: tuple = ()):
    node = tree.node_at(path)
    if node.is_terminal():
        return

    left = tree.node_at(path + (0,))
    right = tree.node_at(path + (1,))

    expected_left = np.array([r for r in node.rows
                     if tree.data[r, node.variable] <= node.value])
    expected_right = np.array([r for r in node.rows
                      if tree.data[r, node.variable] > node.value])

    assert np.array_equal(left.rows, expected_left)
    assert np.array_equal(right.rows, expected_right)
    assert np.array_equal(np.sort(np.concatenate((left.rows, right.rows))),
                          np.sort(node.rows))

    _assert_partition(tree, path + (0,))
    _assert_partition(tree, path + (1,))


def test_tree_grow_change_and_swap_keep_row_bookkeeping_consistent():
    X = np.array(
        [
            [0.10, 0.10],
            [0.20, 0.80],
            [0.40, 0.30],
            [0.60, 0.20],
            [0.80, 0.70],
            [0.90, 0.90],
        ]
    )

    tree = Tree(data=X)
    tree.replace_subtree((), tree.grow((), variable=0, value=0.50))
    tree.replace_subtree((), tree.change((), variable=1, value=0.50)[0])
    _assert_partition(tree)

    tree.replace_subtree((0, ), tree.grow((0,), variable=0, value=0.15))
    tree.replace_subtree((1, ), tree.grow((1,), variable=0, value=0.80))
    _assert_partition(tree)

    swapped = tree.replace_subtree((), tree.swap((), swap="left"))
    _assert_partition(swapped)
    swapped._validate()


def test_incremental_prediction_bookkeeping_matches_tree_predictions():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])

    model = bart(m=2, n_burn=0, n_samples=0, random_state=0)
    model.X = X
    model.y = np.zeros(X.shape[0])
    model.n, model.p = X.shape
    model.sigma2 = 1.0
    model.sigma_mu2 = 1.0
    model.trees = [Tree(data=X) for _ in range(model.m)]
    model.training_predictions = np.zeros((model.n, model.m))
    model.fitted_sums = np.zeros(model.n)

    model.trees[0].root.mu = 1.25
    model._update_tps_and_fitted_sums_incremental(0)
    assert np.allclose(model.training_predictions[:, 0], 1.25)
    assert np.allclose(model.fitted_sums, 1.25)

    model.trees[1] = model.trees[1].grow((), variable=0, value=1.5)
    model.trees[1].node_at((0,)).mu = -0.5
    model.trees[1].node_at((1,)).mu = 0.75
    model._update_tps_and_fitted_sums_incremental(1)

    expected_tp = np.column_stack([
        model.trees[0].predict(X),
        model.trees[1].predict(X),
    ])
    assert np.allclose(model.training_predictions, expected_tp)
    assert np.allclose(model.fitted_sums, expected_tp.sum(axis=1))

    for j in range(model.m):
        expected_partial = (model.y
                            - model.fitted_sums
                            + model.training_predictions[:, j])
        assert np.allclose(model._partial_residuals(j), expected_partial)


def test_change_proposal_runs_without_type_errors_on_internal_tree():
    X = np.array(
        [
            [0.10, 0.10],
            [0.20, 0.80],
            [0.40, 0.30],
            [0.60, 0.20],
            [0.80, 0.70],
            [0.90, 0.90],
        ]
    )
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

    model = bart(m=1, n_burn=0, n_samples=0, random_state=2)
    model.fit(X, y)
    model.trees[0] = model.trees[0].grow((), variable=0, value=0.40)
    model.trees[0].node_at((0,)).mu = -0.5
    model.trees[0].node_at((1,)).mu = 0.5
    model._update_tps_and_fitted_sums_incremental(0)

    proposal, mh_ratio, path = model._propose_tree_change(0)

    assert path == ()
    if proposal is not None:
        proposal._validate()
        assert np.isfinite(mh_ratio)


def test_bart_gives_reasonable_predictions_on_simple_step_data():
    x = np.linspace(0.0, 1.0, 40)
    y = np.where(x <= 0.5, -1.0, 1.0)

    model = bart(m=20, n_burn=60, n_samples=120, random_state=123)
    model.fit(x, y)

    train_pred = model.predict(x)[0]
    rmse = np.sqrt(np.mean((train_pred - y) ** 2))

    left_pred = model.predict(np.array([0.20]))[0]
    right_pred = model.predict(np.array([0.80]))[0]

    assert rmse < 0.35
    assert left_pred < -0.30
    assert right_pred > 0.30
    assert right_pred - left_pred > 0.80
