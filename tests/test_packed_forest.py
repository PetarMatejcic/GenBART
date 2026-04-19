import numpy as np
import pytest

from genbart._backend import PackedForest


def _pack_local_trees(local_trees):
    """
    local_trees: list of dicts with keys variable, value, left, right, mu
    Each tree uses local node indexing starting at 0.
    Returns flat packed arrays suitable for PackedForest.
    """
    node_counts = [len(t["variable"]) for t in local_trees]
    tree_offset = np.empty(len(local_trees) + 1, dtype=np.int64)
    tree_offset[0] = 0
    tree_offset[1:] = np.cumsum(node_counts)

    total_nodes = int(tree_offset[-1])

    variable = np.empty(total_nodes, dtype=np.int32)
    value = np.empty(total_nodes, dtype=np.float32)
    left = np.empty(total_nodes, dtype=np.int32)
    right = np.empty(total_nodes, dtype=np.int32)
    mu = np.empty(total_nodes, dtype=np.float32)

    for t_idx, t in enumerate(local_trees):
        base = int(tree_offset[t_idx])
        end = int(tree_offset[t_idx + 1])

        variable[base:end] = np.asarray(t["variable"], dtype=np.int32)
        value[base:end] = np.asarray(t["value"], dtype=np.float32)
        mu[base:end] = np.asarray(t["mu"], dtype=np.float32)

        local_left = np.asarray(t["left"], dtype=np.int32)
        local_right = np.asarray(t["right"], dtype=np.int32)

        left[base:end] = np.where(local_left >= 0, local_left + base, -1)
        right[base:end] = np.where(local_right >= 0, local_right + base, -1)

    return variable, value, left, right, mu, tree_offset


def _manual_predict_local_tree(x, tree):
    node = 0
    while tree["left"][node] != -1:
        if x[tree["variable"][node]] <= tree["value"][node]:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    return tree["mu"][node]


def _manual_draw_sums(X, draws):
    """
    draws: list of draws; each draw is a list of local-tree dicts
    Returns array of shape (n_draws, n_rows)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    out = np.zeros((len(draws), X.shape[0]), dtype=float)
    for d, draw in enumerate(draws):
        for i, x in enumerate(X):
            s = 0.0
            for tree in draw:
                s += _manual_predict_local_tree(x, tree)
            out[d, i] = s
    return out


def _example_draws():
    # Draw 0, tree 0:
    # if x0 <= 0.5 -> 1.0 else 2.0
    tree_a = {
        "variable": [0, -1, -1],
        "value": [0.5, 0.0, 0.0],
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "mu": [0.0, 1.0, 2.0],
    }

    # Draw 0, tree 1: constant 0.3
    tree_b = {
        "variable": [-1],
        "value": [0.0],
        "left": [-1],
        "right": [-1],
        "mu": [0.3],
    }

    # Draw 1, tree 0:
    # if x1 <= 1.5 -> -1.0 else 0.5
    tree_c = {
        "variable": [1, -1, -1],
        "value": [1.5, 0.0, 0.0],
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "mu": [0.0, -1.0, 0.5],
    }

    # Draw 1, tree 1: constant -0.2
    tree_d = {
        "variable": [-1],
        "value": [0.0],
        "left": [-1],
        "right": [-1],
        "mu": [-0.2],
    }

    draws = [
        [tree_a, tree_b],
        [tree_c, tree_d],
    ]
    return draws


def _build_example_forest():
    draws = _example_draws()
    local_trees = [tree for draw in draws for tree in draw]
    variable, value, left, right, mu, tree_offset = _pack_local_trees(local_trees)

    forest = PackedForest(
        variable=variable,
        value=value,
        left=left,
        right=right,
        mu=mu,
        tree_offset=tree_offset,
        n_draws=2,
        m=2,
        p=2,
    )
    return forest, draws


def test_draw_sums_row_matches_manual():
    forest, draws = _build_example_forest()

    x = np.array([0.25, 2.0], dtype=float)
    got = forest.draw_sums_row(x)
    expected = _manual_draw_sums(x, draws).ravel()

    assert got.shape == (2,)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-7)


def test_draw_sums_matrix_matches_manual():
    forest, draws = _build_example_forest()

    X = np.array([
        [0.25, 2.0],   # draw0: 1.0 + 0.3 = 1.3 ; draw1: 0.5 - 0.2 = 0.3
        [0.75, 1.0],   # draw0: 2.0 + 0.3 = 2.3 ; draw1: -1.0 - 0.2 = -1.2
        [0.10, 1.49],  # draw0: 1.0 + 0.3 = 1.3 ; draw1: -1.0 - 0.2 = -1.2
    ], dtype=float)

    got = forest.draw_sums_matrix(X)
    expected = _manual_draw_sums(X, draws)

    print("got     =", got)
    print("expected=", expected)
    print("diff    =", got - expected)

    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-7)


def test_row_and_matrix_are_consistent():
    forest, _ = _build_example_forest()

    X = np.array([
        [0.25, 2.0],
        [0.75, 1.0],
        [0.10, 1.49],
    ], dtype=float)

    mat = forest.draw_sums_matrix(X)
    for i in range(X.shape[0]):
        row = forest.draw_sums_row(X[i])
        np.testing.assert_allclose(row, mat[:, i], rtol=0.0, atol=1e-12)


def test_constructor_rejects_child_outside_tree_slice():
    draws = _example_draws()
    local_trees = [tree for draw in draws for tree in draw]
    variable, value, left, right, mu, tree_offset = _pack_local_trees(local_trees)

    # Corrupt the first tree: node 0 right child points outside its own slice
    right = right.copy()
    first_tree_begin = int(tree_offset[0])
    first_tree_end = int(tree_offset[1])
    right[first_tree_begin] = first_tree_end  # exactly one past the slice

    with pytest.raises(RuntimeError):
        PackedForest(
            variable=variable,
            value=value,
            left=left,
            right=right,
            mu=mu,
            tree_offset=tree_offset,
            n_draws=2,
            m=2,
            p=2,
        )


def test_constructor_rejects_leaf_with_right_child_only():
    draws = _example_draws()
    local_trees = [tree for draw in draws for tree in draw]
    variable, value, left, right, mu, tree_offset = _pack_local_trees(local_trees)

    # Corrupt a leaf so left == -1 but right != -1
    right = right.copy()
    leaf_idx = 1
    right[leaf_idx] = 2

    with pytest.raises(RuntimeError, match="leaf|right"):
        PackedForest(
            variable=variable,
            value=value,
            left=left,
            right=right,
            mu=mu,
            tree_offset=tree_offset,
            n_draws=2,
            m=2,
            p=2,
        )


def test_draw_sums_row_rejects_wrong_length():
    forest, _ = _build_example_forest()

    x = np.array([1.0], dtype=float)
    with pytest.raises(RuntimeError, match="wrong length"):
        forest.draw_sums_row(x)


def test_draw_sums_matrix_rejects_wrong_number_of_columns():
    forest, _ = _build_example_forest()

    X = np.array([[1.0], [2.0]], dtype=float)
    with pytest.raises(RuntimeError, match="wrong number of columns"):
        forest.draw_sums_matrix(X)

def test_regbart_predict_matches_packed_forest_backend():
    import numpy as np
    from genbart.reg_bart import RegBart

    rng = np.random.default_rng(123)

    X = np.linspace(-1.0, 1.0, 25).reshape(-1, 1)
    y = np.sin(2.5 * X[:, 0]) + 0.05 * rng.normal(size=X.shape[0])

    model = RegBart(
        m=8,
        n_burn=10,
        n_samples=15,
        random_state=123,
    )
    model.fit(X, y)

    assert model.packed_forest is not None

    X_test = np.array([[-0.75], [-0.10], [0.20], [0.90]], dtype=float)
    level = 0.90
    a = 1.0 - level

    draw_sums = model.packed_forest.draw_sums_matrix(X_test)

    expected_mean = model._inverse_transform_y(draw_sums.mean(axis=0))
    expected_median = model._inverse_transform_y(np.median(draw_sums, axis=0))
    q_low, q_high = np.quantile(draw_sums, [a / 2.0, 1.0 - a / 2.0], axis=0)
    expected_low = model._inverse_transform_y(q_low)
    expected_high = model._inverse_transform_y(q_high)

    pred_mean = model.predict(X_test, central_measure="mean", conf_int=True, level=level)
    pred_median = model.predict(X_test, central_measure="median", conf_int=True, level=level)

    np.testing.assert_allclose(pred_mean["prediction"], expected_mean, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pred_mean["conf_int_low"], expected_low, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pred_mean["conf_int_high"], expected_high, rtol=0.0, atol=1e-8)

    np.testing.assert_allclose(pred_median["prediction"], expected_median, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pred_median["conf_int_low"], expected_low, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(pred_median["conf_int_high"], expected_high, rtol=0.0, atol=1e-8)