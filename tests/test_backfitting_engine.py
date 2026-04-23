import numpy as np
import pytest

try:
    from genbart._backend import _BackfittingEngine
except ImportError:  # pragma: no cover
    from _backend import _BackfittingEngine


SIGMA2 = 1.0
SIGMA_MU2 = 1.0
ALPHA = 0.95
BETA = 2.0
DEFAULT_MOVES = (0.25, 0.25, 0.40, 0.10)


def _make_engine(X, m=1, seed=0):
    eng = _BackfittingEngine(np.asarray(X, dtype=float), m, seed)
    eng.initialize_root_forest()
    return eng


def _tree_arrays(engine, j=0):
    variable, value, left, right, mu = engine.serialize_tree(j)
    return (
        np.asarray(variable),
        np.asarray(value, dtype=float),
        np.asarray(left),
        np.asarray(right),
        np.asarray(mu, dtype=float),
    )


def _forest_arrays(engine):
    variable, value, left, right, mu, tree_offset = engine.serialize_forest()
    return (
        np.asarray(variable),
        np.asarray(value, dtype=float),
        np.asarray(left),
        np.asarray(right),
        np.asarray(mu, dtype=float),
        np.asarray(tree_offset),
    )


def _normalize_tree_from_forest_slice(variable, value, left, right, mu, begin, end):
    return (
        variable[begin:end],
        value[begin:end],
        np.where(left[begin:end] >= 0, left[begin:end] - begin, -1),
        np.where(right[begin:end] >= 0, right[begin:end] - begin, -1),
        mu[begin:end],
    )


def _assert_tree_serialization_valid(serialized, p):
    variable, value, left, right, mu = serialized
    n_nodes = len(variable)

    assert value.shape == (n_nodes,)
    assert left.shape == (n_nodes,)
    assert right.shape == (n_nodes,)
    assert mu.shape == (n_nodes,)
    assert n_nodes >= 1

    seen = set()
    stack = [0]

    while stack:
        idx = stack.pop()
        assert 0 <= idx < n_nodes
        assert idx not in seen, "cycle or duplicate serialized node"
        seen.add(idx)

        is_leaf = variable[idx] == -1
        if is_leaf:
            assert left[idx] == -1
            assert right[idx] == -1
        else:
            assert 0 <= variable[idx] < p
            assert left[idx] >= 0
            assert right[idx] >= 0
            stack.append(int(right[idx]))
            stack.append(int(left[idx]))

    assert seen == set(range(n_nodes)), "serialized tree contains unreachable nodes"


def _assert_forest_serialization_valid(serialized, m, p):
    variable, value, left, right, mu, tree_offset = serialized

    assert tree_offset.ndim == 1
    assert len(tree_offset) == m + 1
    assert tree_offset[0] == 0
    assert tree_offset[-1] == len(variable)
    assert np.all(tree_offset[1:] > tree_offset[:-1])

    for j in range(m):
        begin = int(tree_offset[j])
        end = int(tree_offset[j + 1])
        tree_j = _normalize_tree_from_forest_slice(variable, value, left, right, mu, begin, end)
        _assert_tree_serialization_valid(tree_j, p=p)


def _simple_X():
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )

def test_constructor_rejects_invalid_m_and_invalid_X_shapes():
    """Reject nonpositive ensemble sizes and malformed feature matrices."""
    with pytest.raises(RuntimeError, match="m must be positive"):
        _BackfittingEngine(np.zeros((3, 1), dtype=float), 0, 0)

    with pytest.raises(RuntimeError, match="X must be a 2D array"):
        _BackfittingEngine(np.array([1.0, 2.0, 3.0]), 1, 0)

    with pytest.raises(RuntimeError, match="X must have positive shape"):
        _BackfittingEngine(np.zeros((0, 2), dtype=float), 1, 0)

    with pytest.raises(RuntimeError, match="X must have positive shape"):
        _BackfittingEngine(np.zeros((2, 0), dtype=float), 1, 0)


def test_initialize_root_forest_creates_clean_single_node_trees():
    """Initialize a fresh forest and verify every tree is a single valid root leaf."""
    X = _simple_X()
    eng = _make_engine(X, m=3, seed=7)

    assert eng.n() == X.shape[0]
    assert eng.p() == X.shape[1]
    assert eng.m() == 3

    eng.validate_forest()

    for j in range(3):
        variable, value, left, right, mu = _tree_arrays(eng, j)
        np.testing.assert_array_equal(variable, np.array([-1], dtype=variable.dtype))
        np.testing.assert_array_equal(left, np.array([-1], dtype=left.dtype))
        np.testing.assert_array_equal(right, np.array([-1], dtype=right.dtype))
        np.testing.assert_allclose(value, np.array([0.0]))
        np.testing.assert_allclose(mu, np.array([0.0]))
        _assert_tree_serialization_valid((variable, value, left, right, mu), p=X.shape[1])


def test_initialize_root_forest_resets_mutated_forest_state():
    """Reset a mutated forest and verify all trees return to root-only zero-mean state."""
    X = _simple_X()
    residuals = np.linspace(-1.5, 1.5, X.shape[0]).astype(float)

    eng = _make_engine(X, m=2, seed=11)
    eng.draw_mu(0, residuals, SIGMA2, SIGMA_MU2)
    eng.backfitting_sweep(residuals.copy(), SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)
    eng.validate_forest()

    # The forest has now definitely mutated at least in mu values, and possibly structure.
    forest_before = _forest_arrays(eng)
    assert np.any(np.abs(forest_before[4]) > 0.0)

    eng.initialize_root_forest()
    eng.validate_forest()

    for j in range(2):
        variable, value, left, right, mu = _tree_arrays(eng, j)
        assert len(variable) == 1
        assert variable[0] == -1
        assert left[0] == -1
        assert right[0] == -1
        assert value[0] == 0.0
        assert mu[0] == 0.0


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("draw_tree", (-1, np.zeros(6), SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)),
        ("draw_mu", (-1, np.zeros(6), SIGMA2, SIGMA_MU2)),
        ("serialize_tree", (-1,)),
        ("validate_tree", (-1,)),
    ],
)
def test_invalid_tree_indices_raise(method_name, args):
    """Raise cleanly when a per-tree operation is requested on an out-of-bounds tree index."""
    eng = _make_engine(_simple_X(), m=1, seed=0)
    method = getattr(eng, method_name)
    with pytest.raises(RuntimeError, match="Tree index out of bounds"):
        method(*args)


def test_invalid_residual_shapes_raise():
    """Reject residual arrays with wrong dimension or wrong length in residual-sensitive methods."""
    X = _simple_X()
    eng = _make_engine(X, m=1, seed=0)

    bad_ndim = np.zeros((X.shape[0], 1), dtype=float)
    bad_len = np.zeros(X.shape[0] + 1, dtype=float)

    with pytest.raises(RuntimeError, match="residuals must be a 1D array"):
        eng.draw_tree(0, bad_ndim, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)

    with pytest.raises(RuntimeError, match="residuals has wrong length"):
        eng.draw_tree(0, bad_len, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)

    with pytest.raises(RuntimeError, match="residuals must be a 1D array"):
        eng.draw_mu(0, bad_ndim, SIGMA2, SIGMA_MU2)

    with pytest.raises(RuntimeError, match="residuals has wrong length"):
        eng.backfitting_sweep(bad_len, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)


def test_invalid_move_distribution_raises():
    """Reject move distributions with negative weights or weights that do not sum to one."""
    X = _simple_X()
    eng = _make_engine(X, m=1, seed=0)
    residuals = np.zeros(X.shape[0], dtype=float)

    with pytest.raises(RuntimeError, match="negative probabilities"):
        eng.draw_tree(0, residuals, SIGMA2, SIGMA_MU2, ALPHA, BETA, (-0.1, 0.5, 0.3, 0.3))

    with pytest.raises(RuntimeError, match="must sum to 1"):
        eng.draw_tree(0, residuals, SIGMA2, SIGMA_MU2, ALPHA, BETA, (0.2, 0.2, 0.2, 0.2))

    with pytest.raises(RuntimeError, match="must sum to 1"):
        eng.backfitting_sweep(residuals, SIGMA2, SIGMA_MU2, ALPHA, BETA, (1.0, 0.0, 0.0, 0.1))


def test_serialize_forest_matches_per_tree_serialization():
    """Verify packed forest serialization matches the concatenated per-tree serialized views."""
    X = _simple_X()
    residuals = np.linspace(-2.0, 2.0, X.shape[0]).astype(float)

    eng = _make_engine(X, m=3, seed=19)
    eng.backfitting_sweep(residuals.copy(), SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)
    eng.validate_forest()

    forest = _forest_arrays(eng)
    _assert_forest_serialization_valid(forest, m=3, p=X.shape[1])

    variable_all, value_all, left_all, right_all, mu_all, tree_offset = forest

    for j in range(3):
        begin = int(tree_offset[j])
        end = int(tree_offset[j + 1])
        from_forest = _normalize_tree_from_forest_slice(
            variable_all, value_all, left_all, right_all, mu_all, begin, end
        )
        direct = _tree_arrays(eng, j)

        for got, want in zip(from_forest, direct):
            np.testing.assert_array_equal(got, want)


def test_draw_mu_changes_only_leaf_means_and_not_topology():
    """Draw leaf means and verify tree structure is unchanged while mu values are updated."""
    X = _simple_X()
    residuals = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=float)

    eng = _make_engine(X, m=1, seed=23)
    before = _tree_arrays(eng, 0)
    residuals_before = residuals.copy()

    eng.draw_mu(0, residuals, SIGMA2, SIGMA_MU2)
    eng.validate_tree(0)
    after = _tree_arrays(eng, 0)

    np.testing.assert_array_equal(before[0], after[0])  # variable
    np.testing.assert_array_equal(before[1], after[1])  # value
    np.testing.assert_array_equal(before[2], after[2])  # left
    np.testing.assert_array_equal(before[3], after[3])  # right
    np.testing.assert_array_equal(residuals, residuals_before)  # draw_mu must not mutate residuals

    assert len(after[4]) == 1
    assert np.isfinite(after[4][0])
    assert after[4][0] != 0.0


def test_draw_tree_is_reproducible_with_fixed_seed():
    """Run one tree update on two identical engines and verify fixed-seed results are exactly reproducible."""
    X = _simple_X()
    residuals1 = np.linspace(-1.0, 1.0, X.shape[0]).astype(float)
    residuals2 = residuals1.copy()

    eng1 = _make_engine(X, m=1, seed=101)
    eng2 = _make_engine(X, m=1, seed=101)

    out1 = eng1.draw_tree(0, residuals1, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)
    out2 = eng2.draw_tree(0, residuals2, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)

    assert out1 == out2

    tree1 = _tree_arrays(eng1, 0)
    tree2 = _tree_arrays(eng2, 0)
    for a, b in zip(tree1, tree2):
        np.testing.assert_array_equal(a, b)

    eng1.validate_tree(0)
    eng2.validate_tree(0)


def test_backfitting_sweep_is_reproducible_with_fixed_seed():
    """Run identical sweeps from identical initial states and verify the forest and residuals match exactly."""
    X = _simple_X()
    residuals1 = np.linspace(-2.5, 2.5, X.shape[0]).astype(float)
    residuals2 = residuals1.copy()

    eng1 = _make_engine(X, m=3, seed=303)
    eng2 = _make_engine(X, m=3, seed=303)

    eng1.backfitting_sweep(residuals1, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)
    eng2.backfitting_sweep(residuals2, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)

    np.testing.assert_array_equal(residuals1, residuals2)

    forest1 = _forest_arrays(eng1)
    forest2 = _forest_arrays(eng2)
    for a, b in zip(forest1, forest2):
        np.testing.assert_array_equal(a, b)

    eng1.validate_forest()
    eng2.validate_forest()


def test_repeated_backfitting_sweeps_keep_forest_valid_and_residuals_finite():
    """Stress the engine with many sweeps and verify residuals stay finite and every tree remains valid."""
    X = _simple_X()
    residuals = np.linspace(-3.0, 3.0, X.shape[0]).astype(float)

    eng = _make_engine(X, m=4, seed=707)

    for _ in range(25):
        eng.backfitting_sweep(residuals, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)
        eng.validate_forest()

        forest = _forest_arrays(eng)
        _assert_forest_serialization_valid(forest, m=4, p=X.shape[1])

        assert np.all(np.isfinite(residuals))
        assert np.all(np.isfinite(forest[1]))  # value
        assert np.all(np.isfinite(forest[4]))  # mu


def test_draw_tree_returns_bool_and_preserves_validity():
    """Check that a stochastic tree update returns a boolean and leaves the tree in a valid state either way."""
    X = _simple_X()
    residuals = np.linspace(-1.0, 1.0, X.shape[0]).astype(float)

    eng = _make_engine(X, m=1, seed=909)
    out = eng.draw_tree(0, residuals, SIGMA2, SIGMA_MU2, ALPHA, BETA, DEFAULT_MOVES)

    assert isinstance(out, bool)
    eng.validate_tree(0)

    tree = _tree_arrays(eng, 0)
    _assert_tree_serialization_valid(tree, p=X.shape[1])
