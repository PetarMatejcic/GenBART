import numpy as np
import pytest

try:
    from genbart._backend import _Tree as CppTree
except ImportError:  # pragma: no cover
    from _backend import _Tree as CppTree


def _as_arrays(serialized):
    variable, value, left, right, mu = serialized
    return (
        np.asarray(variable),
        np.asarray(value, dtype=float),
        np.asarray(left),
        np.asarray(right),
        np.asarray(mu, dtype=float),
    )


def _serialize(tree):
    return _as_arrays(tree.serialize())


def _count_prunable_from_serialized(left, right):
    n = len(left)
    out = 0
    for i in range(n):
        if left[i] >= 0 and right[i] >= 0:
            if left[left[i]] == -1 and right[left[i]] == -1 and left[right[i]] == -1 and right[right[i]] == -1:
                out += 1
    return out


def _count_swappable_from_serialized(left, right):
    n = len(left)
    out = 0
    for i in range(n):
        if left[i] >= 0 and right[i] >= 0:
            left_internal = left[left[i]] >= 0
            right_internal = left[right[i]] >= 0
            if left_internal or right_internal:
                out += 1
    return out


def _assert_serialization_consistent(tree):
    variable, value, left, right, mu = _serialize(tree)
    n_nodes = len(variable)

    assert value.shape == (n_nodes,)
    assert left.shape == (n_nodes,)
    assert right.shape == (n_nodes,)
    assert mu.shape == (n_nodes,)

    seen = set()
    stack = [0]
    while stack:
        idx = stack.pop()
        assert 0 <= idx < n_nodes
        assert idx not in seen, "cycle or duplicate node in serialized tree"
        seen.add(idx)

        is_leaf = variable[idx] == -1
        if is_leaf:
            assert left[idx] == -1
            assert right[idx] == -1
        else:
            assert left[idx] >= 0
            assert right[idx] >= 0
            stack.append(int(right[idx]))
            stack.append(int(left[idx]))

    assert seen == set(range(n_nodes)), "serialized tree contains unreachable nodes"

    assert len(tree.internal_nodes()) == int(np.sum(variable >= 0))
    assert len(tree.terminal_nodes(False)) == int(np.sum(variable == -1))
    assert len(tree.prunable_nodes()) == _count_prunable_from_serialized(left, right)
    assert len(tree.swappable_nodes()) == _count_swappable_from_serialized(left, right)


def _leaf_index_for_row(serialized, row):
    variable, value, left, right, mu = _as_arrays(serialized)
    idx = 0
    while variable[idx] != -1:
        split_var = int(variable[idx])
        split_val = float(value[idx])
        idx = int(left[idx]) if row[split_var] <= split_val else int(right[idx])
    return idx


def _leaf_counts_for_matrix(tree, X):
    serialized = tree.serialize()
    counts = {}
    for row in np.asarray(X, dtype=float):
        leaf = _leaf_index_for_row(serialized, row)
        counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def _base_grid():
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


def _constant_grid():
    return np.ones((4, 2), dtype=float)


def _duplicates_1d():
    return np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], dtype=float)


def _make_root_grown_tree():
    tree = CppTree(_base_grid())
    assert tree.test_grow(tree.root(), 0, 0) is True
    tree.validate()
    return tree


def _make_changeable_tree():
    tree = CppTree(_base_grid())
    assert tree.test_grow(tree.root(), 0, 0) is True
    left_child = _serialize(tree)[2][0]
    assert tree.test_grow(int(left_child), 1, 0) is True
    tree.validate()
    return tree


def _make_left_swappable_tree():
    tree = CppTree(_base_grid())
    assert tree.test_grow(tree.root(), 0, 1) is True
    left_child = _serialize(tree)[2][0]
    assert tree.test_grow(int(left_child), 1, 0) is True
    tree.validate()
    return tree


def _find_successful_swap_mode(builder):
    successes = []
    for mode in (0, 1, 2):
        tree = builder()
        try:
            ok = tree.test_swap(tree.root(), mode)
        except RuntimeError:
            ok = False
        if ok:
            tree.validate()
            successes.append((mode, tree))
    return successes


def test_constructor_rejects_non_2d_or_empty_inputs():
    with pytest.raises(RuntimeError):
        CppTree(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(RuntimeError):
        CppTree(np.zeros((0, 2), dtype=float))

    with pytest.raises(RuntimeError):
        CppTree(np.zeros((2, 0), dtype=float))

    with pytest.raises(RuntimeError):
        CppTree(np.zeros((2, 2, 2), dtype=float))


def test_initial_root_state_and_serialization_are_correct():
    tree = CppTree(_base_grid())
    tree.validate()

    assert tree.root() == 0
    assert tree.internal_nodes() == []
    assert tree.prunable_nodes() == []
    assert tree.swappable_nodes() == []
    assert tree.terminal_nodes(False) == [0]
    assert tree.terminal_nodes(True) == [0]

    variable, value, left, right, mu = _serialize(tree)
    np.testing.assert_array_equal(variable, np.array([-1], dtype=variable.dtype))
    np.testing.assert_array_equal(left, np.array([-1], dtype=left.dtype))
    np.testing.assert_array_equal(right, np.array([-1], dtype=right.dtype))
    np.testing.assert_allclose(value, np.array([0.0]))
    np.testing.assert_allclose(mu, np.array([0.0]))

    _assert_serialization_consistent(tree)


def test_constant_data_has_no_growable_terminal_nodes():
    tree = CppTree(_constant_grid())
    tree.validate()

    assert tree.terminal_nodes(False) == [0]
    assert tree.terminal_nodes(True) == []

    with pytest.raises(RuntimeError, match="Split is not valid"):
        tree.test_grow(tree.root(), 0, 0)

    _assert_serialization_consistent(tree)


def test_grow_root_creates_one_internal_and_two_terminal_children():
    X = _base_grid()
    tree = CppTree(X)

    assert tree.test_grow(tree.root(), 0, 0) is True
    tree.validate()

    assert len(tree.internal_nodes()) == 1
    assert len(tree.terminal_nodes(False)) == 2
    assert tree.prunable_nodes() == [tree.root()]
    assert tree.swappable_nodes() == []

    variable, value, left, right, mu = _serialize(tree)
    assert variable[0] == 0
    assert value[0] == 0.0
    assert left[0] == 1
    assert right[0] == 2
    assert variable[1] == -1
    assert variable[2] == -1
    np.testing.assert_allclose(mu, np.zeros_like(mu))

    counts = _leaf_counts_for_matrix(tree, X)
    assert sorted(counts.values()) == [2, 4]

    _assert_serialization_consistent(tree)


def test_grow_on_internal_node_raises():
    tree = _make_root_grown_tree()
    with pytest.raises(RuntimeError, match="Cannot grow an internal node"):
        tree.test_grow(tree.root(), 1, 0)


def test_prune_restores_single_leaf_and_sets_leaf_mu():
    tree = _make_root_grown_tree()

    assert tree.test_prune(tree.root(), 2.5) is True
    tree.validate()

    assert tree.internal_nodes() == []
    assert tree.prunable_nodes() == []
    assert tree.swappable_nodes() == []
    assert tree.terminal_nodes(False) == [tree.root()]
    assert tree.terminal_nodes(True) == [tree.root()]

    variable, value, left, right, mu = _serialize(tree)
    np.testing.assert_array_equal(variable, np.array([-1], dtype=variable.dtype))
    np.testing.assert_array_equal(left, np.array([-1], dtype=left.dtype))
    np.testing.assert_array_equal(right, np.array([-1], dtype=right.dtype))
    np.testing.assert_allclose(mu, np.array([2.5]))

    _assert_serialization_consistent(tree)


def test_prune_terminal_node_raises():
    tree = CppTree(_base_grid())
    with pytest.raises(RuntimeError, match="Cannot prune a terminal node"):
        tree.test_prune(tree.root(), 0.0)


def test_change_updates_rule_but_preserves_same_shape_topology():
    tree = _make_changeable_tree()
    before = _serialize(tree)

    assert tree.test_change(tree.root(), 0, 1) is True
    tree.validate()

    after = _serialize(tree)

    assert len(before[0]) == len(after[0]) == 5
    assert np.sum(before[0] >= 0) == np.sum(after[0] >= 0) == 2
    assert before[0][0] == after[0][0] == 0
    assert before[1][0] == 0.0
    assert after[1][0] == 1.0

    _assert_serialization_consistent(tree)


def test_change_returns_false_when_descendant_shape_becomes_invalid():
    tree = _make_changeable_tree()
    before = _serialize(tree)

    assert tree.test_change(tree.root(), 1, 0) is False
    tree.validate()

    after = _serialize(tree)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)

    _assert_serialization_consistent(tree)


def test_change_on_terminal_node_raises():
    tree = CppTree(_base_grid())
    with pytest.raises(RuntimeError, match="Cannot change a terminal node"):
        tree.test_change(tree.root(), 0, 0)


def test_swap_succeeds_for_exactly_one_mode_when_only_left_child_is_internal():
    successes = _find_successful_swap_mode(_make_left_swappable_tree)
    assert len(successes) == 1, "expected exactly one valid swap mode"

    _, tree = successes[0]
    variable, value, left, right, mu = _serialize(tree)

    assert variable[0] == 1  # root now uses the former left-child rule
    assert variable[int(left[0])] == 0  # left child now uses the former root rule

    _assert_serialization_consistent(tree)


def test_swap_on_node_with_two_terminal_children_raises_before_mode_matters():
    tree = _make_root_grown_tree()
    with pytest.raises(RuntimeError, match="Cannot swap at a node with two terminal children"):
        tree.test_swap(tree.root(), 0)


def test_unknown_swap_mode_raises():
    tree = _make_left_swappable_tree()
    with pytest.raises(RuntimeError, match="Unknown swap mode"):
        tree.test_swap(tree.root(), 99)


def test_duplicate_feature_values_produce_expected_threshold_and_partition():
    X = _duplicates_1d()
    tree = CppTree(X)

    assert tree.test_grow(tree.root(), 0, 0) is True
    tree.validate()

    variable, value, left, right, mu = _serialize(tree)
    assert variable[0] == 0
    assert value[0] == 0.0

    counts = _leaf_counts_for_matrix(tree, X)
    assert sorted(counts.values()) == [2, 4]

    _assert_serialization_consistent(tree)


def test_repeated_grow_prune_cycles_keep_structure_valid():
    tree = CppTree(_base_grid())

    for _ in range(10):
        assert tree.test_grow(tree.root(), 0, 0) is True
        tree.validate()
        _assert_serialization_consistent(tree)

        assert tree.test_prune(tree.root(), 0.0) is True
        tree.validate()
        _assert_serialization_consistent(tree)

    variable, value, left, right, mu = _serialize(tree)
    np.testing.assert_array_equal(variable, np.array([-1], dtype=variable.dtype))
    np.testing.assert_array_equal(left, np.array([-1], dtype=left.dtype))
    np.testing.assert_array_equal(right, np.array([-1], dtype=right.dtype))


def test_complex_operation_sequence_stays_valid_end_to_end():
    tree = _make_left_swappable_tree()
    _assert_serialization_consistent(tree)

    successes = _find_successful_swap_mode(_make_left_swappable_tree)
    assert len(successes) == 1
    _, tree = successes[0]
    tree.validate()
    _assert_serialization_consistent(tree)

    assert tree.test_change(tree.root(), 1, 0) is True
    tree.validate()
    _assert_serialization_consistent(tree)

    variable, value, left, right, mu = _serialize(tree)
    assert len(variable) == 5
    assert np.sum(variable >= 0) == 2
