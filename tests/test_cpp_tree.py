import numpy as np
import pytest

from genbart.tree import Tree as PyTree
from genbart._backend import _Tree as CppTree


def make_py_tree(X: np.ndarray) -> PyTree:
    rows_by_var = [np.argsort(X[:, j], kind="mergesort") for j in range(X.shape[1])]
    return PyTree(data=X, rows_by_var=rows_by_var)


def serialize_py(tree: PyTree):
    st = tree.serialize()
    return (
        np.asarray(st.variable),
        np.asarray(st.value),
        np.asarray(st.left),
        np.asarray(st.right),
        np.asarray(st.mu),
    )


def serialize_cpp(tree: CppTree):
    ser = tree.serialize()
    return tuple(np.asarray(x) for x in ser)


def assert_serialized_equal(py_tree: PyTree, cpp_tree: CppTree):
    py_ser = serialize_py(py_tree)
    cpp_ser = serialize_cpp(cpp_tree)
    for a, b in zip(py_ser, cpp_ser):
        np.testing.assert_array_equal(a, b)


def assert_terminal_rows_equal(py_rows, cpp_rows):
    assert len(py_rows) == len(cpp_rows)
    for a, b in zip(py_rows, cpp_rows):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def single_growable_leaf(cpp_tree: CppTree) -> int:
    leaves = cpp_tree.terminal_nodes(True)
    assert len(leaves) == 1, f"Expected exactly one growable leaf, got {leaves}"
    return int(leaves[0])


def test_cpp_root_matches_python():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    assert_serialized_equal(py_tree, cpp_tree)
    cpp_tree.validate()

    assert cpp_tree.root() == 0
    assert cpp_tree.internal_nodes() == []
    assert cpp_tree.prunable_nodes() == []
    assert cpp_tree.swappable_nodes() == []
    assert len(cpp_tree.terminal_nodes(True)) == 1


def test_propose_grow_root_matches_python_and_is_nonmutating():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_before = serialize_py(py_tree)
    cpp_before = serialize_cpp(cpp_tree)

    py_prop = py_tree.grow((), variable=0, split_idx=0)
    cpp_prop = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)

    assert cpp_prop is not None

    # Live trees unchanged
    for a, b in zip(py_before, serialize_py(py_tree)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(cpp_before, serialize_cpp(cpp_tree)):
        np.testing.assert_array_equal(a, b)

    # Proposal subtree matches
    py_prop_ser = serialize_py(py_prop)
    cpp_prop_ser = serialize_cpp(cpp_prop.subtree)
    for a, b in zip(py_prop_ser, cpp_prop_ser):
        np.testing.assert_array_equal(a, b)

    # Proposal terminal row sets match
    py_rows = [py_prop.get_rows((0,)), py_prop.get_rows((1,))]
    assert_terminal_rows_equal(py_rows, cpp_prop.terminals)

    # Proposal subtree internals: just root
    assert cpp_prop.internals == [0]


def test_replace_subtree_after_grow_matches_python():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_prop = py_tree.grow((), variable=0, split_idx=0)
    cpp_prop = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_prop is not None

    py_tree.replace_subtree((), py_prop)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_prop.subtree)

    assert_serialized_equal(py_tree, cpp_tree)
    cpp_tree.validate()

    assert len(cpp_tree.internal_nodes()) == 1
    assert len(cpp_tree.prunable_nodes()) == 1
    assert len(cpp_tree.swappable_nodes()) == 0


def test_propose_prune_root_matches_python_and_is_nonmutating():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_grow = py_tree.grow((), variable=0, split_idx=0)
    cpp_grow = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_grow is not None

    py_tree.replace_subtree((), py_grow)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_grow.subtree)

    py_before = serialize_py(py_tree)
    cpp_before = serialize_cpp(cpp_tree)

    py_prop = py_tree.prune(())
    cpp_prop = cpp_tree.propose_prune(cpp_tree.root(), mu=0.0)

    assert cpp_prop is not None

    # Live trees unchanged
    for a, b in zip(py_before, serialize_py(py_tree)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(cpp_before, serialize_cpp(cpp_tree)):
        np.testing.assert_array_equal(a, b)

    py_prop_ser = serialize_py(py_prop)
    cpp_prop_ser = serialize_cpp(cpp_prop.subtree)
    for a, b in zip(py_prop_ser, cpp_prop_ser):
        np.testing.assert_array_equal(a, b)

    assert_terminal_rows_equal([py_prop.root.rows], cpp_prop.terminals)
    assert cpp_prop.internals == []


def test_replace_subtree_after_prune_matches_python():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_grow = py_tree.grow((), variable=0, split_idx=0)
    cpp_grow = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_grow is not None

    py_tree.replace_subtree((), py_grow)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_grow.subtree)

    py_prune = py_tree.prune(())
    cpp_prune = cpp_tree.propose_prune(cpp_tree.root(), mu=0.0)
    assert cpp_prune is not None

    py_tree.replace_subtree((), py_prune)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_prune.subtree)

    assert_serialized_equal(py_tree, cpp_tree)
    cpp_tree.validate()

    assert cpp_tree.internal_nodes() == []
    assert cpp_tree.prunable_nodes() == []
    assert cpp_tree.swappable_nodes() == []


def test_propose_change_root_matches_python_and_is_nonmutating():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_grow = py_tree.grow((), variable=0, split_idx=0)
    cpp_grow = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_grow is not None

    py_tree.replace_subtree((), py_grow)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_grow.subtree)

    py_before = serialize_py(py_tree)
    cpp_before = serialize_cpp(cpp_tree)

    py_prop, py_terms, py_internals = py_tree.change((), variable=1, split_idx=0)
    cpp_prop = cpp_tree.propose_change(cpp_tree.root(), variable=1, split_idx=0)

    assert cpp_prop is not None

    # Live trees unchanged
    for a, b in zip(py_before, serialize_py(py_tree)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(cpp_before, serialize_cpp(cpp_tree)):
        np.testing.assert_array_equal(a, b)

    py_prop_ser = serialize_py(py_prop)
    cpp_prop_ser = serialize_cpp(cpp_prop.subtree)
    for a, b in zip(py_prop_ser, cpp_prop_ser):
        np.testing.assert_array_equal(a, b)

    assert_terminal_rows_equal(py_terms, cpp_prop.terminals)
    assert len(py_internals) == len(cpp_prop.internals)


def test_replace_subtree_after_change_matches_python():
    X = np.array(
        [
            [0.1, 2.0],
            [0.4, 1.0],
            [0.4, 3.0],
            [0.9, 0.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    py_grow = py_tree.grow((), variable=0, split_idx=0)
    cpp_grow = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_grow is not None

    py_tree.replace_subtree((), py_grow)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_grow.subtree)

    py_change, _, _ = py_tree.change((), variable=1, split_idx=0)
    cpp_change = cpp_tree.propose_change(cpp_tree.root(), variable=1, split_idx=0)
    assert cpp_change is not None

    py_tree.replace_subtree((), py_change)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_change.subtree)

    assert_serialized_equal(py_tree, cpp_tree)
    cpp_tree.validate()


def test_propose_swap_root_right_matches_python():
    # Designed so root split on x0 at split_idx=0 leaves only the RIGHT child growable.
    X = np.array(
        [
            [0.1, 0.0],
            [0.9, 0.0],
            [1.0, 1.0],
            [1.1, 2.0],
        ],
        dtype=float,
    )

    py_tree = make_py_tree(X)
    cpp_tree = CppTree(X)

    # Grow root on x0. Left side has one row, right side has three rows.
    py_root_prop = py_tree.grow((), variable=0, split_idx=0)
    cpp_root_prop = cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)
    assert cpp_root_prop is not None

    py_tree.replace_subtree((), py_root_prop)
    cpp_tree.replace_subtree(cpp_tree.root(), cpp_root_prop.subtree)

    # Only one growable leaf remains; in Python it is path (1,), in C++ we get its live node idx.
    growable_idx = single_growable_leaf(cpp_tree)

    py_child_prop = py_tree.grow((1,), variable=1, split_idx=0)
    cpp_child_prop = cpp_tree.propose_grow(growable_idx, variable=1, split_idx=0)
    assert cpp_child_prop is not None

    py_tree.replace_subtree((1,), py_child_prop)
    cpp_tree.replace_subtree(growable_idx, cpp_child_prop.subtree)

    assert cpp_tree.swappable_nodes() == [cpp_tree.root()]

    py_swap_prop, py_terms, py_internals = py_tree.swap((), swap="right")
    cpp_swap_prop = cpp_tree.propose_swap(cpp_tree.root(), mode=1)

    assert py_swap_prop is None
    assert cpp_swap_prop is None


def test_invalid_grow_raises():
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )

    cpp_tree = CppTree(X)

    with pytest.raises(Exception):
        cpp_tree.propose_grow(cpp_tree.root(), variable=0, split_idx=0)