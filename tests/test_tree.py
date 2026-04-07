from genbart.tree import Node, Tree
import numpy as np
import pytest


@pytest.fixture
def fixed_test_tree():
    node1 = Node.terminal(0.0, [0, 1])
    node2_1 = Node.terminal(0.0, [2])
    node2_2 = Node.terminal(0.0, [3])
    node2 = Node.internal(variable=0,
                            value=0.0,
                            left_node=node2_1,
                            right_node=node2_2,
                            rows=[2, 3])
    node3_1 = Node.internal(variable=0,
                            value=0.0,
                            left_node=node1,
                            right_node=node2,
                            rows=[0, 1, 2, 3])
    node3_2 = Node.terminal(0.0, [4, 5])
    root = Node.internal(variable=0,
                            value=0.0,
                            left_node=node3_1,
                            right_node=node3_2,
                            rows=[0, 1, 2, 3, 4, 5])
    return Tree(root, np.array([0]))


@pytest.fixture
def tiny_data():
    X = np.array([
        [-3.0, -2.0],
        [-2.0, -1.0],
        [-1.0,  0.0],
        [ 1.0,  1.0],
        [ 2.0,  2.0],
        [ 3.0,  2.0]
    ])
    y = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    return X, y


def test_node_starts_empty():
    node = Node()
    assert node.left is None
    assert node.right is None
    assert node.variable is None
    assert node.value is None
    assert node.mu is None
    assert node.rows is None


def test_init_terminal():
    node = Node.terminal(1.5, [0])

    assert node.mu == 1.5
    assert node.rows == [0]
    assert node.left is None
    assert node.right is None
    assert node.variable is None
    assert node.value is None


def test_init_internal():
    node = Node.internal(variable=1,
                         value=2.0,
                         left_node=Node(),
                         right_node=Node(),
                         rows = [0, 1])

    assert node.variable == 1
    assert node.value == 2.0
    assert node.left is not None
    assert node.right is not None
    assert node.rows == [0, 1]
    assert node.mu is None


def test_is_terminal_and_is_internal():
    node_t = Node.terminal(1.5, [0])
    node_i = Node.internal(variable=1,
                           value=2.0,
                           left_node=Node(),
                           right_node=Node(),
                           rows=[0, 1])

    assert node_t.is_terminal() is True
    assert node_t.is_internal() is False
    assert node_i.is_terminal() is False
    assert node_i.is_internal() is True


def test_terminal_paths(fixed_test_tree):
    t = fixed_test_tree

    terminal_paths = t.terminal_paths()

    assert len(terminal_paths) == 2


def test_internal_paths(fixed_test_tree):
    t = fixed_test_tree

    internal_paths = t.internal_paths()

    assert len(internal_paths) == 3


def test_prunable_paths(fixed_test_tree):
    t = fixed_test_tree

    prunable_paths = t.prunable_paths()

    assert len(prunable_paths) == 1


def test_swappable_paths(fixed_test_tree):
    t = fixed_test_tree

    swappable_paths = t.swappable_paths()

    assert len(swappable_paths) == 2


def test_terminal_nodes(fixed_test_tree):
    t = fixed_test_tree

    terminal_nodes = t.terminal_nodes()

    assert len(terminal_nodes) == 4
    assert terminal_nodes[0].mu == 0.0
    assert terminal_nodes[0].variable is None


def test_max_depth(fixed_test_tree):
    t = fixed_test_tree

    assert t.max_depth() == 3


def test_get_rows(fixed_test_tree):
    t = fixed_test_tree

    assert t.get_rows(()) == [0, 1, 2, 3, 4, 5]
    assert t.get_rows((0, )) == [0, 1, 2, 3]


def test_grow_tree(tiny_data):
    X, y = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)

    assert t.node_at(()).is_internal() is True
    assert t.node_at((0, )).is_internal() is True
    assert t.node_at((1, )).is_terminal() is True
    assert t.node_at((0, 0)).is_terminal() is True
    assert t.node_at((0, 1)).is_terminal() is True
    assert t.get_rows(()) == [0, 1, 2, 3, 4, 5]
    assert t.get_rows((0, )) == [0, 1, 2]


def test_prune_tree(tiny_data):
    X, _ = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)
    t = t.prune((0, ))

    assert t.node_at(()).is_internal() is True
    assert t.node_at((0, )).is_terminal() is True
    assert t.node_at((1, )).is_terminal() is True
    assert t.get_rows((0, )) == [0, 1, 2]
    assert t.get_rows((1, )) == [3, 4, 5]
    assert t.get_rows(()) == [0, 1, 2, 3, 4, 5]


def test_change_tree(tiny_data):
    X, _ = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)
    t = t.change((), 0, 1.0)

    assert t.root.variable == 0
    assert t.root.value == 1.0
    assert t.get_rows((0, )) == [0, 1, 2, 3]
    assert t.get_rows((1, )) == [4, 5]
    assert t.get_rows((0, 0)) == [0, 1]
    assert t.get_rows((0, 1)) == [2, 3]


def test_swap_tree(tiny_data):
    X, _ = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)
    t = t.swap((), "left")

    assert t.root.variable == 1
    assert t.root.value == -1.0
    assert t.root.left.variable == 0
    assert t.root.left.value == -1.0
    assert t.get_rows(()) == [0, 1, 2, 3, 4, 5]
    assert t.get_rows((0, )) == [0, 1]
    assert t.get_rows((1, )) == [2, 3, 4, 5]
    assert t.get_rows((0, 0)) == [0, 1]
    assert t.get_rows((0, 1)) == []


def test_predict(tiny_data):
    X, y = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)
    t = t.grow((0, 0), 1, -2.0)
    t = t.grow((1,), 1, 1.0)
    t = t.grow((1, 1), 0, 2.0)
    terminal_nodes = t.terminal_nodes()
    terminal_nodes[0].mu = -2.0
    terminal_nodes[1].mu = -1.0
    terminal_nodes[2].mu = -0.5
    terminal_nodes[3].mu = 0.5
    terminal_nodes[4].mu = 1.0
    terminal_nodes[5].mu = 2.0
    y_pred = t.predict(X)

    assert y_pred[0] == y[0]
    assert y_pred[1] == y[1]
    assert y_pred[2] == y[2]
    assert y_pred[3] == y[3]
    assert y_pred[4] == y[4]
    assert y_pred[5] == y[5]

def test_validate(tiny_data):
    X, y = tiny_data
    t = Tree(data=X)
    t = t.grow((), 0, -1.0)
    t = t.grow((0, ), 1, -1.0)

    t._validate()