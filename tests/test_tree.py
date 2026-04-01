from genbart.tree import Node, Tree
import numpy as np


def test_node_starts_empty():
    node = Node()
    assert node.left is None
    assert node.right is None
    assert node.variable is None
    assert node.value is None
    assert node.mu is None


def test_init_terminal():
    node = Node.terminal(1.5)

    assert node.mu == 1.5
    assert node.left is None
    assert node.right is None
    assert node.variable is None
    assert node.value is None


def test_init_internal():
    node = Node.internal(variable=1,
                         value=2.0,
                         left_node=Node(),
                         right_node=Node())

    assert node.mu is None
    assert node.left is not None
    assert node.right is not None
    assert node.variable is not None
    assert node.value is not None


def test_is_terminal_and_is_internal():
    node_t = Node.terminal(1.0)

    node_i = Node.internal(variable=1,
                           value=2.0,
                           left_node=Node(),
                           right_node=Node())

    assert node_t.is_terminal() is True
    assert node_t.is_internal() is False
    assert node_i.is_terminal() is False
    assert node_i.is_internal() is True


def test_terminal_paths():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    terminal_paths = t.terminal_paths()

    assert len(terminal_paths) == 4


def test_internal_paths():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    internal_paths = t.internal_paths()

    assert len(internal_paths) == 3


def test_prunable_paths():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    prunable_paths = t.prunable_paths()

    assert len(prunable_paths) == 1


def test_swappable_paths():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    swappable_paths = t.swappable_paths()

    assert len(swappable_paths) == 2


def test_terminal_nodes():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    terminal_nodes = t.terminal_nodes()

    assert len(terminal_nodes) == 4
    assert terminal_nodes[0].mu == 0.0
    assert terminal_nodes[0].variable is None


def test_grow_tree():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))

    assert t.node_at(()).is_internal() is True
    assert t.node_at((0, )).is_internal() is True
    assert t.node_at((1, )).is_terminal() is True
    assert t.node_at((0, 0)).is_terminal() is True
    assert t.node_at((0, 1)).is_internal() is True
    assert t.node_at((0, 1, 0)).is_terminal() is True
    assert t.node_at((0, 1, 1)).is_terminal() is True


def test_prune_tree():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))
    t = t.prune((0, 1))

    assert t.node_at(()).is_internal() is True
    assert t.node_at((0, )).is_internal() is True
    assert t.node_at((1, )).is_terminal() is True
    assert t.node_at((0, 0)).is_terminal() is True
    assert t.node_at((0, 1)).is_terminal() is True


def test_change_tree():
    t = Tree(Node.terminal(0.0))
    t = t.grow(())
    t = t.grow((0, ))
    t = t.grow((0, 1))
    old_root_left = t.root.left
    old_root_right = t.root.right
    t = t.change((), 2, 5)

    assert t.root.variable == 2
    assert t.root.value == 5
    assert t.root.left is old_root_left
    assert t.root.right is old_root_right


def test_swap_tree():
    t = Tree(Node.terminal(0.0))
    t = t.grow((), variable=1, value=2,)
    t = t.grow((0, ), variable=3, value=4)
    t = t.grow((0, 1))
    t = t.swap((0, ))

    assert t.root.variable == 3
    assert t.root.value == 4
    assert t.root.left.variable == 1
    assert t.root.left.value == 2


def test_predict():
    t = Tree(Node.terminal(0.0))
    t = t.grow((), variable=0, value=1.0)
    t = t.grow((0,), variable=1, value=-2.0, mu_left=10.0, mu_right=-1.0)
    t = t.grow((1,), variable=1, value=2.0, mu_left=3.0, mu_right=2.0)

    x1 = [2.0, 3.0]
    x2 = [2.0, -2.0]
    x3 = [-2.0, -3.0]
    x4 = [-2.0, 0.0]

    y = t.predict(np.asarray([x1, x2, x3, x4]))

    assert t.predict(x1) == 2.0
    assert t.predict(x2) == 3.0
    assert t.predict(x3) == 10.0
    assert t.predict(x4) == -1.0
    assert y[0] == 2.0
    assert y[1] == 3.0
    assert y[2] == 10.0
    assert y[3] == -1.0

def test_validate():
    t = Tree(Node.terminal(0.0))
    t = t.grow((), variable=0, value=1.0)
    t = t.grow((0,), variable=1, value=-2.0, mu_left=10.0, mu_right=-1.0)
    t = t.grow((1,), variable=1, value=2.0, mu_left=3.0, mu_right=2.0)

    t._validate()