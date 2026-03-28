"""Tree structures and operations for a BART-style tree.

This module defines node and tree objects for representing binary regression
trees used in Bayesian Additive Regression Trees. It includes helpers for
navigating a tree, applying updates, validating tree shape, and computing 
predictions.
"""
from __future__ import annotations
from dataclasses import dataclass

import random
import numpy as np


@dataclass
class Node:
    """A node in a binary regression tree.

    A node is either terminal, in which case it stores a leaf value ``mu``,
    or internal, in which case it stores a split rule and references to its
    left and right children.
    """
    left: Node | None
    right: Node | None
    variable: int | None
    value: float | None
    mu: float | None

    def __init__(self,
                 mu: float | None = None,
                 variable: int | None = None,
                 value: float | None = None,
                 left: "Node | None" = None,
                 right: "Node | None" = None,):
        self.mu = mu
        self.variable = variable
        self.value = value
        self.left = left
        self.right = right

    @classmethod
    def internal(cls,
                 variable: int,
                 value: float,
                 left_node: Node,
                 right_node: Node):
        """Create an internal node.

        The returned node stores a split rule and references to its left and
        right child nodes.
        """
        return Node(variable=variable,
                    value=value,
                    left=left_node,
                    right=right_node,
                    mu=None)

    @classmethod
    def terminal(cls, mu: float):
        """Create a terminal node.

        The returned node stores ``mu`` value and has no children.
        """
        return Node(mu=mu,
                    left=None,
                    right=None,
                    variable=None,
                    value=None)

    def is_internal(self):
        """Return whether this node is a valid internal node.

        An internal node has a split variable, a split value, two children,
        and no ``mu`` value.
        """
        return (self.variable is not None
                and self.value is not None
                and self.left is not None
                and self.right is not None
                and self.mu is None)

    def is_terminal(self):
        """Return whether this node is a valid terminal node.

        A terminal node has a ``mu`` value and no split rule or children.
        """
        return (self.variable is None
                and self.value is None
                and self.left is None
                and self.right is None
                and self.mu is not None)


class Tree:
    """A binary regression tree.

    The tree stores a root node and provides methods for querying structure,
    replacing subtrees, applying tree modifications, and making predictions.
    """
    root: Node

    def __init__(self, root: Node):
        self.root = root

    def node_at(self, path: tuple):
        """Return the node at the given path.

        The path is a tuple of directions, where ``0`` means go left and ``1``
        means go right.
        """
        node = self.root
        for p in path:
            if p == 0:
                node = node.left
            else:
                node = node.right
        return node

    def _collect_paths(self,
                       node: Node,
                       current_path: tuple,
                       paths: list,
                       condition: function):
        """Collect paths to nodes that satisfy a condition.

        This helper traverses the subtree rooted at ``node`` and appends the
        path of each matching node to ``paths``.
        """
        if condition(node):
            paths.append(current_path)
        if node.is_terminal():
            return
        self._collect_paths(node.left, current_path + (0,), paths, condition)
        self._collect_paths(node.right, current_path + (1,), paths, condition)

    def terminal_paths(self):
        """Return the paths of all terminal nodes in the tree."""
        paths = []
        self._collect_paths(self.root, (), paths, lambda x: x.is_terminal())
        return paths

    def internal_paths(self):
        """Return the paths of all internal nodes in the tree."""
        paths = []
        self._collect_paths(self.root, (), paths, lambda x: x.is_internal())
        return paths

    def prunable_paths(self):
        """Return the paths of all prunable internal nodes.

        A prunable node is an internal node whose two children are both
        terminal nodes.
        """
        paths = []
        self._collect_paths(self.root,
                            (),
                            paths,
                            lambda x: (x.is_internal()
                                       and x.left.is_terminal()
                                       and x.right.is_terminal()))
        return paths

    def swappable_paths(self):
        """Return the paths of internal nodes that can be swapped.

        Each returned path points to a child internal node whose split rule
        can be exchanged with its parent.
        """
        paths = []

        def visit(node, path):
            if node.is_terminal():
                return

            left_path = path + (0,)
            right_path = path + (1,)

            if node.left.is_internal():
                paths.append(left_path)

            if node.right.is_internal():
                paths.append(right_path)

            visit(node.left, left_path)
            visit(node.right, right_path)

        visit(self.root, ())
        return paths

    def _depth(self, node: Node):
        """Return the depth of the subtree rooted at ``node``.

        A terminal node has depth zero. An internal node has depth equal to
        one plus the maximum depth of its children.
        """
        if node.is_terminal():
            return 0
        return max(self._depth(node.left),
                   self._depth(node.right))

    def max_depth(self):
        """Return the maximum depth of the tree."""
        return self._depth(self.root)

    def _replace_subtree(self, node: Node, path: tuple, replacement: Tree):
        """Return a copy of a subtree with one branch replaced.

        The subtree rooted at ``node`` is copied recursively, and the subtree
        at ``path`` is replaced with ``replacement``.
        """
        if len(path) == 0:
            return replacement

        if node.is_terminal():
            raise ValueError

        direction = path[0]
        rest = path[1:]

        if direction == 0:
            if node.left is None:
                raise ValueError
            new_left = self._replace_subtree(node.left, rest, replacement)

            return Node.internal(variable=node.variable,
                                 value=node.value,
                                 left_node=new_left,
                                 right_node=node.right)
        elif direction == 1:
            if node.right is None:
                raise ValueError
            new_right = self._replace_subtree(node.right, rest, replacement)

            return Node.internal(variable=node.variable,
                                 value=node.value,
                                 left_node=node.left,
                                 right_node=new_right)
        else:
            raise ValueError

    def grow(self,
             path: tuple,
             variable=0,
             value=0,
             mu_left=0.0,
             mu_right=0.0):
        """Return a new tree with a terminal node split into two children.

        The node at ``path`` must be terminal. It is replaced by an internal
        node with the given split rule and two new terminal children.
        """
        if self.node_at(path).is_internal():
            raise ValueError("Cannot grow on an internal node.")
        replacement = Node.internal(variable=variable,
                                    value=value,
                                    left_node=Node.terminal(mu_left),
                                    right_node=Node.terminal(mu_right))
        new_root = self._replace_subtree(self.root, path, replacement)
        return Tree(new_root)

    def prune(self, path: tuple, mu=0.0):
        """Return a new tree with a subtree collapsed into a terminal node.

        The subtree at ``path`` is replaced by a terminal node with leaf value
        ``mu``.
        """
        if self.node_at(path).is_terminal():
            raise ValueError("Cannot prune an internal node.")
        replacement = Node.terminal(mu)
        new_root = self._replace_subtree(self.root, path, replacement)
        return Tree(new_root)

    def change(self, path: tuple, variable=0, value=0):
        """Return a new tree with an updated split rule.

        The node at ``path`` must be internal. Its children are kept, but its
        split variable and split value are replaced.
        """
        if self.node_at(path).is_terminal():
            raise ValueError("Cannot change split rule of a terminal node.")
        old_node = self.node_at(path)
        if old_node.is_terminal():
            raise ValueError
        replacement = Node.internal(variable=variable,
                                    value=value,
                                    left_node=old_node.left,
                                    right_node=old_node.right)
        new_root = self._replace_subtree(self.root, path, replacement)
        return Tree(new_root)

    def swap(self, path: tuple):
        """Return a new tree with a parent-child split swap applied.

        The path must point to an internal child node. The split rule at that
        child is exchanged with the split rule at its parent.
        """
        if path == ():
            raise ValueError("Path must point to the child node which is to be swapped.")
        child = self.node_at(path)
        parent = self.node_at(path[:-1])
        if child.is_terminal() or parent.is_internal():
            raise ValueError("Both parent and child must be internal nodes in order to be swapped.")
        if path[-1] == 0:
            replacement = Node.internal(variable=child.variable,
                                        value=child.value,
                                        left_node=Node.internal(parent.variable,
                                                                parent.value,
                                                                child.left,
                                                                child.right),
                                        right_node=parent.right)
        else:
            replacement = Node.internal(variable=child.variable,
                                        value=child.value,
                                        left_node=parent.left,
                                        right_node=Node.internal(parent.variable,
                                                                 parent.value,
                                                                 child.left,
                                                                 child.right))
        new_root = self._replace_subtree(self.root, path[:-1], replacement)
        return Tree(new_root)

    def _predict(self, x):
        """Return the prediction for a single input vector.

        The input is routed from the root to a terminal node according to the
        stored split rules, and the ``mu`` value is returned.
        """
        current_node = self.root
        while current_node.is_internal():
            if x[current_node.variable] <= current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.mu

    def predict(self, X):
        """Return predictions for one or more input vectors.

        If ``X`` is one-dimensional, a single prediction is returned. If
        ``X`` is two-dimensional, a prediction is returned for each row.
        """
        X = np.asarray(X)

        if X.ndim == 1:
            return self._predict(X)

        if X.ndim == 2:
            n = X.shape[0]
            y = np.zeros(n)
            for i in range(n):
                y[i] = self._predict(X[i])
            return y
        raise ValueError("X must be an array of a 2d-matrix.")

    def _validate(self):
        """Check that the tree is structurally valid.

        This method raises a ``ValueError`` if any node violates the internal
        or terminal node invariants.
        """
        def visit(node: Node):
            if node.is_terminal():
                if node.mu is None:
                    raise ValueError("Terminal node must have mu.")
                if node.left is not None or node.right is not None:
                    raise ValueError("Terminal node cannot have children.")
                if node.variable is not None or node.value is not None:
                    raise ValueError("Terminal node cannot have split rule.")
            if node.is_internal():
                if node.mu is not None:
                    raise ValueError("Internal node cannot have mu.")
                if node.left is None or node.right is None:
                    raise ValueError("Internal node must have both children.")
                if node.variable is None or node.value is None:
                    raise ValueError("Internal node must have split rule.")
            else:
                raise ValueError("Every node must be internal or terminal.")
            visit(node.left)
            visit(node.right)
        if self.root is None:
            raise ValueError("Tree must have a root.")
        visit(self.root)

    def _draw(self) -> str:
        """Return a string representation of the tree.

        Internal nodes are shown by their split rules, and terminal nodes are
        shown by their leaf values.
        """
        def visit(node, prefix="", is_last=True):
            if node.is_terminal():
                label = f"{node.mu}"
            else:
                label = f"x{node.variable} <= {node.value}"

            lines = [prefix + ("└─ " if prefix else "") + label]

            if not node.is_terminal():
                child_prefix = prefix + ("   " if is_last else "│  ")
                lines += visit(node.left, child_prefix, False)
                lines += visit(node.right, child_prefix, True)

            return lines
        return "\n".join(visit(self.root))

    def show(self) -> None:
        """Print a readable representation of the tree."""
        print(self._draw())
