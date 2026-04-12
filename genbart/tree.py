"""Tree structures and operations for a BART-style tree.

This module defines node and tree objects for representing binary regression
trees used in Bayesian Additive Regression Trees. It includes helpers for
navigating a tree, applying updates, validating tree shape, and computing
predictions.
"""
from __future__ import annotations
from dataclasses import dataclass


import numpy as np
from typing import Literal


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
    rows: np.ndarray

    def __init__(self,
                 mu: float | None = None,
                 variable: int | None = None,
                 value: float | None = None,
                 left: "Node | None" = None,
                 right: "Node | None" = None,
                 rows: np.ndarray | None = None):
        self.mu = mu
        self.variable = variable
        self.value = value
        self.left = left
        self.right = right
        if rows is None:
            self.rows = None
        else:
            self.rows = np.asarray(rows, dtype=np.intp)

    @classmethod
    def internal(cls,
                 variable: int,
                 value: float,
                 left_node: Node,
                 right_node: Node,
                 rows):
        """Create an internal node.

        The returned node stores a split rule and references to its left and
        right child nodes.
        """
        return Node(variable=variable,
                    value=value,
                    left=left_node,
                    right=right_node,
                    rows=rows,
                    mu=None)

    @classmethod
    def terminal(cls,
                 mu: float,
                 rows):
        """Create a terminal node.

        The returned node stores ``mu`` value and has no children.
        """
        return Node(mu=mu,
                    rows=rows,
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
                and self.mu is None
                and self.rows is not None)

    def is_terminal(self):
        """Return whether this node is a valid terminal node.

        A terminal node has a ``mu`` value and no split rule or children.
        """
        return (self.variable is None
                and self.value is None
                and self.left is None
                and self.right is None
                and self.mu is not None
                and self.rows is not None)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.is_internal():
                return (other.is_internal()
                        and self.variable == other.variable
                        and self.value == other.value)
            if self.is_terminal():
                return (other.is_terminal()
                        and self.mu == other.mu)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Tree:
    """A binary regression tree.

    The tree stores a root node and the data associated with the tree,
    and provides methods for querying structure, replacing subtrees,
    applying tree modifications, and making predictions.
    """
    root: Node
    data: np.ndarray

    def __init__(self,
                 root: Node | None = None,
                 data: np.ndarray | None = None):
        if root is None:
            self.root = Node.terminal(0.0, np.arange(data.shape[0], dtype=np.intp))
        else:
            self.root = root
        self.data = data

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

    def terminal_paths(self, root_path=(), growable=True):
        """Return the paths of all (growable) terminal nodes in the tree."""
        root = self.node_at(root_path)
        paths = []
        if growable:
            self._collect_paths(root, root_path, paths,
                                lambda x: x.is_terminal() and len(x.rows) > 1)
        else:
            self._collect_paths(root, root_path, paths,
                                lambda x: x.is_terminal())
        return paths

    def internal_paths(self, root_path=()):
        """Return the paths of all internal nodes in the tree."""
        root = self.node_at(root_path)
        paths = []
        self._collect_paths(root, root_path, paths, lambda x: x.is_internal())
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

        Each returned path points to a parent internal node whose split rule
        can be exchanged with at least one of its children.
        """
        paths = []

        def visit(node, path):
            if node.is_terminal():
                return

            left_path = path + (0,)
            right_path = path + (1,)

            if node.left.is_internal() or node.right.is_internal():
                paths.append(path)

            visit(node.left, left_path)
            visit(node.right, right_path)

        visit(self.root, ())
        return paths

    def terminal_nodes(self):
        """Return a list of terminal nodes.

        Each element of the list is a reference to a terminal node
        in the tree.
        """
        return list(self._iter_terminal_node(self.root))

    def get_rows(self, path: tuple):
        """Return indices of rows of data associated with the node at path."""
        return self.node_at(path).rows

    def is_in_path(self, path: tuple, x):
        "Return True if data x belongs to path."
        x = np.asarray(x)

        current_node = self.root
        i = 0
        while current_node.is_internal():
            if x[current_node.variable] <= current_node.value and path[i] == 1:
                return False
            elif x[current_node.variable] <= current_node.value:
                current_node = current_node.left
                i = i + 1
                continue
            elif x[current_node.variable] > current_node.value and path[i] == 0:
                return False
            else:
                current_node = current_node.right
                i = i + 1
        return True

    def max_depth(self):
        """Return the maximum depth of the tree."""
        return self._depth(self.root)

    def _depth(self, node: Node):
        """Return the depth of the subtree rooted at ``node``.

        A terminal node has depth zero. An internal node has depth equal to
        one plus the maximum depth of its children.
        """
        if node.is_terminal():
            return 0
        return max(self._depth(node.left) + 1,
                   self._depth(node.right) + 1)

    def replace_subtree(self, path: tuple, replacement: Tree, node: Node = None):
        """Return a copy of a subtree with one branch replaced.

        The subtree rooted at ``node`` is copied recursively, and the subtree
        at ``path`` is replaced with ``replacement``.
        """
        if len(path) == 0:
            self.root = replacement.root
        else:
            parent = self.node_at(path[:-1])
            if path[-1] == 0:
                parent.left = replacement.root
            else:
                parent.right = replacement.root
        return self

    def grow(self,
             path: tuple,
             variable=0,
             value=0):
        """Return a new tree with a terminal node split into two children.

        The node at ``path`` must be terminal. It is replaced by an internal
        node with the given split rule and two new terminal children.
        """
        if self.node_at(path).is_internal():
            raise ValueError("Cannot grow on an internal node.")
        rows = self.get_rows(path)
        left_mask = self.data[rows, variable] <= value
        rows_l = rows[left_mask]
        rows_r = rows[~left_mask]
        replacement = Node.internal(variable=variable,
                                    value=value,
                                    left_node=Node.terminal(0.0,
                                                            rows_l),
                                    right_node=Node.terminal(0.0,
                                                             rows_r),
                                    rows=rows)
        return Tree(replacement, self.data)

    def prune(self, path: tuple, mu=0.0):
        """Return a new tree with a subtree collapsed into a terminal node.

        The subtree at ``path`` is replaced by a terminal node with leaf value
        ``mu``.
        """
        if self.node_at(path).is_terminal():
            raise ValueError("Cannot prune an internal node.")
        replacement = Node.terminal(mu, self.get_rows(path))
        return Tree(replacement, self.data)

    def change(self, path: tuple, variable=0, value=0):
        """Return a new tree with an updated split rule.

        The node at ``path`` must be internal. Its children are kept, but its
        split variable and split value are replaced.
        """
        old_node = self.node_at(path)
        if old_node.is_terminal():
            raise ValueError("Cannot change split rule of a terminal node.")
        replacement = Node.internal(variable=variable,
                                    value=value,
                                    left_node=old_node.left,
                                    right_node=old_node.right,
                                    rows=old_node.rows)
        terminals = []
        internals = []
        replacement = self._update_data_rows(replacement, terminals, internals)
        if replacement is None:
            return None, None, None
        else:
            return Tree(replacement, self.data), terminals, internals

    def swap(self, path: tuple, swap: Literal["left", "right", "both"]):
        """Return a new tree with a parent-child split swap applied.

        The path must point to an internal parent node. The split rule at that
        parent is exchanged with the split rule at its child(ren).
        """
        parent = self.node_at(path)
        if parent.is_terminal():
            raise ValueError("Path must point to the parent node which is to be swapped.")
        if swap == "left":
            child = self.node_at(path + (0, ))
            
            replacement = Node.internal(variable=child.variable,
                                        value=child.value,
                                        left_node=Node.internal(parent.variable,
                                                                parent.value,
                                                                child.left,
                                                                child.right,
                                                                []),
                                        right_node=parent.right,
                                        rows=parent.rows)
        elif swap == "right":
            child = self.node_at(path + (1, ))
            replacement = Node.internal(variable=child.variable,
                                        value=child.value,
                                        left_node=parent.left,
                                        right_node=Node.internal(parent.variable,
                                                                 parent.value,
                                                                 child.left,
                                                                 child.right,
                                                                 child.rows),
                                        rows=parent.rows)
        else:
            child_l = self.node_at(path + (0, ))
            child_r = self.node_at(path + (1, ))
            replacement = Node.internal(variable=child_l.variable,
                                        value=child_l.value,
                                        left_node=Node.internal(parent.variable,
                                                                parent.value,
                                                                child_l.left,
                                                                child_l.right,
                                                                child_l.rows),
                                        right_node=Node.internal(parent.variable,
                                                                 parent.value,
                                                                 child_r.left,
                                                                 child_r.right,
                                                                 child_r.rows),
                                        rows=parent.rows)

        replacement = self._update_data_rows(replacement)
        if replacement is None:
            return None
        else:
            return Tree(replacement, self.data)

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
                y[i] = self._predict(X[i, :])
            return y
        raise ValueError("X must be an array of a 2d-matrix.")
    
    def count_nodes(self):
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += 1
            if not node.is_terminal():
                stack.append(node.right)
                stack.append(node.left)
        return n
    
    def serialize(self):
        n = self.count_nodes()

        variable = np.full(n, -1, dtype=np.int32)
        value = np.zeros(n, dtype=np.float32)
        left = np.full(n, -1, dtype=np.int32)
        right = np.full(n, -1, dtype=np.int32)
        mu = np.zeros(n, dtype=np.float32)

        stack = [(self.root, -1, False)]
        next_idx = 0

        while stack:
            node, parent_idx, is_right_child = stack.pop()
            idx = next_idx
            next_idx += 1

            if parent_idx != -1:
                if is_right_child:
                    right[parent_idx] = idx
                else:
                    left[parent_idx] = idx

            if node.is_terminal():
                mu[idx] = float(node.mu)
            else:
                variable[idx] = int(node.variable)
                value[idx] = float(node.value)

                stack.append((node.right, idx, True))
                stack.append((node.left, idx, False))

        return SerializedTree(
            variable=variable,
            value=value,
            left=left,
            right=right,
            mu=mu,
        )

    def _update_data_rows(self, node: Node,
                          terminals: list[Node] = None,
                          internals: list[Node] = None):
        """Return a subtree with data rows updated.

        Data rows are updated recursevly from node downwards. Used
        for calculating replacement trees in change and swap functions.
        """
        if node.is_terminal():
            if terminals is not None:
                terminals.append(node)
            return node
        if internals is not None:
            internals.append(node)

        variable = node.variable
        rows = node.rows
        value = node.value

        left_mask = self.data[rows, variable] <= value
        rows_l = rows[left_mask]
        rows_r = rows[~left_mask]
        if rows_l.size == 0 or rows_r.size == 0:
            return None

        if node.left.is_terminal():
            node_l = Node.terminal(node.left.mu, rows_l)
        else:
            node_l = Node.internal(node.left.variable,
                                   node.left.value,
                                   node.left.left,
                                   node.left.right,
                                   rows_l)
        if node.right.is_terminal():
            node_r = Node.terminal(node.right.mu, rows_r)
        else:
            node_r = Node.internal(node.right.variable,
                                   node.right.value,
                                   node.right.left,
                                   node.right.right,
                                   rows_r)
            
        new_left = self._update_data_rows(node_l, terminals, internals)
        if new_left is None:
            return None
        new_right = self._update_data_rows(node_r, terminals, internals)
        if new_right is None:
            return None

        return Node.internal(variable=variable,
                             value=value,
                             left_node=new_left,
                             right_node=new_right,
                             rows=node.rows)

    def _map_to_value(self, rows: np.ndarray, variable: int, old_value: float):
        vals = np.unique(self.data[rows, variable])
        if vals.size <= 1:
            return None

        left_count = int(np.searchsorted(vals, old_value, side="right"))
        if left_count == 0 or left_count == vals.size:
            return None

        return vals[left_count - 1]

    def _iter_terminal_node(self, node):
        if node.is_terminal():
            yield node
        else:
            yield from self._iter_terminal_node(node.left)
            yield from self._iter_terminal_node(node.right)

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
            elif node.is_internal():
                if node.mu is not None:
                    raise ValueError("Internal node cannot have mu.")
                if node.left is None or node.right is None:
                    raise ValueError("Internal node must have both children.")
                if node.variable is None or node.value is None:
                    raise ValueError("Internal node must have split rule.")
                visit(node.left)
                visit(node.right)
            else:
                raise ValueError("Every node must be internal or terminal.")

        if self.root is None:
            raise ValueError("Tree must have a root.")
        visit(self.root)

    def _draw(self, show_rows: bool = True) -> str:
        """Return a compact top-down string representation of the tree.

        Internal nodes are shown as ``x{variable}<={value}``.
        Terminal nodes are shown as ``mu={value}``.

        If ``show_rows`` is True, the row indices are printed on a second line
        under the node label.
        """
        def fmt_num(x):
            return f"{x:g}"

        def fmt_rows(rows, max_items=4):
            if rows is None:
                return ""
            if len(rows) <= max_items + 1:
                body = ", ".join(str(r) for r in rows)
            else:
                body = ", ".join(str(r) for r in rows[:max_items])
                body += f", ..., {rows[-1]}"
            return f"r=[{body}]"

        def node_lines(node):
            if node.is_terminal():
                lines = [f"mu={fmt_num(node.mu)}"]
            else:
                lines = [f"x{node.variable}<={fmt_num(node.value)}"]

            if show_rows:
                row_text = fmt_rows(node.rows)
                if row_text:
                    lines.append(row_text)

            width = max(len(line) for line in lines)
            return [line.center(width) for line in lines], width

        def pad_lines(lines, width, height):
            out = [line.ljust(width) for line in lines]
            while len(out) < height:
                out.append(" " * width)
            return out

        def build(node):
            label, label_width = node_lines(node)

            if node.is_terminal():
                root = label_width // 2
                return label, label_width, root

            left_lines, left_width, left_root = build(node.left)
            right_lines, right_width, right_root = build(node.right)

            gap = 3
            children_width = left_width + gap + right_width
            total_width = max(label_width, children_width)

            if total_width == children_width:
                label_start = (total_width - label_width) // 2
                left_start = 0
            else:
                label_start = 0
                left_start = (total_width - children_width) // 2

            right_start = left_start + left_width + gap
            root = label_start + label_width // 2
            left_child = left_start + left_root
            right_child = right_start + right_root

            top = []
            for line in label:
                top.append(
                    " " * label_start
                    + line
                    + " " * (total_width - label_start - len(line))
                )

            # only show the branch endpoints, no horizontal connector line
            branch = [" "] * total_width
            if left_child != root:
                branch[left_child] = "/"
            if right_child != root:
                branch[right_child] = "\\"
            branch_line = "".join(branch)

            child_height = max(len(left_lines), len(right_lines))
            left_lines = pad_lines(left_lines, left_width, child_height)
            right_lines = pad_lines(right_lines, right_width, child_height)

            merged = []
            for left, right in zip(left_lines, right_lines):
                merged.append(
                    " " * left_start
                    + left
                    + " " * gap
                    + right
                    + " " * (total_width - right_start - right_width)
                )

            return top + [branch_line] + merged, total_width, root

        lines, _, _ = build(self.root)
        return "\n".join(line.rstrip() for line in lines)

    def show(self, show_rows: bool = True) -> None:
        """Print a readable top-down representation of the tree."""
        print(self._draw(show_rows=show_rows))


@dataclass(slots=True)
class SerializedTree:
    variable: np.ndarray
    value: np.ndarray
    left: np.ndarray
    right: np.ndarray
    mu: np.ndarray
    # leaf iff left[idx] == -1


    def count_nodes(root):
        n = 0
        stack = [root]
        while stack:
            node = stack.pop()
            n += 1
            if not node.is_terminal():
                stack.append(node.right)
                stack.append(node.left)
        return n