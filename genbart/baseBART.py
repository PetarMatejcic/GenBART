import numpy as np
from scipy.stats import chi2
from genbart.tree import Node, Tree, SerializedTree
from .profilestats import ProfileStats
from contextlib import nullcontext


class BaseBART:
    m: int
    alpha: float
    beta: float
    k: float
    n_burn: int
    n_samples: int
    move_distribution: tuple
    rng: int | None

    X: np.ndarray
    n: int
    p: int

    sigma2: float
    sigma_mu2: float
    y_work: np.ndarray

    trees: list[Tree]
    training_predictions: np.ndarray
    residuals: np.ndarray
    fitted_sums: np.ndarray
    tree_sample: list[dict]

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=None):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.move_distribution = move_distribution
        self.rng = np.random.default_rng(seed=random_state)

    def _init_trees(self):
        rows_by_var = [np.argsort(self.X[:, var], kind="mergesort") for var in range(self.p)]
        self.trees = [Tree(data=self.X, rows_by_var=rows_by_var) for _ in range(self.m)]

    def _init_common_arrays(self):
        self.training_predictions = np.zeros((self.m, self.n))
        self.fitted_sums = np.zeros(self.n)
        self.residuals = np.zeros(self.n)
        self.tree_sample = []

    def variable_importance(self):
        if not self.tree_sample:
            raise RuntimeError("No posterior samples stored.")

        importance = np.zeros(self.p, dtype=float)

        for draw in self.tree_sample:
            counts = np.zeros(self.p, dtype=np.int64)
            total_splits = 0

            for tree in draw["sample"]:
                mask = tree.variable >= 0
                if np.any(mask):
                    counts += np.bincount(tree.variable[mask], minlength=self.p)
                    total_splits += int(mask.sum())

            if total_splits > 0:
                importance += counts / total_splits
        return importance / self.n_samples
    
    def _draw_tree(self, j: int):
        move = self.rng.choice(["grow", "prune", "change", "swap"],
                            p=self.move_distribution)
        if move == "grow":
            proposed_subtree, mh_ratio, old_path = self._propose_tree_grow(j)
        elif move == "prune":
            proposed_subtree, mh_ratio, old_path = self._propose_tree_prune(j)
        elif move == "change":
            proposed_subtree, mh_ratio, old_path = self._propose_tree_change(j)
        elif move == "swap":
            proposed_subtree, mh_ratio, old_path = self._propose_tree_swap(j)
        else:
            return

        if proposed_subtree is None:
            return

        u = np.log(self.rng.uniform())
        if u < min(0.0, mh_ratio):
            self.trees[j].replace_subtree(old_path, proposed_subtree)
            self.trees[j].finalize_subtree(old_path)
            return True
        return False

    def _propose_tree_grow(self, j: int):
        possible_paths = self.trees[j].terminal_paths()
        if not possible_paths:
            return None, None, None
        p_ = len(possible_paths) 
        path = possible_paths[self.rng.choice(p_)]
        node = self.trees[j].node_at(path)
        b_ = len(node.valid_vars)
        if b_ == 0:
            return None, None, None 
        variable = node.valid_vars[self.rng.integers(b_)]
        eta_ = int(node.eta_by_var[variable])
        split_idx = int(self.rng.integers(eta_))

        proposed_subtree = self.trees[j].grow(path=path,
                                        variable=variable,
                                        split_idx=split_idx)
        if proposed_subtree is None:
            return None, None, None
        log_transition_ratio = (np.log(self.move_distribution[1])
                                + np.log(p_)
                                + np.log(b_)
                                + np.log(eta_)
                                - np.log(self.move_distribution[0])
                                - np.log(max(1, len(self.trees[j].prunable_paths()))))

        rows = node.rows
        rows_l = proposed_subtree.get_rows((0, ))
        rows_r = proposed_subtree.get_rows((1, ))
        log_likelihood_ratio = self._log_likelihood([rows_l, rows_r],
                                                    [rows])

        d = len(path)
        log_tree_ratio = (np.log(self.alpha)
                        + 2.0*np.log(1 - self._p_split(d+1))
                        - np.log(((1+d)**self.beta) - self.alpha)
                        - np.log(b_)
                        - np.log(eta_))

        mh_ratio = (log_transition_ratio
                    + log_likelihood_ratio
                    + log_tree_ratio)
        return proposed_subtree, mh_ratio, path

    def _propose_tree_prune(self, j: int):
        possible_paths = self.trees[j].prunable_paths()
        if len(possible_paths) < 1:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        node = self.trees[j].node_at(path)

        rows = self.trees[j].get_rows(path)
        rows_l = self.trees[j].get_rows(path + (0, ))
        rows_r = self.trees[j].get_rows(path + (1, ))
        old_variable = self.trees[j].node_at(path).variable
        proposed_subtree = self.trees[j].prune(path)
        b_ = (len(self.trees[j].terminal_paths())
              - (len(rows_l) > 1)
              - (len(rows_r) > 1)
              + 1)
        
        p_ = len(node.valid_vars)
        eta_ = int(node.eta_by_var[old_variable])

        log_transition_ratio = (np.log(self.move_distribution[0])
                                + np.log(len(possible_paths))
                                - np.log(self.move_distribution[1])
                                - np.log(b_)
                                - np.log(p_)
                                - np.log(eta_))

        log_likelihood_ratio = self._log_likelihood([rows_l, rows_r],
                                                    [rows])

        d = len(path)
        log_tree_ratio = (np.log((1+d)**self.beta - self.alpha)
                          + np.log(p_)
                          + np.log(eta_)
                          - np.log(self.alpha)
                          - 2.0*np.log(1 - self._p_split(d+1)))

        mh_ratio = (log_transition_ratio
                    + log_likelihood_ratio
                    + log_tree_ratio)
        return proposed_subtree, mh_ratio, path

    def _propose_tree_change(self, j: int):
        possible_paths = self.trees[j].internal_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        node = self.trees[j].node_at(path)

        new_rule = self._sample_uniform_change_rule(node, self.trees[j])
        if new_rule is None:
            return None, None, None
        new_variable, new_split_idx = new_rule

        proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].change(path, new_variable, new_split_idx)
        if proposed_subtree is None:
            return None, None, None
        
        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]
        log_likelihood_ratio = self._log_likelihood(subtree_terminals,
                                                    old_tree_terminals)

        old_tree_internals = [self.trees[j].node_at(path)
                              for path in self.trees[j].internal_paths(path)]
        log_tree_ratio = self._log_prior_ratio(subtree_internals,
                                               old_tree_internals)
        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_subtree, mh_ratio, path

    def _propose_tree_swap(self, j: int):
        possible_paths = self.trees[j].swappable_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        parent = self.trees[j].node_at(path)
        children = [parent.left,
                    parent.right]
        if children[0].is_terminal():
            proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].swap(path, swap="right")
        elif children[1].is_terminal():
            proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].swap(path, swap="left")
        else:
            if children[0] == children[1]:
                proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].swap(path, swap="both")
            else:
                child = self.rng.choice(["left", "right"])
                proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].swap(path, swap=child)
        
        if proposed_subtree is None:
            return None, None, None

        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]

        log_likelihood_ratio = self._log_likelihood(subtree_terminals,
                                                    old_tree_terminals)


        old_tree_internals = [self.trees[j].node_at(path)
                              for path in self.trees[j].internal_paths(path)]
        log_tree_ratio = self._log_prior_ratio(subtree_internals,
                                               old_tree_internals)

        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_subtree, mh_ratio, path

    def _sample_uniform_change_rule(self, node, tree):
        vars_ = node.valid_vars
        counts = node.eta_by_var[vars_]
        total_rules = int(counts.sum())
        if total_rules <= 1:
            return None

        cur_var_pos = int(np.where(vars_ == node.variable)[0][0])
        cur_split_pos = tree.split_pos_of_value(
            node.rows_by_var[node.variable],
            node.variable,
            node.value,
        )
        cur_global = int(counts[:cur_var_pos].sum() + cur_split_pos)

        u = int(self.rng.integers(total_rules - 1))
        if u >= cur_global:
            u += 1

        prefix = np.cumsum(counts)
        var_pos = int(np.searchsorted(prefix, u, side="right"))
        prev = 0 if var_pos == 0 else int(prefix[var_pos - 1])
        split_pos = int(u - prev)

        var = int(vars_[var_pos])
        return var, split_pos

    def _draw_mu(self, j: int):
        nodes = self.trees[j].terminal_nodes()
        for node in nodes:
            rows = node.rows
            denom = len(rows)*self.sigma_mu2 + self.sigma2
            mu = self.rng.normal(loc=(self.sigma_mu2
                                      * self.residuals[rows].sum())/denom,
                                 scale=np.sqrt(self.sigma2
                                               * self.sigma_mu2/denom))
            node.mu = mu

    def _log_likelihood(self,
                        new_rows: list[np.ndarray],
                        old_rows: list[np.ndarray]):
        ratio = 0.0
        for row in new_rows:
            denom = self.sigma2 + len(row)*self.sigma_mu2 + 1e-12
            sse = self.residuals[row].sum()**2
            ratio += (0.5*np.log(self.sigma2/denom)
                      + (self.sigma_mu2*sse/(2.0*self.sigma2*denom)))
        for row in old_rows:
            denom = self.sigma2 + len(row)*self.sigma_mu2 + 1e-12
            sse = self.residuals[row].sum()**2
            ratio -= (0.5*np.log(self.sigma2/denom)
                      + (self.sigma_mu2*sse/(2.0*self.sigma2*denom)))
        return ratio

    def _p_split(self, d: int):
        return self.alpha / (1+d)**self.beta

    def _log_tree_prior_for_internal(self, node: Node):
        p_ = len(node.valid_vars)
        eta_ = int(node.eta_by_var[node.variable])
        return -np.log(p_) - np.log(eta_)

    def _log_prior_ratio(self,
                         proposed_tree_nodes: list[Node],
                         old_tree_nodes: list[Node]):
        prior_ratio = 0.0
        for node in proposed_tree_nodes:
            prior_ratio += self._log_tree_prior_for_internal(node)
        for node in old_tree_nodes:
            prior_ratio -= self._log_tree_prior_for_internal(node)
        return prior_ratio

    def _predict_serialized_tree_row(self,
                                     x: np.array,
                                     serialized_tree: SerializedTree):
        node = 0
        while serialized_tree.left[node] != -1:
            if x[serialized_tree.variable[node]] <= serialized_tree.value[node]:
                node = serialized_tree.left[node]
            else:
                node = serialized_tree.right[node]
        return serialized_tree.mu[node]
    
    def _predict_serialized_tree_matrix(self,
                                        X: np.ndarray,
                                        serialized_tree: SerializedTree):
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            out[i] = self._predict_serialized_tree_row(X[i, :], serialized_tree)
        return out
    
    def _partial_residuals(self, j):
        self.residuals = self.y_work - self.fitted_sums + self.training_predictions[j, :]

    def _update_tps_and_fitted_sums_incremental(self, j: int):
        self.fitted_sums -= self.training_predictions[j, :]
        for node in self.trees[j].terminal_nodes():
            self.training_predictions[j, node.rows] = node.mu
        self.fitted_sums += self.training_predictions[j, :]
