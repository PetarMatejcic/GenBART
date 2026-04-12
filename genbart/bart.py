import numpy as np
import copy
from scipy.stats import chi2
from genbart.tree import Node, Tree, SerializedTree
from .profilestats import ProfileStats
from contextlib import nullcontext


class bart:
    m: int
    alpha: float
    beta: float
    k: float
    nu: float
    q: float
    n_burn: int
    n_samples: int
    move_distribution: tuple
    rng: int | None

    X: np.ndarray
    y: np.ndarray
    n: int
    p: int
    y_shift: float
    y_scale: float

    sigma2: float
    sigma_mu2: float
    lambda_: float

    trees: list[Tree]
    training_predictions: np.ndarray
    fitted_sums: np.ndarray
    tree_sample: list[dict]

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 nu=3.0,
                 q=0.90,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=(0.25, 0.25, 0.40, 0.10),
                 random_state=None,
                 profile: bool = False):
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.nu = nu
        self.q = q
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.move_distribution = move_distribution
        self.rng = np.random.default_rng(seed=random_state)
        self.profile = profile
        self.profiler = ProfileStats() if profile else None

    def _section(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.section(name)

    def fit(self, X, y):
        # Transforming and storing data.
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X.reshape((-1, 1))
        elif self.X.ndim > 2:
            raise ValueError
        self.y = np.asarray(y)
        self.y_scale = max(self.y) - min(self.y)
        if self.y_scale < 1e-16:
            self.y_scale = 1.0
        self.y_shift = (min(self.y)+max(self.y)) / 2.0
        self.y = (self.y - self.y_shift) / self.y_scale
        self.n, self.p = self.X.shape

        # Initializing trees.
        rows_by_var = [np.argsort(self.X[:, var], kind="mergesort") for var in range(self.p)]
        self.trees = [Tree(data=self.X, rows_by_var=rows_by_var) for _ in range(self.m)]

        # Initializing mu_ij|T_j prior ~ N(0, sigma_mu2)
        self.sigma_mu2 = (0.5 / (self.k * np.sqrt(self.m)))**2
        # Initializing sigma2 prior.
        A_sigma = np.column_stack([self.X, np.ones(self.n)])
        denom = (self.n - np.linalg.matrix_rank(A_sigma))
        if denom > 0:
            sigma_linalg = np.linalg.lstsq(A_sigma, self.y, rcond=None)
            rss = np.sum((self.y - A_sigma @ sigma_linalg[0])**2)
            self.sigma2 = rss / denom
        else:
            self.sigma2 = np.var(self.y)
        self.lambda_ = (self.sigma2 / self.nu) * chi2.ppf(1-self.q, df=self.nu)

        # Initializing predictions.
        self.training_predictions = np.zeros((self.m, self.n))
        self.fitted_sums = np.zeros(self.n)
        self.tree_sample = []

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            sample = []
            for j in range(self.m):
                sample.append(self.trees[j].serialize())
            self.tree_sample.append({"sample": sample,
                            "sigma2": self.sigma2})
        return self

    def predict(self, X, level: float = 0.90):
        data = np.asarray(X)
        a = 1 - level
        if data.ndim == 1 and self.p > 1:
            predictions = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[i] += self._predict_serialized_tree_row(data,
                                                                        self.tree_sample[i]["sample"][j])
            conf_int = np.quantile(predictions, [a/2.0, 1 - a/2.0])
            return (self._inverse_transform_y(predictions.mean()),
                    tuple(self._inverse_transform_y(conf_int)))
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            predictions = np.zeros((X.shape[0], self.n_samples))
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[:, i] += self._predict_serialized_tree_matrix(data,
                                                                              self.tree_sample[i]["sample"][j])
            conf_ints = np.quantile(predictions, [a/2.0, 1 - a/2.0], axis=1)
            return (self._inverse_transform_y(predictions.mean(axis=1)),
                    self._inverse_transform_y(conf_ints))
        
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

    def _one_mcmc_iteration(self):
        with self._section("iter.total"):
            for j in range(self.m):
                self._draw_tree(j)
                with self._section("draw_mu"):
                    self._draw_mu(j)
                with self._section("update_tps"):
                    self._update_tps_and_fitted_sums_incremental(j)
            self.sigma2 = self._draw_sigma()

    def _draw_tree(self, j: int):
        move = self.rng.choice(["grow", "prune", "change", "swap"],
                               p=self.move_distribution)
        if move == "grow":
            with self._section("propose.grow"):
                proposed_subtree, mh_ratio, old_path = self._propose_tree_grow(j)
        elif move == "prune":
            with self._section("propose.prune"):
                proposed_subtree, mh_ratio, old_path = self._propose_tree_prune(j)
        elif move == "change":
            with self._section("propose.change"):
                proposed_subtree, mh_ratio, old_path = self._propose_tree_change(j)
        elif move == "swap":
            with self._section("propose.swap"):
                proposed_subtree, mh_ratio, old_path = self._propose_tree_swap(j)
        else:
            return

        if proposed_subtree is None:
            return

        u = np.log(self.rng.uniform())
        if u < min(0.0, mh_ratio):
            self.trees[j].replace_subtree(old_path, proposed_subtree)
            return True
        return False

    def _propose_tree_grow(self, j: int):
        with self._section("grow.propose"):
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
            splits = node.split_values_by_var[variable]
            eta_ = node.eta_by_var[variable]
            value = splits[self.rng.integers(eta_)]

        with self._section("grow.mh_ratio"):
            proposed_subtree = self.trees[j].grow(path=path,
                                            variable=variable,
                                            value=value)
            log_transition_ratio = (np.log(self.move_distribution[1])
                                    + np.log(p_)
                                    + np.log(b_)
                                    + np.log(eta_)
                                    - np.log(self.move_distribution[0])
                                    - np.log(max(1, len(self.trees[j].prunable_paths()))))

            rows = node.rows
            rows_l = proposed_subtree.get_rows((0, ))
            rows_r = proposed_subtree.get_rows((1, ))
            log_likelihood_ratio = self._log_likelihood(j,
                                                        [rows_l, rows_r],
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

        log_likelihood_ratio = self._log_likelihood(j,
                                                    [rows_l, rows_r],
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

        new_rule = self._sample_uniform_change_rule(node)
        if new_rule is None:
            return None, None, None
        new_variable, new_value = new_rule

        proposed_subtree, subtree_terminals, subtree_internals = self.trees[j].change(path, new_variable, new_value)
        if proposed_subtree is None:
            return None, None, None
        subtree_terminal_paths = [ter.rows for ter in subtree_terminals]
        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]
        log_likelihood_ratio = self._log_likelihood(j,
                                                    subtree_terminal_paths,
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

        prop_subtree_terminals = [ter.rows
                                  for ter
                                  in subtree_terminals]
        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]

        log_likelihood_ratio = self._log_likelihood(j,
                                                    prop_subtree_terminals,
                                                    old_tree_terminals)


        old_tree_internals = [self.trees[j].node_at(path)
                              for path in self.trees[j].internal_paths(path)]
        log_tree_ratio = self._log_prior_ratio(subtree_internals,
                                               old_tree_internals)

        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_subtree, mh_ratio, path

    def _sample_uniform_change_rule(self, node):
        total_rules = int(node.eta_by_var[node.valid_vars].sum())
        if total_rules <= 1:
            return None

        u = int(self.rng.integers(total_rules - 1))

        skipped_current = False
        for var in node.valid_vars:
            splits = node.split_values_by_var[var]
            for value in splits:
                if (not skipped_current
                    and var == node.variable
                    and value == node.value):
                    skipped_current = True
                    continue

                if u == 0:
                    return int(var), value
                u -= 1
        raise RuntimeError("Failed to decode sampled splitting rule.")

    def _log_likelihood(self,
                        j: int,
                        new_rows: list[np.ndarray],
                        old_rows: list[np.ndarray]):
        residuals = self._partial_residuals(j)
        ratio = 0.0
        for row in new_rows:
            denom = self.sigma2 + len(row)*self.sigma_mu2 + 1e-12
            sse = residuals[row].sum()**2
            ratio += (0.5*np.log(self.sigma2/denom)
                      + (self.sigma_mu2*sse/(2.0*self.sigma2*denom)))
        for row in old_rows:
            denom = self.sigma2 + len(row)*self.sigma_mu2 + 1e-12
            sse = residuals[row].sum()**2
            ratio -= (0.5*np.log(self.sigma2/denom)
                      + (self.sigma_mu2*sse/(2.0*self.sigma2*denom)))
        return ratio

    def _p_split(self, d: int):
        return self.alpha / (1+d)**self.beta

    def _log_tree_prior_for_terminal(self, depth: int):
        return np.log(1 - self._p_split(depth))

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

    def _draw_mu(self, j: int):
        residuals = self._partial_residuals(j)
        nodes = self.trees[j].terminal_nodes()
        for node in nodes:
            rows = node.rows
            denom = len(rows)*self.sigma_mu2 + self.sigma2
            mu = self.rng.normal(loc=(self.sigma_mu2
                                      * residuals[rows].sum())/denom,
                                 scale=np.sqrt(self.sigma2
                                               * self.sigma_mu2/denom))
            node.mu = mu

    def _draw_sigma(self):
        sse = np.sum((self.y - self.fitted_sums)**2)
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.n),
                                    scale=2.0/(self.nu*self.lambda_ + sse))

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift

    def _partial_residuals(self, j):
        return self.y - self.fitted_sums + self.training_predictions[j, :]

    def _update_tps_and_fitted_sums_incremental(self, j: int):
        self.fitted_sums -= self.training_predictions[j, :]
        for node in self.trees[j].terminal_nodes():
            self.training_predictions[j, node.rows] = node.mu
        self.fitted_sums += self.training_predictions[j, :]
