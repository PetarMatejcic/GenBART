import numpy as np
import copy
from scipy.stats import chi2
from genbart.tree import Node, Tree


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
                 random_state=None):
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
        self.trees = [Tree(data=self.X) for _ in range(self.m)]

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
        self.training_predictions = np.zeros((self.n, self.m))
        self.fitted_sums = np.zeros(self.n)
        self.tree_sample = []

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()

        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            self.tree_sample.append({"sample": copy.deepcopy(self.trees),
                                     "sigma2": self.sigma2})
        return self

    def predict(self, X, level: float = 0.90):
        data = np.asarray(X)
        a = 1 - level
        if data.ndim == 1 and self.p > 1:
            predictions = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[i] += self.tree_sample[i]["sample"][j].predict(data)
            conf_int = np.quantile(predictions, [a/2.0, 1 - a/2.0])
            return (self._inverse_transform_y(predictions.mean()),
                    tuple(self._inverse_transform_y(conf_int)))
        else:
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            predictions = np.zeros((X.shape[0], self.n_samples))
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[:, i] += self.tree_sample[i]["sample"][j].predict(data)
            conf_ints = np.quantile(predictions, [a/2.0, 1 - a/2.0], axis=1)
            return (self._inverse_transform_y(predictions.mean(axis=1)),
                    self._inverse_transform_y(conf_ints))

    def _one_mcmc_iteration(self):
        for j in range(self.m):
            self._draw_tree(j)
            self._draw_mu(j)
            self._update_tps_and_fitted_sums_incremental(j)
        self.sigma2 = self._draw_sigma()

    def _draw_tree(self, j: int):
        move = self.rng.choice(["grow", "prune", "change", "swap"],
                               p=self.move_distribution)
        if move == "grow":
            proposed_tree, mh_ratio, old_path = self._propose_tree_grow(j)
        elif move == "prune":
            proposed_tree, mh_ratio, old_path = self._propose_tree_prune(j)
        elif move == "change":
            proposed_tree, mh_ratio, old_path = self._propose_tree_change(j)
        elif move == "swap":
            proposed_tree, mh_ratio, old_path = self._propose_tree_swap(j)
        else:
            return

        if proposed_tree is None:
            return

        u = np.log(self.rng.uniform())
        if u < min(0.0, mh_ratio):
            self.trees[j] = proposed_tree

    def _propose_tree_grow(self, j: int):
        possible_paths = self.trees[j].terminal_paths()
        if not possible_paths:
            return None, None, None
        p_ = len(possible_paths)
        path = possible_paths[self.rng.choice(p_)]
        rows = self.trees[j].get_rows(path)
        value_counts = {}
        for var in range(self.p):
            value_counts[var] = np.unique(self.X[rows, var])
        possible_variables = [k for k, v in value_counts.items() if len(v) > 1]
        b_ = len(possible_variables)
        if b_ < 1:
            return None, None, None
        variable = possible_variables[self.rng.choice(b_)]
        eta_ = len(value_counts[variable]) - 1
        value = value_counts[variable][self.rng.choice(eta_)]

        proposed_tree = self.trees[j].grow(path=path,
                                           variable=variable,
                                           value=value)
        log_transition_ratio = (np.log(self.move_distribution[1])
                                + np.log(p_)
                                + np.log(b_)
                                + np.log(eta_)
                                - np.log(self.move_distribution[0])
                                - np.log(len(proposed_tree.prunable_paths())))

        rows_l = [r for r in proposed_tree.get_rows(path + (0,))
                  if self.X[r, variable] <= value]
        rows_r = [r for r in proposed_tree.get_rows(path + (1, ))
                  if self.X[r, variable] > value]
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
        return proposed_tree, mh_ratio, path

    def _propose_tree_prune(self, j: int):
        possible_paths = self.trees[j].prunable_paths()
        if len(possible_paths) < 1:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        rows = self.trees[j].get_rows(path)
        rows_l = self.trees[j].get_rows(path + (0, ))
        rows_r = self.trees[j].get_rows(path + (1, ))
        old_variable = self.trees[j].node_at(path).variable
        proposed_tree = self.trees[j].prune(path)
        b_ = len(proposed_tree.terminal_paths())
        value_count_dict = {}
        for var in range(self.p):
            value_count_dict[var] = np.unique(self.X[rows, var])
        possible_variables = [k for k, v in value_count_dict.items()
                              if len(v) > 1]
        p_ = len(possible_variables)
        possible_values = value_count_dict[old_variable]
        eta_ = len(possible_values) - 1

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
        return proposed_tree, mh_ratio, path

    def _propose_tree_change(self, j: int):
        possible_paths = self.trees[j].internal_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        old_variable = self.trees[j].node_at(path).variable
        old_value = self.trees[j].node_at(path).value
        rows = self.trees[j].get_rows(path)
        splitting_rules = self._valid_splitting_rules(rows)
        splitting_rules.remove((old_variable, old_value))
        if not splitting_rules:
            return None, None, None

        new_variable, new_value = self.rng.choice(splitting_rules)
        new_variable = int(new_variable)
        proposed_tree = self.trees[j].change(path, new_variable, new_value)

        prop_tree_terminals = [proposed_tree.get_rows(ter_path)
                               for ter_path
                               in proposed_tree.terminal_paths(path, False)]
        for ter_rows in prop_tree_terminals:
            if not ter_rows:
                return None, None, None
        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]
        log_likelihood_ratio = self._log_likelihood(j,
                                                    prop_tree_terminals,
                                                    old_tree_terminals)

        prop_tree_internals = [proposed_tree.node_at(path)
                               for path in proposed_tree.internal_paths(path)]
        old_tree_internals = [self.trees[j].node_at(path)
                              for path in self.trees[j].internal_paths(path)]
        log_tree_ratio = self._log_prior_ratio(prop_tree_internals,
                                               old_tree_internals)
        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_tree, mh_ratio, path

    def _propose_tree_swap(self, j: int):
        possible_paths = self.trees[j].swappable_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        children = [self.trees[j].node_at(path + (0, )),
                    self.trees[j].node_at(path + (1, ))]
        if children[0].is_terminal():
            proposed_tree = self.trees[j].swap(path, swap="right")
        elif children[1].is_terminal():
            proposed_tree = self.trees[j].swap(path, swap="left")
        else:
            if children[0] == children[1]:
                proposed_tree = self.trees[j].swap(path, swap="both")
            else:
                child = self.rng.choice(["left", "right"])
                proposed_tree = self.trees[j].swap(path, swap=child)

        prop_tree_terminals = [proposed_tree.get_rows(ter_path)
                               for ter_path
                               in proposed_tree.terminal_paths(path, False)]
        for ter_rows in prop_tree_terminals:
            if not ter_rows:
                return None, None, None
        old_tree_terminals = [self.trees[j].get_rows(ter_path)
                              for ter_path
                              in self.trees[j].terminal_paths(path, False)]

        log_likelihood_ratio = self._log_likelihood(j,
                                                    prop_tree_terminals,
                                                    old_tree_terminals)

        prop_tree_internals = [proposed_tree.node_at(path)
                               for path in proposed_tree.internal_paths(path)]
        old_tree_internals = [self.trees[j].node_at(path)
                              for path in self.trees[j].internal_paths(path)]
        log_tree_ratio = self._log_prior_ratio(prop_tree_internals,
                                               old_tree_internals)

        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_tree, mh_ratio, path

    def _valid_splitting_rules(self, rows: list[int]):
        value_count_dict = {}
        for var in range(self.p):
            value_count_dict[var] = np.unique(self.X[rows, var])
        return [(var, val) for var, values in value_count_dict.items()
                for val in values[:-1] if len(values) > 1]

    def _log_likelihood(self,
                        j: int,
                        new_rows: list[list],
                        old_rows: list[list]):
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
        variable = node.variable
        rows = node.rows
        value_counts = {}
        for var in range(self.p):
            value_counts[var] = len(np.unique(self.X[rows, var]))
        p_ = len([k for k, v in value_counts.items() if v > 1])
        eta_ = value_counts[variable] - 1
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
        paths = self.trees[j].terminal_paths(growable=False)
        for path in paths:
            rows = self.trees[j].get_rows(path)
            denom = len(rows)*self.sigma_mu2 + self.sigma2
            mu = self.rng.normal(loc=(self.sigma_mu2
                                      * residuals[rows].sum())/denom,
                                 scale=np.sqrt(self.sigma2
                                               * self.sigma_mu2/denom))
            self.trees[j].node_at(path).mu = mu

    def _draw_sigma(self):
        sse = np.sum((self.y - self.fitted_sums)**2)
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.n),
                                    scale=2.0/(self.nu*self.lambda_ + sse))

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift

    def _partial_residuals(self, j):
        return self.y - self.fitted_sums + self.training_predictions[:, j]

    def _update_training_predictions(self, j: int | None = None):
        if j is None:
            old_tp = self.training_predictions.copy()
            for i in range(self.n):
                for j in range(self.m):
                    self.training_predictions[i, j] = self.trees[j].predict(self.X[i, :])
            return old_tp - self.training_predictions
        else:
            old_tp = self.training_predictions[:, j].copy()
            self.training_predictions[:, j] = self.trees[j].predict(self.X)
            return old_tp - self.training_predictions[:, j]

    def _update_tps_and_fitted_sums_incremental(self, j: int):
        diff = self._update_training_predictions(j)
        self.fitted_sums = self.fitted_sums - diff
