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
    move_distribution: list
    rng: int | None

    X: np.ndarray
    y: np.ndarray
    n: int
    p: int
    y_shift: float
    y_scale: float

    sigma: float
    sigma_mu: float
    lambda_: float

    trees: list[Tree]
    training_predicitons: np.ndarray
    fitted_sums: np.ndarray
    tree_sample: list[dict]
    terminals_data_cache: list[dict]
    internals_data_cache: list[dict]

    def __init__(self,
                 m=200,
                 alpha=0.95,
                 beta=2.0,
                 k=2.0,
                 nu=3.0,
                 q=0.90,
                 n_burn=200,
                 n_samples=1000,
                 move_distribution=[0.25, 0.25, 0.40, 0.10],
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
        self.y = np.asarray(y)
        self.y_scale = max(self.y) - min(self.y)
        if self.y_scale < 1e-12:
            self.y_scale = 1.0
        self.y_shift = (min(self.y)+max(self.y)) / 2.0
        self.y = (self.y - self.y_shift) / self.y_scale
        self.n, self.p = X.shape

        # Initializing trees.
        self.trees = [Tree(Node.terminal(0.0)) for _ in range(self.m)]

        # Initializing mu_ij|T_j prior ~ N(0, sigma_mu)
        self.sigma_mu = (0.5 / (self.k * np.sqrt(self.m)))**2
        # Initializing sigma prior.
        A_sigma = np.column_stack([self.X, np.ones(self.n)])
        sigma_linalg = np.linalg.lstsq(A_sigma, self.y, rcond=None)
        rss = np.sum((self.y - A_sigma @ sigma_linalg[0])**2)
        self.sigma = rss / (self.n - np.linalg.matrix_rank(A_sigma))
        self.lambda_ = (self.sigma / self.nu) * chi2.ppf(1-self.q, df=self.nu)

        # Initializing predictions.
        self.training_predicitons = np.zeros((self.n, self.m))
        self.fitted_sums = np.zeros(self.n)
        self.tree_sample = []
        self.terminals_data_cache = [{(): [i for i in range(self.n)]} for _ in range(self.m)]
        self.internals_data_cache = [{} for _ in range(self.m)]

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()
        
        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            self.tree_sample.append({"sample": copy.deepcopy(self.trees), "sigma": self.sigma})
    
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            predictions = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[i] += self.tree_sample[i]["sample"][j].predict(X)
            return self._inverse_transform_y(predictions.mean())
        else:
            predictions = np.zeros((X.shape[0], self.n_samples))
            for i in range(self.n_samples):
                for j in range(self.m):
                    predictions[:, i] += self.tree_sample[i]["sample"][j].predict(X)
            return self._inverse_transform_y(predictions.mean(axis=1))
    
    def _one_mcmc_iteration(self):
        for j in range(self.m):
            self._draw_tree(j)
            self._draw_mu(j)
            self._update_training_predictions()
            self._update_fitted_sums()
        self.sigma = self._draw_sigma()
    
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
        if u < min(1.0, mh_ratio):
            self.trees[j] = proposed_tree
            if move == "grow":
                old_data = self.terminals_data_cache[j][old_path]
                variable = proposed_tree.node_at(old_path).variable
                value = proposed_tree.node_at(old_path).value
                data_l = [r for r in old_data if self.X[r, variable] <= value]
                data_r = [r for r in old_data if self.X[r, variable] > value]
                self.terminals_data_cache[j][old_path + (0, )] = data_l
                self.terminals_data_cache[j][old_path + (1, )] = data_r
                self.internals_data_cache[j][old_path] = old_data
                self.terminals_data_cache[j].pop(old_path, None)
            elif move == "prune":
                new_data = self.internals_data_cache[j][old_path]
                path_l = old_path + (0, )
                path_r = old_path + (1, )
                self.terminals_data_cache[j][old_path] = new_data
                self.terminals_data_cache[j].pop(path_l, None)
                self.terminals_data_cache[j].pop(path_r, None)
                self.internals_data_cache[j].pop(old_path, None)
            elif move == "change" or move == "swap":
                new_internals_cache = {}
                new_terminals_cache = {}
                proposed_tree.rebuild_data_cache(old_path,
                                                 self.internals_data_cache[j][old_path],
                                                 self.X,
                                                 new_internals_cache,
                                                 new_terminals_cache)
                for path, rows in new_internals_cache.items():
                    self.internals_data_cache[j][path] = rows
                for path, rows in new_terminals_cache.items():
                    self.terminals_data_cache[j][path] = rows
            else:
                return

    def _propose_tree_grow(self, j: int):
        possible_paths = [(path, data) for path, data in self.terminals_data_cache[j].items() if len(data) > 1]
        relevant_data = [d for p, d in possible_paths]
        possible_paths = [p for p, d in possible_paths]
        
        value_counts = []
        for path_data in relevant_data:
            value_count_dict = {}
            for var in range(self.p):
                value_count_dict[var] = np.unique(self.X[path_data, var])
            value_counts.append(value_count_dict)
        data_mask = [any(len(v) > 1 for v in value_dict.values()) for value_dict in value_counts]
        possible_paths = [path for path, m in zip(possible_paths, data_mask) if m]
        relevant_data = [data for data, m in zip(relevant_data, data_mask) if m]
        value_counts = [v for v, m in zip(value_counts, data_mask) if m]

        if len(possible_paths) < 1:
            return None, None, None

        path_index = self.rng.choice(len(possible_paths))
        old_path = possible_paths[path_index]
        possible_variables = [k for k, v in value_counts[path_index].items() if len(v) > 1]
        variable = self.rng.choice(possible_variables)
        possible_values = value_counts[path_index][variable]
        value = self.rng.choice(possible_values[:-1])

        proposed_tree = self.trees[j].grow(path=possible_paths[path_index],
                                            variable=variable,
                                            value=value)
        log_transition_ratio = (np.log(self.move_distribution[0])
                                + np.log(len(possible_paths))
                                + np.log(len(possible_variables))
                                + np.log(len(possible_values)-1)
                                - np.log(self.move_distribution[1])
                                + np.log(len(proposed_tree.prunable_paths())))
        
        rows = relevant_data[path_index]
        rows_l = [r for r in rows if self.X[r, variable] <= value]
        rows_r = [r for r in rows if self.X[r, variable] > value]
        log_likelihood_ratio = (self._log_likelihood(j, rows_l)
                                + self._log_likelihood(j, rows_r)
                                - self._log_likelihood(j, rows))

        d = len(old_path)
        log_tree_ratio = (np.log(self.alpha) 
                            + 2.0*np.log(1 - self._p_split(d+1))
                            - np.log(((1+d)**self.beta)
                                    *len(possible_variables)
                                    *len(possible_values[:-1])))
        
        mh_ratio = (log_transition_ratio
                    + log_likelihood_ratio
                    + log_tree_ratio)
        return proposed_tree, mh_ratio, old_path
    
    def _propose_tree_prune(self, j: int):
        possible_paths = self.trees[j].prunable_paths()
        if len(possible_paths) < 1:
            return None, None, None
        old_path = possible_paths[self.rng.choice(len(possible_paths))]
        rows_l = self.terminals_data_cache[j][old_path + (0, )] 
        rows_r = self.terminals_data_cache[j][old_path + (1, )]
        rows = self.internals_data_cache[j][old_path]
        old_variable = self.trees[j].node_at(old_path).variable
        proposed_tree = self.trees[j].prune(old_path)
        b_ = (len([path for path in self.trees[j].terminal_paths()])
              - (len(rows_l) > 1)
              - (len(rows_r) > 1)
              + 1)
        value_count_dict = {}
        for var in range(self.p):
            value_count_dict[var] = np.unique(self.X[rows, var])    
        possible_variables = [k for k, v in value_count_dict.items() if len(v) > 1]
        p_ = len(possible_variables)
        possible_values = value_count_dict[old_variable]
        ni_ = len(possible_values) - 1

        log_transition_ratio = (np.log(self.move_distribution[0])
                                + np.log(len(possible_paths))
                                - np.log(self.move_distribution[1])
                                - np.log(b_)
                                - np.log(p_)
                                - np.log(ni_))
        
        log_likelihood_ratio = (self._log_likelihood(j, rows)
                                - self._log_likelihood(j, rows_l)
                                - self._log_likelihood(j, rows_r))
        
        d = len(old_path)
        log_tree_ratio = (np.log((1+d)**self.beta - self.alpha)
                          + np.log(p_)
                          + np.log(ni_)
                          - np.log(self.alpha)
                          - 2.0*np.log(1 - self._p_split(d+1)))
        
        mh_ratio = (log_transition_ratio
                    + log_likelihood_ratio
                    + log_tree_ratio)
        return proposed_tree, mh_ratio, old_path

    def _propose_tree_change(self, j: int):
        possible_paths = self.trees[j].internal_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        old_variable = self.trees[j].node_at(path).variable
        old_value = self.trees[j].node_at(path).value
        rows = self.internals_data_cache[j][path]
        value_count_dict = {}
        for var in range(self.p):
            if var == old_variable:
                if len(np.unique(self.X[rows, var])) <= 2:
                    continue
            value_count_dict[var] = np.unique(self.X[rows, var])    
        possible_variables = [k for k, v in value_count_dict.items() if len(v) > 1]
        p_ = len(possible_variables)
        if p_ < 1:
            return None, None, None
        new_variable = possible_variables[self.rng.choice(p_)]
        possible_values = [v for v in value_count_dict[new_variable] if v != old_value]
        ni_ = len(possible_values)
        if ni_ < 1:
            return None, None, None
        new_value = possible_values[self.rng.choice(ni_)]

        proposed_tree = self.trees[j].change(path, new_variable, new_value)
        old_ter_cache = {k: v for k, v in self.terminals_data_cache[j].items() if k[:len(path)] == path}
        old_int_cache = {k: v for k, v in self.internals_data_cache[j].items() if k[:len(path)] == path}
        new_ter_cache = {}
        new_int_cache = {}
        proposed_tree.rebuild_data_cache(path, rows, self.X, new_int_cache, new_ter_cache)
        if any([len(v)==0 for v in new_ter_cache.values()]):
            return None, None, None
        
        log_likelihood_ratio = 0.0
        for rows in new_ter_cache.values():
            log_likelihood_ratio = (log_likelihood_ratio
                                    + self._log_likelihood(j, rows))
        for rows in old_ter_cache.values():
            log_likelihood_ratio = (log_likelihood_ratio
                                    - self._log_likelihood(j, rows))
        
        log_tree_ratio = 0.0
        for int_path, int_rows in new_int_cache.items():
            var = proposed_tree.node_at(int_path).variable
            log_tree_ratio = (log_tree_ratio
                              + self._log_tree_prior_for_internal(var, len(int_path), int_rows))
        for ter_path in new_ter_cache.keys():
            log_tree_ratio = (log_tree_ratio
                              + self._log_tree_prior_for_terminal(len(ter_path)))
        for int_path, int_rows in old_int_cache.items():
            var = self.trees[j].node_at(int_path).variable
            log_tree_ratio = (log_tree_ratio
                              - self._log_tree_prior_for_internal(var, len(int_path), int_rows))
        for ter_path in old_ter_cache.keys():
            log_tree_ratio = (log_tree_ratio
                              - self._log_tree_prior_for_terminal(len(ter_path)))
            
        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_tree, mh_ratio, path

    def _propose_tree_swap(self, j: int):
        possible_paths = self.trees[j].swappable_paths()
        if not possible_paths:
            return None, None, None
        path = possible_paths[self.rng.choice(len(possible_paths))]
        rows = self.internals_data_cache[j][path]
        path_l = path + (0, )
        path_r = path + (1, )
        children = []
        if self.trees[j].node_at(path_l).is_internal():
            children.append(path_l)
        if self.trees[j].node_at(path_r).is_internal():
            children.append(path_r)
        if len(children) > 1:
            if self.trees[j].node_at(path_l) != self.trees[j].node_at(path_r):
                children = children[self.rng.choice(2)]
        
        proposed_tree = self.trees[j].swap(children[0])
        old_ter_cache = {k: v for k, v in self.terminals_data_cache[j].items() if k[:len(path)] == path}
        old_int_cache = {k: v for k, v in self.internals_data_cache[j].items() if k[:len(path)] == path}
        new_ter_cache = {}
        new_int_cache = {}
        proposed_tree.rebuild_data_cache(path, rows, self.X, new_int_cache, new_ter_cache)
        if any([len(v)==0 for v in new_ter_cache.values()]):
            return None, None, None
        
        log_likelihood_ratio = 0.0
        for rows in new_ter_cache.values():
            log_likelihood_ratio = (log_likelihood_ratio
                                    + self._log_likelihood(j, rows))
        for rows in old_ter_cache.values():
            log_likelihood_ratio = (log_likelihood_ratio
                                    - self._log_likelihood(j, rows))
        
        log_tree_ratio = 0.0
        for int_path, int_rows in new_int_cache.items():
            var = proposed_tree.node_at(int_path).variable
            log_tree_ratio = (log_tree_ratio
                              + self._log_tree_prior_for_internal(var, len(int_path), int_rows))
        for ter_path in new_ter_cache.keys():
            log_tree_ratio = (log_tree_ratio
                              + self._log_tree_prior_for_terminal(len(ter_path)))
        for int_path, int_rows in old_int_cache.items():
            var = self.trees[j].node_at(int_path).variable
            log_tree_ratio = (log_tree_ratio
                              - self._log_tree_prior_for_internal(var, len(int_path), int_rows))
        for ter_path in old_ter_cache.keys():
            log_tree_ratio = (log_tree_ratio
                              - self._log_tree_prior_for_terminal(len(ter_path)))
        
        mh_ratio = log_likelihood_ratio + log_tree_ratio
        return proposed_tree, mh_ratio, path


    def _log_likelihood(self, j: int, rows: list):
        residuals = self._partial_residuals(j)[rows].sum()
        denom = self.sigma + len(rows)*self.sigma_mu + 1e-12
        return (0.5*np.log(self.sigma/denom)
                + (self.sigma_mu*(residuals**2)/(2.0*self.sigma*denom)))

    def _p_split(self, d: int):
        return self.alpha / (1+d)**self.beta

    def _log_tree_prior_for_terminal(self, depth: int):
        return np.log(1 - self._p_split(depth))
    
    def _log_tree_prior_for_internal(self, variable: int, d: int, rows: list):
        value_count_dict = {}
        for var in range(self.p):
            value_count_dict[var] = np.unique(self.X[rows, var])    
        p_ = len([k for k, v in value_count_dict.items() if len(v) > 1])
        ni_ = len(value_count_dict[variable]) - 1.0
        return (np.log(self._p_split(d))
                - np.log(p_)
                - np.log(ni_))
    
    def _draw_mu(self, j: int):
        residuals = self._partial_residuals(j)
        rows = self.terminals_data_cache[j]
        for path in rows.keys():
            denom = len(rows[path])*self.sigma_mu + self.sigma
            mu = self.rng.normal(loc=(self.sigma_mu*residuals[rows[path]].sum())/denom,
                                 scale=np.sqrt(self.sigma*self.sigma_mu/denom))
            self.trees[j].node_at(path).mu = mu

    def _draw_sigma(self):
        sse = np.sum((self.y - self.fitted_sums)**2)
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.n),
                                    scale=2.0/(self.nu*self.lambda_ + sse))

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift
    
    def _partial_residuals(self, j):
        return self.y - self.fitted_sums + self.training_predicitons[:, j]
    
    def _update_training_predictions(self):
        for i in range(self.n):
            for j in range(self.m):
                self.training_predicitons[i, j] = self.trees[j].predict(self.X[i, :])

    def _update_fitted_sums(self):
        self.fitted_sums = self.training_predicitons.sum(axis=1)

    def _validate_trees(self):
        for j in range(self.m):
            self.trees[j]._validate()
            terminal_paths = self.trees[j].terminal_paths()
            if set(self.terminals_data_cache[j]) != set(terminal_paths):
                raise ValueError("Terminal nodes and data cache don't match.")
