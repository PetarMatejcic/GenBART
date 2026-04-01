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
    tree_sample: list[Tree]
    terminal_node_data_cache: list[dict]

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
        self.sigma_mu = 0.5 / (self.k * np.sqrt(self.m))
        # Initializing sigma prior.
        A_sigma = np.column_stack([self.X, np.ones(self.n)])
        sigma_linalg = np.linalg.lstsq(A_sigma, self.y, rcond=None)
        rss = np.sum((self.y - A_sigma @ sigma_linalg[0])**2)
        self.sigma = np.sqrt(rss) / (self.n - self.p)
        self.lambda_ = (self.sigma / self.nu) * chi2.ppf(1-self.q, df=self.nu)

        # Initializing predictions.
        self.training_predicitons = np.zeros((self.n, self.m))
        self.fitted_sums = np.zeros(self.n)
        self.tree_sample = []
        self.terminal_node_data_cache = [{(): [i for i in range(self.n)]} for _ in range(self.m)]

        for _ in range(self.n_burn):
            self._one_mcmc_iteration()
        
        for _ in range(self.n_samples):
            self._one_mcmc_iteration()
            self.tree_sample.append({"sample": copy.deepcopy(self.trees), "sigma": self.sigma})
    
    def _one_mcmc_iteration(self):
        for j in range(self.m):
            self._draw_tree(j)
            self._draw_mu(j)
        self._draw_sigma()
    
    def _draw_tree(self, j: int):
        move = self.rng.choice(["grow", "prune", "change", "swap"],
                               p=self.move_distribution)
        if move == "grow":
            proposed_tree, mh_ratio, old_path = self._propose_tree_grow(j)
            if proposed_tree is None:
                return
        else:
            return

        u = np.log(self.rng.uniform())
        if u < mh_ratio:
            self.trees[j] = proposed_tree
            if move == "grow":
                old_data = self.terminal_node_data_cache[j][old_path]
                variable = proposed_tree.node_at(old_path).variable
                value = proposed_tree.node_at(old_path).value
                data_l = [r for r in old_data if self.X[r, variable] <= value]
                data_r = [r for r in old_data if self.X[r, variable] > value]
                self.terminal_node_data_cache[j][old_path + (0, )] = data_l
                self.terminal_node_data_cache[j][old_path + (1, )] = data_r
                self.terminal_node_data_cache[j].pop(old_path, None)
            else:
                pass

    def _propose_tree_grow(self, j: int):
        possible_paths = [(path, data) for path, data in self.terminal_node_data_cache[j].items() if len(data) > 1]
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
        log_likelihood_ratio = (self._log_likelihood_ratio(j, rows_l)
                                + self._log_likelihood_ratio(j, rows_r)
                                - self._log_likelihood_ratio(j, rows))

        d = len(old_path)
        log_tree_ratio = (np.log(self.alpha) 
                            + 2.0*np.log(1 - self.alpha/((2+d)**self.beta))
                            - np.log(((1+d)**self.beta)
                                    *len(possible_variables)
                                    *len(possible_values[:-1])))
        
        mh_ratio = (log_transition_ratio
                    + log_likelihood_ratio
                    + log_tree_ratio)

        return proposed_tree, mh_ratio, old_path
    
    def _log_likelihood_ratio(self, j: int, rows: list):
        residuals = self._partial_residuals(j)[rows].sum()
        denom = self.sigma + len(rows)*self.sigma_mu
        return (0.5*np.log(self.sigma/denom)
                + (self.sigma_mu*(residuals**2)/(2*self.sigma*denom)))
    
    def _draw_mu(self, j: int):
        residuals = self._partial_residuals(j)
        rows = self.terminal_node_data_cache[j]
        for path in rows.keys():
            denom = len(rows[path])*self.sigma_mu + self.sigma
            mu = self.rng.normal(loc=(self.sigma_mu*residuals[rows[path]].sum())/denom,
                                 scale=np.sqrt(self.sigma*self.sigma_mu/denom))
            self.trees[j].node_at(path).mu = mu

    def _draw_sigma(self):
        return 1.0 / self.rng.gamma(shape=0.5*(self.nu+self.lambda_),
                                    scale=0.5*(self.nu*self.lambda_ + (self.y - self.fitted_sums).sum()**2))

    def _inverse_transform_y(self, v: np.ndarray):
        return (v * self.y_scale) + self.y_shift
    
    def _partial_residuals(self, j):
        return self.y - self.fitted_sums + self.training_predicitons[:, j]

    def _validate_trees(self):
        for j in range(self.m):
            self.trees[j]._validate()
            terminal_paths = self.trees[j].terminal_paths()
            if set(self.terminal_node_data_cache[j]) != set(terminal_paths):
                raise ValueError("Terminal nodes are not up to date")
