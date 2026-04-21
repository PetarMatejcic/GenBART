#include "backfitting_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

BackfittingEngine::BackfittingEngine(DoubleArray X_in, int32_t m, uint64_t seed):
    X_owner_(std::move(X_in)),
    X_(nullptr),
    n_(0),
    p_(0),
    m_(m),
    rng_(seed) {

    if (m_ <= 0) {
        throw std::runtime_error("m must be positive.");
    }
    if (X_owner_.ndim() != 2) {
        throw std::runtime_error("X must be a 2D array.");
    }

    const auto buf = X_owner_.request();
    n_ = static_cast<int32_t>(X_owner_.shape(0));
    p_ = static_cast<int32_t>(X_owner_.shape(1));

    if (n_ <= 0 || p_ <= 0) {
        throw std::runtime_error("X must have positive shape.");
    }

    X_ = static_cast<const double*>(buf.ptr);
    if (X_ == nullptr) {
        throw std::runtime_error("X data pointer is null.");
    }

    build_root_rows_by_var();
}

void BackfittingEngine::build_root_rows_by_var() {
    root_rows_by_var_.assign(static_cast<size_t>(p_), {});

    for (int32_t v = 0; v < p_; ++v) {
        auto& ord = root_rows_by_var_[static_cast<size_t>(v)];
        ord.resize(static_cast<size_t>(n_));
        for (int32_t i = 0; i < n_; ++i) {
            ord[static_cast<size_t>(i)] = i;
        }

        std::stable_sort(
            ord.begin(),
            ord.end(),
            [this, v](int32_t a, int32_t b) {
                return X_[static_cast<size_t>(a) * p_ + v]
                     < X_[static_cast<size_t>(b) * p_ + v];
            }
        );
    }
}

void BackfittingEngine::initialize_root_forest() {
    forest_.clear();
    forest_.reserve(static_cast<size_t>(m_));

    for (int32_t j = 0; j < m_; ++j) {
        forest_.emplace_back(X_, n_, p_, root_rows_by_var_);
    }
}

void BackfittingEngine::backfitting_sweep(
    DoubleArray training_predictions,
    DoubleArray residuals,
    double sigma2,
    double sigma_mu2,
    double alpha,
    double beta,
    const std::array<double, 4>& move_distribution
) {
    validate_training_state_arrays(training_predictions);
    validate_residuals_array(residuals);

    auto tp = training_predictions.mutable_unchecked<2>();
    auto r = residuals.mutable_unchecked<1>();

    py::gil_scoped_release release;

    for (int32_t j = 0; j < m_; ++j) {
        for (int32_t i = 0; i < n_; ++i) {
            r(i) += tp(j, i);
        }

        draw_tree_impl(j, residuals, sigma2, sigma_mu2, alpha, beta, move_distribution);

        draw_mu_impl( j, residuals, sigma2, sigma_mu2);

        refresh_tree_training_predictions_impl( j, training_predictions, residuals);
    }
}

void BackfittingEngine::check_tree_index(int32_t j) const {
    if (j < 0 || j >= m_) {
        throw std::runtime_error("Tree index out of bounds.");
    }
}

void BackfittingEngine::validate_residuals_array(const DoubleArray& residuals) const {
    if (residuals.ndim() != 1) {
        throw std::runtime_error("residuals must be a 1D array.");
    }
    if (static_cast<int32_t>(residuals.shape(0)) != n_) {
        throw std::runtime_error("residuals has wrong length.");
    }
}

void BackfittingEngine::validate_training_state_arrays(
    const DoubleArray& training_predictions
) const {
    if (training_predictions.ndim() != 2) {
        throw std::runtime_error("training_predictions must be a 2D array.");
    }
    if (static_cast<int32_t>(training_predictions.shape(0)) != m_ ||
        static_cast<int32_t>(training_predictions.shape(1)) != n_) {
        throw std::runtime_error("training_predictions has wrong shape.");
    }
}

int BackfittingEngine::sample_move(const std::array<double, 4>& move_distribution) {
    double total = 0.0;
    for (double w : move_distribution) {
        if (w < 0.0) {
            throw std::runtime_error("move_distribution cannot contain negative probabilities.");
        }
        total += w;
    }

    if (std::abs(total - 1.0) > 1e-12) {
        throw std::runtime_error("move_distribution must sum to 1.");
    }

    std::discrete_distribution<int> dist(
        move_distribution.begin(),
        move_distribution.end()
    );
    return dist(rng_);
}

int BackfittingEngine::sample_swap_mode(const Tree& tree, int32_t parent_idx) {
    const Node& parent = tree.node(parent_idx);
    const Node& left_child = tree.node(parent.left);
    const Node& right_child = tree.node(parent.right);

    const bool left_internal = left_child.is_internal();
    const bool right_internal = right_child.is_internal();

    if (!left_internal && !right_internal) {
        throw std::runtime_error("sample_swap_mode called on non-swappable node.");
    }

    if (!left_internal) {
        return SWAP_RIGHT;
    }
    if (!right_internal) {
        return SWAP_LEFT;
    }

    const bool same_rule =
        (left_child.variable == right_child.variable) &&
        (left_child.value == right_child.value);

    if (same_rule) {
        return SWAP_BOTH;
    }

    std::uniform_int_distribution<int> dist(0, 1);
    return dist(rng_) == 0 ? SWAP_LEFT : SWAP_RIGHT;
}

std::optional<std::pair<int32_t, int32_t>> BackfittingEngine::sample_uniform_change_rule(const Tree& tree, int32_t node_idx) {
    const Node& node = tree.node(node_idx);
    const auto& vars = node.valid_vars;

    if (vars.empty()) return std::nullopt;

    std::vector<int32_t> counts;
    counts.reserve(vars.size());

    int64_t total_rules = 0;
    for (int32_t var : vars) {
        const int32_t eta = node.eta_by_var.at(static_cast<size_t>(var));
        counts.push_back(eta);
        total_rules += eta;
    }

    if (total_rules <= 1) return std::nullopt;

    auto it = std::find(vars.begin(), vars.end(), node.variable);
    if (it == vars.end()) throw std::runtime_error("Current split variable is not in valid_vars.");

    const int32_t cur_var_pos = static_cast<int32_t>(std::distance(vars.begin(), it));
    const int32_t cur_split_pos = tree.split_pos_of_value(
        node.rows_by_var.at(static_cast<size_t>(node.variable)),
        node.variable,
        node.value
    );

    int64_t cur_global = cur_split_pos;
    for (int32_t i = 0; i < cur_var_pos; ++i) {
        cur_global += counts[static_cast<size_t>(i)];
    }

    std::uniform_int_distribution<int64_t> dist(0, total_rules - 2);
    int64_t u = dist(rng_);
    if (u >= cur_global) ++u;

    int64_t prefix = 0;
    for (size_t pos = 0; pos < vars.size(); ++pos) {
        const int64_t next_prefix = prefix + counts[pos];
        if (u < next_prefix) {
            const int32_t var = vars[pos];
            const int32_t split_idx = static_cast<int32_t>(u - prefix);
            return std::make_pair(var, split_idx);
        }
        prefix = next_prefix;
    }
    throw std::runtime_error("Failed to sample change rule.");
}

void BackfittingEngine::collect_terminal_stats(
    const Tree& tree,
    int32_t node_idx,
    const DoubleArray& residuals,
    std::vector<TerminalStat>& out
) const {
    const auto r = residuals.unchecked<1>();
    const Node& node = tree.node(node_idx);

    if (node.is_terminal()) {
        double sum_r = 0.0;
        for (int32_t row : node.rows) {
            sum_r += r(row);
        }
        out.push_back(TerminalStat{
            static_cast<int32_t>(node.rows.size()),
            sum_r
        });
        return;
    }

    collect_terminal_stats(tree, node.left, residuals, out);
    collect_terminal_stats(tree, node.right, residuals, out);
}

void BackfittingEngine::collect_internal_stats(
    const Tree& tree,
    int32_t node_idx,
    std::vector<InternalStat>& out
) const {
    const Node& node = tree.node(node_idx);

    if (node.is_terminal()) {
        return;
    }

    out.push_back(InternalStat{
        static_cast<int32_t>(node.valid_vars.size()),
        node.eta_by_var.at(static_cast<size_t>(node.variable))
    });

    collect_internal_stats(tree, node.left, out);
    collect_internal_stats(tree, node.right, out);
}

double BackfittingEngine::p_split(int32_t depth, double alpha, double beta) const {
    return alpha / std::pow(1.0 + static_cast<double>(depth), beta);
}

double BackfittingEngine::log_likelihood_ratio(
    const std::vector<TerminalStat>& new_terminals,
    const std::vector<TerminalStat>& old_terminals,
    double sigma2,
    double sigma_mu2
) const {
    auto contribution = [&](const TerminalStat& s) -> double {
        const double denom = sigma2 + static_cast<double>(s.n) * sigma_mu2;
        const double sse = s.sum_r * s.sum_r;

        return 0.5 * std::log(sigma2 / denom)
             + (sigma_mu2 * sse) / (2.0 * sigma2 * denom);
    };

    double ratio = 0.0;
    for (const auto& s : new_terminals) {
        ratio += contribution(s);
    }
    for (const auto& s : old_terminals) {
        ratio -= contribution(s);
    }
    return ratio;
}

double BackfittingEngine::log_prior_ratio(
    const std::vector<InternalStat>& new_internals,
    const std::vector<InternalStat>& old_internals
) const {
    auto contribution = [](const InternalStat& s) -> double {
        return -std::log(static_cast<double>(s.b))
               -std::log(static_cast<double>(s.eta));
    };

    double ratio = 0.0;
    for (const auto& s : new_internals) {
        ratio += contribution(s);
    }
    for (const auto& s : old_internals) {
        ratio -= contribution(s);
    }
    return ratio;
}

void BackfittingEngine::draw_mu(
    int32_t j,
    const DoubleArray& residuals,
    double sigma2,
    double sigma_mu2
) {
    check_tree_index(j);
    validate_residuals_array(residuals);
    draw_mu_impl(j, residuals, sigma2, sigma_mu2);
}

void BackfittingEngine::draw_mu_impl(
    int32_t j,
    const DoubleArray& residuals,
    double sigma2,
    double sigma_mu2
) {
    auto r = residuals.unchecked<1>();
    Tree& tree = forest_[static_cast<size_t>(j)];
    const auto terminals = tree.terminal_nodes(false);

    for (int32_t node_idx : terminals) {
        Node& node = tree.node(node_idx);

        double sum_r = 0.0;
        for (int32_t row : node.rows) {
            sum_r += r(row);
        }

        const double denom = static_cast<double>(node.rows.size()) * sigma_mu2 + sigma2;
        const double mean = (sigma_mu2 * sum_r) / denom;
        const double var = (sigma2 * sigma_mu2) / denom;

        std::normal_distribution<double> dist(mean, std::sqrt(var));
        node.mu = static_cast<double>(dist(rng_));
    }
}

void BackfittingEngine::refresh_tree_training_predictions(
    int32_t j,
    DoubleArray training_predictions,
    DoubleArray residuals
) {
    check_tree_index(j);
    validate_training_state_arrays(training_predictions);
    refresh_tree_training_predictions_impl(j, training_predictions, residuals);
}

void BackfittingEngine::refresh_tree_training_predictions_impl(
    int32_t j,
    DoubleArray training_predictions,
    DoubleArray residuals
) {
    auto tp = training_predictions.mutable_unchecked<2>();
    auto r = residuals.mutable_unchecked<1>();

    const Tree& tree = forest_[static_cast<size_t>(j)];
    const auto terminals = tree.terminal_nodes(false);

    for (int32_t node_idx : terminals) {
        const Node& node = tree.node(node_idx);
        const double mu = static_cast<double>(node.mu);
        for (int32_t row : node.rows) {
            tp(j, row) = mu;
            r(row) -= mu;
        }
    }
}

py::tuple BackfittingEngine::serialize_tree(int32_t j) const {
    check_tree_index(j);

    std::vector<int32_t> variable, left, right;
    std::vector<double> value, mu;

    forest_[static_cast<size_t>(j)].serialize(variable, value, left, right, mu);

    py::array_t<int32_t> variable_arr(variable.size());
    py::array_t<double> value_arr(value.size());
    py::array_t<int32_t> left_arr(left.size());
    py::array_t<int32_t> right_arr(right.size());
    py::array_t<double> mu_arr(mu.size());

    std::memcpy(variable_arr.mutable_data(), variable.data(), variable.size() * sizeof(int32_t));
    std::memcpy(value_arr.mutable_data(), value.data(), value.size() * sizeof(double));
    std::memcpy(left_arr.mutable_data(), left.data(), left.size() * sizeof(int32_t));
    std::memcpy(right_arr.mutable_data(), right.data(), right.size() * sizeof(int32_t));
    std::memcpy(mu_arr.mutable_data(), mu.data(), mu.size() * sizeof(double));

    return py::make_tuple(
        std::move(variable_arr),
        std::move(value_arr),
        std::move(left_arr),
        std::move(right_arr),
        std::move(mu_arr)
    );
}

py::tuple BackfittingEngine::serialize_forest() const {
    std::vector<int32_t> variable_all, left_all, right_all;
    std::vector<double> value_all, mu_all;
    std::vector<int64_t> tree_offset;
    tree_offset.reserve(static_cast<size_t>(m_) + 1);
    tree_offset.push_back(0);

    int64_t base = 0;

    for (int32_t j = 0; j < m_; ++j) {
        std::vector<int32_t> variable, left, right;
        std::vector<double> value, mu;

        forest_[static_cast<size_t>(j)].serialize(variable, value, left, right, mu);

        const int64_t n_nodes = static_cast<int64_t>(variable.size());

        variable_all.insert(variable_all.end(), variable.begin(), variable.end());
        value_all.insert(value_all.end(), value.begin(), value.end());
        mu_all.insert(mu_all.end(), mu.begin(), mu.end());

        for (int32_t x : left) {
            left_all.push_back(x >= 0 ? static_cast<int32_t>(x + base) : -1);
        }
        for (int32_t x : right) {
            right_all.push_back(x >= 0 ? static_cast<int32_t>(x + base) : -1);
        }

        base += n_nodes;
        tree_offset.push_back(base);
    }

    py::array_t<int32_t> variable_arr(variable_all.size());
    py::array_t<double> value_arr(value_all.size());
    py::array_t<int32_t> left_arr(left_all.size());
    py::array_t<int32_t> right_arr(right_all.size());
    py::array_t<double> mu_arr(mu_all.size());
    py::array_t<int64_t> tree_offset_arr(tree_offset.size());

    std::memcpy(variable_arr.mutable_data(), variable_all.data(),
                variable_all.size() * sizeof(int32_t));
    std::memcpy(value_arr.mutable_data(), value_all.data(),
                value_all.size() * sizeof(double));
    std::memcpy(left_arr.mutable_data(), left_all.data(),
                left_all.size() * sizeof(int32_t));
    std::memcpy(right_arr.mutable_data(), right_all.data(),
                right_all.size() * sizeof(int32_t));
    std::memcpy(mu_arr.mutable_data(), mu_all.data(),
                mu_all.size() * sizeof(double));
    std::memcpy(tree_offset_arr.mutable_data(), tree_offset.data(),
                tree_offset.size() * sizeof(int64_t));

    return py::make_tuple(
        std::move(variable_arr),
        std::move(value_arr),
        std::move(left_arr),
        std::move(right_arr),
        std::move(mu_arr),
        std::move(tree_offset_arr)
    );
}

void BackfittingEngine::validate_tree(int32_t j) const {
    check_tree_index(j);
    forest_[static_cast<size_t>(j)].validate();
}

void BackfittingEngine::validate_forest() const {
    for (int32_t j = 0; j < m_; ++j) {
        forest_[static_cast<size_t>(j)].validate();
    }
}

bool BackfittingEngine::draw_tree(
    int32_t j,
    const DoubleArray& residuals,
    double sigma2,
    double sigma_mu2,
    double alpha,
    double beta,
    const std::array<double, 4>& move_distribution
) {
    check_tree_index(j);
    validate_residuals_array(residuals);
    return draw_tree_impl(j, residuals, sigma2, sigma_mu2, alpha, beta, move_distribution);
}

bool BackfittingEngine::draw_tree_impl(
    int32_t j,
    const DoubleArray& residuals,
    double sigma2,
    double sigma_mu2,
    double alpha,
    double beta,
    const std::array<double, 4>& move_distribution
) {
    Tree& tree = forest_[static_cast<size_t>(j)];

    auto depth_of_node = [&](int32_t node_idx) -> int32_t {
        int32_t depth = 0;
        int32_t cur = node_idx;
        while (tree.node(cur).parent != -1) {
            cur = tree.node(cur).parent;
            ++depth;
        }
        return depth;
    };

    auto log_accept_draw = [&]() -> double {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double u = std::max(dist(rng_), std::numeric_limits<double>::min());
        return std::log(u);
    };

    const int move = sample_move(move_distribution);

    if (move == GROW) {
        const auto candidates = tree.terminal_nodes(true);
        if (candidates.empty()) return false;

        std::uniform_int_distribution<int32_t> node_dist(
            0, static_cast<int32_t>(candidates.size()) - 1
        );
        const int32_t node_idx = candidates[static_cast<size_t>(node_dist(rng_))];
        const Node& node = tree.node(node_idx);

        const int32_t b_ = static_cast<int32_t>(node.valid_vars.size());
        if (b_ == 0) return false;

        std::uniform_int_distribution<int32_t> var_dist(0, b_ - 1);
        const int32_t variable = node.valid_vars[static_cast<size_t>(var_dist(rng_))];

        const int32_t eta_ = node.eta_by_var.at(static_cast<size_t>(variable));
        if (eta_ <= 0) return false;

        std::uniform_int_distribution<int32_t> split_dist(0, eta_ - 1);
        const int32_t split_idx = split_dist(rng_);

        auto proposal_opt = tree.propose_grow(node_idx, variable, split_idx);
        if (!proposal_opt.has_value()) return false;
        const GrowProposalLite& proposal = *proposal_opt;

        const int32_t p_old = static_cast<int32_t>(candidates.size());
        const int32_t n_prunable_live =
        static_cast<int32_t>(tree.prunable_nodes().size());

        int32_t delta = 1;
        const Node& live_node = tree.node(node_idx);
        if (live_node.parent != -1) {
            const Node& parent = tree.node(live_node.parent);
            const int32_t sibling_idx = (parent.left == node_idx) ? parent.right : parent.left;
            const bool sibling_terminal = tree.node(sibling_idx).is_terminal();
            if (sibling_terminal) {
                delta -= 1;
            }
        }

        const int32_t n_prunable_new = std::max<int32_t>(1, n_prunable_live + delta);

        const double log_transition_ratio =
            std::log(move_distribution[PRUNE])
            + std::log(static_cast<double>(p_old))
            + std::log(static_cast<double>(b_))
            + std::log(static_cast<double>(eta_))
            - std::log(move_distribution[GROW])
            - std::log(static_cast<double>(n_prunable_new));

        const auto r = residuals.unchecked<1>();

        auto terminal_stat_from_rows = [&](const std::vector<int32_t>& rows) -> TerminalStat {
            double sum_r = 0.0;
            for (int32_t row : rows) {
                sum_r += r(row);
            }
            return TerminalStat{
                static_cast<int32_t>(rows.size()),
                sum_r
            };
        };

        std::vector<TerminalStat> old_terminals{
            terminal_stat_from_rows(node.rows)
        };

        std::vector<TerminalStat> new_terminals{
            terminal_stat_from_rows(proposal.left_by_var[0]),
            terminal_stat_from_rows(proposal.right_by_var[0])
        };

        const double log_lik_ratio = log_likelihood_ratio(
            new_terminals,
            old_terminals,
            sigma2,
            sigma_mu2
        );

        const int32_t d = depth_of_node(node_idx);
        const double log_tree_ratio = std::log(alpha)
                                      + 2.0 * std::log(1.0 - p_split(d + 1, alpha, beta))
                                      - std::log(std::pow(1.0 + static_cast<double>(d), beta) - alpha)
                                      - std::log(static_cast<double>(b_))
                                      - std::log(static_cast<double>(eta_));

        const double mh_ratio = log_transition_ratio + log_lik_ratio + log_tree_ratio;

        if (log_accept_draw() < std::min(0.0, mh_ratio)) {
            tree.apply_grow(proposal);
            return true;
        }
        return false;
    }

    if (move == PRUNE) {
        const auto candidates = tree.prunable_nodes();
        if (candidates.empty()) return false;

        std::uniform_int_distribution<int32_t> node_dist(
            0, static_cast<int32_t>(candidates.size()) - 1
        );
        const int32_t node_idx = candidates[static_cast<size_t>(node_dist(rng_))];
        const Node& node = tree.node(node_idx);
        const Node& left_child = tree.node(node.left);
        const Node& right_child = tree.node(node.right);

        auto proposal_opt = tree.propose_prune(node_idx, 0.0f);
        if (!proposal_opt.has_value()) return false;
        const PruneProposalLite& proposal = *proposal_opt;

        const int32_t b_ =
            static_cast<int32_t>(tree.terminal_nodes(true).size()) -
            (left_child.rows.size() > 1 ? 1 : 0) -
            (right_child.rows.size() > 1 ? 1 : 0) + 1;

        const int32_t p_ = static_cast<int32_t>(node.valid_vars.size());
        const int32_t eta_ = node.eta_by_var.at(static_cast<size_t>(node.variable));

        const double log_transition_ratio = std::log(move_distribution[GROW])
                                            + std::log(static_cast<double>(candidates.size()))
                                            - std::log(move_distribution[PRUNE])
                                            - std::log(static_cast<double>(b_))
                                            - std::log(static_cast<double>(p_))
                                            - std::log(static_cast<double>(eta_));

        const auto r = residuals.unchecked<1>();

        auto terminal_stat_from_rows = [&](const std::vector<int32_t>& rows) -> TerminalStat {
            double sum_r = 0.0;
            for (int32_t row : rows) {
                sum_r += r(row);
            }
            return TerminalStat{
                static_cast<int32_t>(rows.size()),
                sum_r
            };
        };

        std::vector<TerminalStat> new_terminals{
            terminal_stat_from_rows(node.rows)
        };

        std::vector<TerminalStat> old_terminals{
            terminal_stat_from_rows(left_child.rows),
            terminal_stat_from_rows(right_child.rows)
        };

        const double log_lik_ratio = log_likelihood_ratio(
            old_terminals,
            new_terminals,
            sigma2,
            sigma_mu2
        );

        const int32_t d = depth_of_node(node_idx);
        const double log_tree_ratio =
            std::log(std::pow(1.0 + static_cast<double>(d), beta) - alpha) +
            std::log(static_cast<double>(p_)) +
            std::log(static_cast<double>(eta_)) -
            std::log(alpha) -
            2.0 * std::log(1.0 - p_split(d + 1, alpha, beta));

        const double mh_ratio = log_transition_ratio + log_lik_ratio + log_tree_ratio;

        if (log_accept_draw() < std::min(0.0, mh_ratio)) {
            tree.apply_prune(proposal);
            return true;
        }
        return false;
    }

    if (move == CHANGE) {
        const auto candidates = tree.internal_nodes();
        if (candidates.empty()) return false;

        std::uniform_int_distribution<int32_t> node_dist(
            0, static_cast<int32_t>(candidates.size()) - 1
        );
        const int32_t node_idx = candidates[static_cast<size_t>(node_dist(rng_))];

        auto rule_opt = sample_uniform_change_rule(tree, node_idx);
        if (!rule_opt.has_value()) return false;

        auto proposal_opt = tree.propose_change(node_idx, rule_opt->first, rule_opt->second);
        if (!proposal_opt.has_value()) return false;
        const ProposalSubtree& proposal = *proposal_opt;

        std::vector<TerminalStat> old_terminals;
        collect_terminal_stats(tree, node_idx, residuals, old_terminals);

        std::vector<InternalStat> old_internals;
        collect_internal_stats(tree, node_idx, old_internals);

        std::vector<TerminalStat> new_terminals;
        collect_terminal_stats(proposal.subtree, proposal.subtree.root(), residuals, new_terminals);

        std::vector<InternalStat> new_internals;
        collect_internal_stats(proposal.subtree, proposal.subtree.root(), new_internals);

        const double log_lik_ratio = log_likelihood_ratio(
            new_terminals,
            old_terminals,
            sigma2,
            sigma_mu2
        );

        const double log_prior = log_prior_ratio(
            new_internals,
            old_internals
        );

        const double mh_ratio = log_lik_ratio + log_prior;

        if (log_accept_draw() < std::min(0.0, mh_ratio)) {
            tree.apply_rebuilt_subtree_same_shape(node_idx, proposal.subtree);
            return true;
        }
        return false;
    }

    if (move == SWAP) {
        const auto candidates = tree.swappable_nodes();
        if (candidates.empty()) return false;

        std::uniform_int_distribution<int32_t> node_dist(
            0, static_cast<int32_t>(candidates.size()) - 1
        );
        const int32_t node_idx = candidates[static_cast<size_t>(node_dist(rng_))];
        const int mode = sample_swap_mode(tree, node_idx);

        auto proposal_opt = tree.propose_swap(node_idx, mode);
        if (!proposal_opt.has_value()) return false;
        const ProposalSubtree& proposal = *proposal_opt;

        std::vector<TerminalStat> old_terminals;
        collect_terminal_stats(tree, node_idx, residuals, old_terminals);

        std::vector<InternalStat> old_internals;
        collect_internal_stats(tree, node_idx, old_internals);

        std::vector<TerminalStat> new_terminals;
        collect_terminal_stats(proposal.subtree, proposal.subtree.root(), residuals, new_terminals);

        std::vector<InternalStat> new_internals;
        collect_internal_stats(proposal.subtree, proposal.subtree.root(), new_internals);

        const double log_lik_ratio = log_likelihood_ratio(
            new_terminals,
            old_terminals,
            sigma2,
            sigma_mu2
        );

        const double log_prior = log_prior_ratio(
            new_internals,
            old_internals
        );

        const double mh_ratio = log_lik_ratio + log_prior;

        if (log_accept_draw() < std::min(0.0, mh_ratio)) {
            tree.apply_rebuilt_subtree_same_shape(node_idx, proposal.subtree);
            return true;
        }
        return false;
    }

    throw std::runtime_error("Unknown move type.");
}

void bind_backfitting_engine(py::module_& m) {
    py::class_<BackfittingEngine>(m, "_BackfittingEngine")
        .def(
            py::init<
                BackfittingEngine::DoubleArray,
                int32_t,
                uint64_t
            >(),
            py::arg("X"),
            py::arg("m"),
            py::arg("seed") = 0
        )
        .def("initialize_root_forest", &BackfittingEngine::initialize_root_forest)
        .def(
            "backfitting_sweep",
            &BackfittingEngine::backfitting_sweep,
            py::arg("training_predictions"),
            py::arg("residuals"),
            py::arg("sigma2"),
            py::arg("sigma_mu2"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("move_distribution")
        )
        .def(
            "draw_tree",
            &BackfittingEngine::draw_tree,
            py::arg("j"),
            py::arg("residuals"),
            py::arg("sigma2"),
            py::arg("sigma_mu2"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("move_distribution")
        )
        .def(
            "draw_mu",
            &BackfittingEngine::draw_mu,
            py::arg("j"),
            py::arg("residuals"),
            py::arg("sigma2"),
            py::arg("sigma_mu2")
        )
        .def(
            "refresh_tree_training_predictions",
            &BackfittingEngine::refresh_tree_training_predictions,
            py::arg("j"),
            py::arg("training_predictions"),
            py::arg("fitted_sums")
        )
        .def("serialize_tree", &BackfittingEngine::serialize_tree, py::arg("j"))
        .def("serialize_forest", &BackfittingEngine::serialize_forest)
        .def("validate_tree", &BackfittingEngine::validate_tree, py::arg("j"))
        .def("validate_forest", &BackfittingEngine::validate_forest)
        .def("n", &BackfittingEngine::n)
        .def("p", &BackfittingEngine::p)
        .def("m", &BackfittingEngine::m);
}