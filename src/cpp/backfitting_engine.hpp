#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tree.hpp"

namespace py = pybind11;

struct TerminalStat {
    int32_t n;
    double sum_r;
};

struct InternalStat {
    int32_t b;
    int32_t eta;
};

/**
 * @brief Coordinates Bayesian backfitting updates for a BART forest.
 *
 * BackfittingEngine owns the live collection of trees used during MCMC. It
 * updates each tree conditionally on the current partial residuals, computes
 * Metropolis-Hastings acceptance quantities, draws terminal-node means, and
 * serializes the live forest for posterior storage.
 */
class BackfittingEngine {
public:
    using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    BackfittingEngine(
        DoubleArray X_in,
        int32_t m,
        uint64_t seed = 0
    );

    void initialize_root_forest();

    void backfitting_sweep(
        DoubleArray residuals,
        double sigma2,
        double sigma_mu2,
        double alpha,
        double beta,
        const std::array<double, 4>& move_distribution
    );

    //draw_tree, draw_mu, refresh_training_predictions are not used in BaseBart.
    //They remain for debuging and testing.
    bool draw_tree(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2,
        double alpha,
        double beta,
        const std::array<double, 4>& move_distribution
    );

    void draw_mu(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2
    );

    py::tuple serialize_tree(int32_t j) const;
    py::tuple serialize_forest() const;

    void validate_tree(int32_t j) const;
    void validate_forest() const;

    int32_t n() const noexcept { return n_; }
    int32_t p() const noexcept { return p_; }
    int32_t m() const noexcept { return m_; }

    void test_apply_tree_to_residuals(int32_t j, DoubleArray residuals, double sign);
    void test_draw_mu_and_subtract(int32_t j, DoubleArray residuals, double sigma2, double sigma_mu2);

private:
    enum MoveKind : int {
        GROW = 0,
        PRUNE = 1,
        CHANGE = 2,
        SWAP = 3
    };

    enum SwapMode : int {
        SWAP_LEFT = 0,
        SWAP_RIGHT = 1,
        SWAP_BOTH = 2
    };

    DoubleArray X_owner_;
    const double* X_ = nullptr;
    int32_t n_ = 0;
    int32_t p_ = 0;
    int32_t m_ = 0;

    std::vector<std::vector<int32_t>> root_rows_by_var_;
    std::vector<Tree> forest_;
    std::mt19937_64 rng_;

    void build_root_rows_by_var();

    void check_tree_index(int32_t j) const;
    void validate_residuals_array(const DoubleArray& residuals) const;

    bool draw_tree_impl(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2,
        double alpha,
        double beta,
        const std::array<double, 4>& move_distribution
    );
    void draw_mu_impl(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2
    );
    void draw_mu_and_subtract_impl(
        int32_t j,
        DoubleArray residuals,
        double sigma2,
        double sigma_mu2
    );
    void apply_tree_to_residuals_impl(
        int32_t j,
        DoubleArray residuals,
        double sign
    );

    int sample_move(const std::array<double, 4>& move_distribution);
    int sample_swap_mode(const Tree& tree, int32_t parent_idx);

    std::optional<std::pair<int32_t, int32_t>> sample_uniform_change_rule(const Tree& tree, int32_t node_idx);

    void collect_terminal_stats(
        const Tree& tree,
        int32_t node_idx,
        const DoubleArray& residuals,
        std::vector<TerminalStat>& out
    ) const;
    void collect_internal_stats(
        const Tree& tree,
        int32_t node_idx,
        std::vector<InternalStat>& out
    ) const;

    void collect_terminal_stats_workspace(
        const SameShapeWorkspace& ws,
        const DoubleArray& residuals,
        std::vector<TerminalStat>& out
    ) const;
    void collect_internal_stats_workspace(
        const SameShapeWorkspace& ws,
        std::vector<InternalStat>& out
    ) const;

    double log_likelihood_ratio(
        const std::vector<TerminalStat>& new_terminals,
        const std::vector<TerminalStat>& old_terminals,
        double sigma2,
        double sigma_mu2
    ) const;

    double log_prior_ratio(
        const std::vector<InternalStat>& new_internals,
        const std::vector<InternalStat>& old_internals
    ) const;

    double p_split(int32_t depth, double alpha, double beta) const;
};

void bind_backfitting_engine(py::module_& m);