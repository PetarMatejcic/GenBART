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

class BackfittingEngine {
public:
    using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    BackfittingEngine(
        DoubleArray X_in,
        int32_t m,
        uint64_t seed = 0
    );

    void initialize_root_forest();

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

    void refresh_tree_training_predictions(
        int32_t j,
        DoubleArray training_predictions,
        DoubleArray fitted_sums
    );

    py::tuple serialize_tree(int32_t j) const;

    void validate_tree(int32_t j) const;
    void validate_forest() const;

    int32_t n() const noexcept { return n_; }
    int32_t p() const noexcept { return p_; }
    int32_t m() const noexcept { return m_; }

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
    void validate_training_state_arrays(const DoubleArray& training_predictions, const DoubleArray& fitted_sums) const;

    int sample_move(const std::array<double, 4>& move_distribution);
    int sample_swap_mode(const Tree& tree, int32_t parent_idx);

    std::optional<std::pair<int32_t, int32_t>>
    sample_uniform_change_rule(const Tree& tree, int32_t node_idx);

    void collect_terminal_rows(const Tree& tree, int32_t node_idx, std::vector<std::vector<int32_t>>& out) const;

    void collect_internal_nodes(const Tree& tree, int32_t node_idx, std::vector<int32_t>& out) const;

    double p_split(int32_t depth, double alpha, double beta) const;

    double log_likelihood_ratio(
        const DoubleArray& residuals,
        const std::vector<std::vector<int32_t>>& new_terminals,
        const std::vector<std::vector<int32_t>>& old_terminals,
        double sigma2,
        double sigma_mu2
    ) const;

    double log_tree_prior_for_internal(
        const Tree& tree,
        int32_t node_idx
    ) const;

    double log_prior_ratio(
        const Tree& proposed_tree,
        const std::vector<int32_t>& proposed_internals,
        const Tree& live_tree,
        const std::vector<int32_t>& old_internals
    ) const;
};

void bind_backfitting_engine(py::module_& m);