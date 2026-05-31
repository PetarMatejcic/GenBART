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

    /**
     * @brief Construct a backfitting engine for a BART ensemble.
     *
     * Owns the training feature matrix, initializes shared sorted-row state, and
     * manages the live forest used during MCMC backfitting.
     *
     * @param X Input feature matrix with shape `(n, p)`.
     * @param m Number of trees in the live ensemble.
     * @param seed Random seed used by the C++ proposal and parameter samplers.
     *
     * @throws std::runtime_error If `m` is non-positive or `X` is not a positive-shape 2D array.
     */
    BackfittingEngine(
        DoubleArray X_in,
        int32_t m,
        uint64_t seed = 0
    );

    /**
     * @brief Initialize the live forest as `m` single-node root trees.
     *
     * Clears any existing trees and replaces them with root-only trees that all
     * share the precomputed root sorted-row representation.
     */
    void initialize_root_forest();

    /**
     * @brief Run one complete Bayesian backfitting sweep over the forest.
     *
     * For each tree, adds the tree's current fitted values back into the residuals,
     * proposes a tree-structure update, draws new terminal-node means, and subtracts
     * the updated tree contribution from the residuals.
     *
     * @param residuals Mutable residual vector of length `n`.
     * @param sigma2 Current observation-noise variance.
     * @param sigma_mu2 Prior variance for terminal-node means.
     * @param alpha Tree-depth prior numerator parameter.
     * @param beta Tree-depth prior decay parameter.
     * @param move_distribution Probabilities for grow, prune, change, and swap moves.
     *
     * @throws std::runtime_error If `residuals` has wrong shape or move probabilities are invalid.
     */
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

    /**
     * @brief Propose and possibly accept one tree-structure update.
     *
     * Samples one Metropolis-Hastings move from the supplied move distribution and
     * applies it to tree `j` if accepted. Supported moves are grow, prune, change,
     * and swap.
     *
     * @param j Index of the tree to update.
     * @param residuals Current partial residual vector of length `n`.
     * @param sigma2 Current observation-noise variance.
     * @param sigma_mu2 Prior variance for terminal-node means.
     * @param alpha Tree-depth prior numerator parameter.
     * @param beta Tree-depth prior decay parameter.
     * @param move_distribution Probabilities for grow, prune, change, and swap moves.
     *
     * @return `true` if a proposal was accepted; otherwise `false`.
     *
     * @throws std::runtime_error If `j`, `residuals`, or `move_distribution` is invalid.
     */
    bool draw_tree(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2,
        double alpha,
        double beta,
        const std::array<double, 4>& move_distribution
    );

    /**
     * @brief Draw terminal-node means for one tree conditional on residuals.
     *
     * Updates each terminal node of tree `j` by drawing from its conjugate Gaussian
     * full conditional distribution.
     *
     * @param j Index of the tree to update.
     * @param residuals Current partial residual vector of length `n`.
     * @param sigma2 Current observation-noise variance.
     * @param sigma_mu2 Prior variance for terminal-node means.
     *
     * @throws std::runtime_error If `j` is out of bounds or `residuals` has wrong shape.
     */
    void draw_mu(
        int32_t j,
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2
    );

    /**
     * @brief Serialize one live tree into flat node arrays.
     *
     * Returns arrays for split variables, split values, left-child indices,
     * right-child indices, and terminal-node means. Child indices are local to the
     * serialized tree.
     *
     * @param j Index of the tree to serialize.
     *
     * @return Python tuple `(variable, value, left, right, mu)`.
     *
     * @throws std::runtime_error If `j` is out of bounds.
     */
    py::tuple serialize_tree(int32_t j) const;

    /**
     * @brief Serialize the full live forest into flat node arrays.
     *
     * Concatenates all `m` live trees into one serialized block and returns a
     * `tree_offset` array identifying each tree slice. Child indices are shifted
     * to refer to positions in the concatenated arrays.
     *
     * @return Python tuple `(variable, value, left, right, mu, tree_offset)`.
     */
    py::tuple serialize_forest() const;

    /**
     * @brief Compute raw and log-marginal-likelihood-weighted variable inclusion.
     *
     * Returns two length-p arrays for the current live forest:
     *   1. raw split-count variable inclusion
     *   2. subtree-collapse log marginal likelihood weighted inclusion
     *
     * The supplied residuals are the current full residuals:
     *     y_work - sum_j g_j(x)
     *
     * This method does not mutate residuals or the live forest.
     */
    py::tuple variable_inclusion(
        const DoubleArray& residuals,
        double sigma2,
        double sigma_mu2
    ) const;

    /**
     * @brief Validate one tree in the live forest.
     *
     * Checks tree topology, parent-child links, cached row state, and live/dead node
     * consistency for tree `j`.
     *
     * @param j Index of the tree to validate.
     *
     * @throws std::runtime_error If `j` is invalid or tree invariants are violated.
     */
    void validate_tree(int32_t j) const;

    /**
     * @brief Validate every tree in the live forest.
     *
     * Runs structural and cache consistency checks over all `m` live trees.
     *
     * @throws std::runtime_error If any tree invariant is violated.
     */
    void validate_forest() const;

    /**
     * @brief Return the number of training rows.
     *
     * @return Number of rows in the training matrix.
     */
    int32_t n() const noexcept { return n_; }

    /**
     * @brief Return the number of predictor columns.
     *
     * @return Number of columns in the training matrix.
     */
    int32_t p() const noexcept { return p_; }

    /**
     * @brief Return the number of trees in the ensemble.
     *
     * @return Number of live trees managed by the engine.
     */
    int32_t m() const noexcept { return m_; }

    /**
     * @brief Test helper that applies one tree's fitted values to residuals.
     *
     * Adds or subtracts the contribution of tree `j` from the supplied residual
     * vector according to `sign`. This is exposed only for low-level tests.
     *
     * @param j Index of the tree to apply.
     * @param residuals Mutable residual vector of length `n`.
     * @param sign Multiplier for the tree contribution, typically `+1.0` or `-1.0`.
     *
     * @throws std::runtime_error If `j` is out of bounds or `residuals` has wrong shape.
     */
    void test_apply_tree_to_residuals(int32_t j, DoubleArray residuals, double sign);

    /**
     * @brief Test helper that draws terminal means and subtracts the tree fit.
     *
     * Draws new terminal-node means for tree `j`, stores them in the tree, and
     * subtracts the resulting tree contribution from `residuals`. This is exposed
     * only for low-level tests.
     *
     * @param j Index of the tree to update.
     * @param residuals Mutable residual vector of length `n`.
     * @param sigma2 Current observation-noise variance.
     * @param sigma_mu2 Prior variance for terminal-node means.
     *
     * @throws std::runtime_error If `j` is out of bounds or `residuals` has wrong shape.
     */
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

    double leaf_log_marginal_contribution(
        int32_t n,
        double sum_r,
        double sigma2,
        double sigma_mu2
    ) const;

    double sum_residuals_for_node(
        const Node& node,
        const std::vector<double>& residuals
    ) const;

    double subtree_log_marginal_contribution(
        const Tree& tree,
        int32_t node_idx,
        const std::vector<double>& residuals,
        double sigma2,
        double sigma_mu2
    ) const;

    void add_tree_prediction_to_vector(
        const Tree& tree,
        std::vector<double>& values,
        double sign
    ) const;

    void accumulate_variable_inclusion_for_tree(
        const Tree& tree,
        const std::vector<double>& partial_residuals,
        double sigma2,
        double sigma_mu2,
        std::vector<double>& raw_counts,
        double& total_splits,
        std::vector<double>& logml_gains
    ) const;

    double p_split(int32_t depth, double alpha, double beta) const;
};

void bind_backfitting_engine(py::module_& m);