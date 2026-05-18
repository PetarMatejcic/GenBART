#pragma once
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * @brief Packed posterior forest representation used for fast prediction.
 *
 * PackedForest stores all retained posterior trees in flat arrays. Each tree is
 * represented by contiguous node slices indexed by ``tree_offset``. This layout
 * is used after fitting to evaluate posterior draw sums for a single row or a
 * matrix of rows without keeping the mutable MCMC tree objects alive.
 */
class PackedForest {
public:
    /**
     * @brief Construct a packed posterior forest from flat serialized node arrays.
     *
     * Stores all posterior tree draws in contiguous arrays so prediction can be
     * performed without traversing Python objects. Each tree is represented by a
     * slice of the node arrays, with `tree_offset` marking the start of each tree.
     *
     * @param variable Split variable for each node, or -1 for terminal nodes.
     * @param value Split threshold for each node.
     * @param left Left-child node index for each node, or -1 for terminal nodes.
     * @param right Right-child node index for each node, or -1 for terminal nodes.
     * @param mu Terminal-node mean value for each leaf node.
     * @param tree_offset Offsets delimiting each serialized tree.
     * @param n_draws Number of posterior forest draws.
     * @param m Number of trees per posterior draw.
     * @param p Number of predictor columns expected at prediction time.
     *
     * @throws std::runtime_error If dimensions, offsets, or tree structure are invalid.
     */
    PackedForest(
        py::array_t<int32_t> variable_,
        py::array_t<double, py::array::c_style | py::array::forcecast> value_,
        py::array_t<int32_t> left_,
        py::array_t<int32_t> right_,
        py::array_t<double, py::array::c_style | py::array::forcecast> mu_,
        py::array_t<int64_t> tree_offset_,
        int64_t n_samples_,
        int64_t m_,
        int64_t p_
    );

    /**
     * @brief Evaluate every posterior forest draw at a single predictor row.
     *
     * For each posterior draw, sums the predictions from its `m` trees at `x`.
     * The returned vector therefore contains one sum-of-trees prediction per
     * retained posterior sample.
     *
     * @param x One-dimensional predictor row of length `p`.
     *
     * @return NumPy array of shape `(n_draws,)` containing posterior draw sums.
     *
     * @throws std::runtime_error If `x` is not one-dimensional or has wrong length.
     */
    py::array_t<double> draw_sums_row(py::array_t<double, py::array::c_style | py::array::forcecast> x_in) const;
    /**
     * @brief Evaluate every posterior forest draw on a matrix of predictor rows.
     *
     * For each posterior draw and each row of `X`, sums the predictions from the
     * corresponding `m` trees. The output is arranged with posterior draws on the
     * first axis and input rows on the second axis.
     *
     * @param X Two-dimensional predictor matrix with shape `(n_rows, p)`.
     *
     * @return NumPy array of shape `(n_draws, n_rows)` containing posterior draw sums.
     *
     * @throws std::runtime_error If `X` is not two-dimensional or has wrong column count.
     */
    py::array_t<double> draw_sums_matrix(py::array_t<double, py::array::c_style | py::array::forcecast> X_in) const;

private:
    std::vector<int32_t> variable;
    std::vector<double> value;
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    std::vector<double> mu;
    std::vector<int64_t> tree_offset;
    int64_t n_samples;
    int64_t m;
    int64_t p;

    void validate_structure() const;
    void validate_row_input(const py::array_t<double, py::array::c_style | py::array::forcecast>& x) const;
    void validate_matrix_input(const py::array_t<double, py::array::c_style | py::array::forcecast>& X) const;

    inline bool is_leaf(int64_t node) const noexcept { return left[node] == -1; }
    inline int64_t root_of_tree(int64_t tree_id) const noexcept { return tree_offset[tree_id]; }

    inline double predict_tree_row_ptr(const double* x, int64_t root) const noexcept {
        int64_t node = root;
        while (!is_leaf(node)) {
            node = (x[variable[node]] <= value[node]) ? left[node] : right[node];
        }
        return static_cast<double>(mu[node]);
    }
};

void bind_packed_forest(py::module_& m);