#pragma once
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class PackedForest {
public:
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

    py::array_t<double> draw_sums_row(py::array_t<double, py::array::c_style | py::array::forcecast> x_in) const;
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