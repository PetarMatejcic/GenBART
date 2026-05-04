#include<stdexcept>
#include<cstdint>
#include<sstream>
#include<vector>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include "packed_forest.hpp"

namespace py = pybind11;


PackedForest::PackedForest(
        py::array_t<int32_t> variable_,
        py::array_t<double, py::array::c_style | py::array::forcecast> value_,
        py::array_t<int32_t> left_,
        py::array_t<int32_t> right_,
        py::array_t<double, py::array::c_style | py::array::forcecast> mu_,
        py::array_t<int64_t> tree_offset_,
        int64_t n_samples_,
        int64_t m_,
        int64_t p_)
            : n_samples(n_samples_), m(m_), p(p_)
{
    
    if (n_samples <= 0) throw std::runtime_error("n_samples must be positive!");
    if (m <= 0) throw std::runtime_error("m must be positive!");
    if (p <= 0) throw std::runtime_error("p must be positive!");

    if(variable_.ndim() != 1 || value_.ndim() != 1 || tree_offset_.ndim() != 1
       || left_.ndim() != 1 || right_.ndim() != 1 || mu_.ndim() != 1){
        throw std::runtime_error("Input arrays must be 1D!");
    }

    const auto n_nodes = static_cast<int64_t>(variable_.shape(0));
    if (static_cast<int64_t>(value_.shape(0)) != n_nodes 
        || static_cast<int64_t>(left_.shape(0)) != n_nodes
        || static_cast<int64_t>(right_.shape(0)) != n_nodes
        || static_cast<int64_t>(mu_.shape(0)) != n_nodes){
            throw std::runtime_error("Node arrays must all be the same length!");
    }

    const int64_t n_trees_total = n_samples * m_;
    if (static_cast<int64_t>(tree_offset_.shape(0)) != n_trees_total + 1){
        throw std::runtime_error("tree_offset has wrong length!");
    }

    auto var_buf = variable_.request();
    auto val_buf = value_.request();
    auto left_buf = left_.request();
    auto right_buf = right_.request();
    auto mu_buf = mu_.request();
    auto off_buf = tree_offset_.request();

    variable.assign(static_cast<int32_t*>(var_buf.ptr),
                    static_cast<int32_t*>(var_buf.ptr) + n_nodes);
    value.assign(static_cast<const double*>(val_buf.ptr),
                    static_cast<const double*>(val_buf.ptr) + n_nodes);
    left.assign(static_cast<int32_t*>(left_buf.ptr),
                    static_cast<int32_t*>(left_buf.ptr) + n_nodes);
    right.assign(static_cast<int32_t*>(right_buf.ptr),
                    static_cast<int32_t*>(right_buf.ptr) + n_nodes);
    mu.assign(static_cast<const double*>(mu_buf.ptr),
                    static_cast<const double*>(mu_buf.ptr) + n_nodes);
    tree_offset.assign(static_cast<int64_t*>(off_buf.ptr),
                    static_cast<int64_t*>(off_buf.ptr) + n_trees_total + 1);
    
    if (tree_offset[0] != 0) throw std::runtime_error("tree_offset[0] must be 0!");

    for (int64_t t = 0; t < n_trees_total; ++t){
        if (tree_offset[t+1] <= tree_offset[t]){
            throw std::runtime_error("Every tree slice must contain a node!");
        }
    }

    if (tree_offset.back() != n_nodes) throw std::runtime_error("tree_offset.back() must be equal to node array length!");

    validate_structure();
}

py::array_t<double> PackedForest::draw_sums_row(py::array_t<double, py::array::c_style | py::array::forcecast> x_in) const
{
    validate_row_input(x_in);

    const auto* x = static_cast<const double*>(x_in.request().ptr);

    py::array_t<double> out(static_cast<py::ssize_t>(n_samples));
    auto* out_ptr = static_cast<double*>(out.request().ptr);

    py::gil_scoped_release release;

    for (int64_t d = 0; d < n_samples; ++d) {
        double s = 0.0;
        const int64_t first_tree = d * m;

        for (int64_t j = 0; j < m; ++j) {
            s += predict_tree_row_ptr(x, root_of_tree(first_tree + j));
        }

        out_ptr[d] = s;
    }

    return out;
}


py::array_t<double> PackedForest::draw_sums_matrix(py::array_t<double, py::array::c_style | py::array::forcecast> X_in) const
{
    validate_matrix_input(X_in);

    auto X_buf = X_in.request();
    const auto* X_ptr = static_cast<const double*>(X_buf.ptr);
    const int64_t n_rows = static_cast<int64_t>(X_in.shape(0));

    py::array_t<double> out(
        {static_cast<py::ssize_t>(n_samples),
            static_cast<py::ssize_t>(n_rows)}
    );
    auto* out_ptr = static_cast<double*>(out.request().ptr);

    py::gil_scoped_release release;

    for (int64_t d = 0; d < n_samples; ++d) {
        const int64_t first_tree = d * m;
        const int64_t out_base = d * n_rows;

        for (int64_t i = 0; i < n_rows; ++i) {
            out_ptr[out_base + i] = 0.0;
        }

        for (int64_t j = 0; j < m; ++j) {
            const int64_t root = root_of_tree(first_tree + j);

            for (int64_t i = 0; i < n_rows; ++i) {
                const double* xrow = X_ptr + i * p;
                out_ptr[out_base + i] += predict_tree_row_ptr(xrow, root);
            }
        }
    }

    return out;
}

void PackedForest::validate_structure() const 
{
    const int64_t n_trees_total = n_samples * m;
    const int64_t n_nodes = static_cast<int64_t>(variable.size());

    for (int64_t i = 0; i < n_nodes; ++i) {
        const bool is_leaf = (left[i] == -1);

        if (is_leaf) {
            if (right[i] != -1) {
                throw std::runtime_error("Leaf node has right child but no left child");
            }
        } else {
            if (right[i] == -1) {
                throw std::runtime_error("Internal node has left child but no right child");
            }
            if (variable[i] < 0 || variable[i] >= p) {
                throw std::runtime_error("Internal node has invalid split variable");
            }
        }
    }

    for (int64_t t = 0; t < n_trees_total; ++t) {
        const int64_t begin = tree_offset[t];
        const int64_t end   = tree_offset[t + 1];
        const int64_t root  = begin;

        std::vector<uint8_t> seen(static_cast<size_t>(end - begin), 0);
        std::vector<int64_t> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int64_t node = stack.back();
            stack.pop_back();

            if (node < begin || node >= end) {
                throw std::runtime_error("Node outside tree slice reached during DFS");
            }

            const size_t local = static_cast<size_t>(node - begin);
            if (seen[local]) {
                throw std::runtime_error("Cycle or duplicate parent detected in tree slice");
            }
            seen[local] = 1;

            if (left[node] != -1) {
                if (left[node] < begin || left[node] >= end ||
                    right[node] < begin || right[node] >= end) {
                    throw std::runtime_error("Child pointer escapes tree slice");
                }
                stack.push_back(right[node]);
                stack.push_back(left[node]);
            }
        }

        for (uint8_t flag : seen) {
            if (!flag) {
                throw std::runtime_error("Tree slice contains unreachable nodes");
            }
        }
    }
}

void PackedForest::validate_row_input(const py::array_t<double, py::array::c_style | py::array::forcecast>& x) const
{
    if (x.ndim() != 1) {
        throw std::runtime_error("x must be a 1D NumPy array.");
    }
    if (static_cast<int64_t>(x.shape(0)) != p) {
        std::ostringstream oss;
        oss << "x has wrong length: got " << x.shape(0)
            << ", expected " << p << ".";
        throw std::runtime_error(oss.str());
    }
}

void PackedForest::validate_matrix_input(const py::array_t<double, py::array::c_style | py::array::forcecast>& X) const
{
    if (X.ndim() != 2) {
        throw std::runtime_error("X must be a 2D NumPy array.");
    }
    if (static_cast<int64_t>(X.shape(1)) != p) {
        std::ostringstream oss;
        oss << "X has wrong number of columns: got " << X.shape(1)
            << ", expected " << p << ".";
        throw std::runtime_error(oss.str());
    }
}

void bind_packed_forest(py::module_& m) {
    py::class_<PackedForest>(m, "PackedForest",
        R"pbdoc(
        Packed representation of retained posterior BART forest draws.

        A PackedForest stores all retained MCMC forest draws in flat node arrays.
        It is optimized for fast posterior prediction by summing tree predictions
        across each retained draw.
        )pbdoc")
        .def(
            py::init<
                py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                py::array_t<double,   py::array::c_style | py::array::forcecast>,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                py::array_t<double,   py::array::c_style | py::array::forcecast>,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                int64_t,
                int64_t,
                int64_t
            >(),
            R"pbdoc(
            Create a packed forest from flat serialized node arrays.

            Args:
                variable: Split-variable array. Negative values indicate terminal nodes.
                value: Split-threshold array.
                left: Left-child node indices, or -1 for terminal nodes.
                right: Right-child node indices, or -1 for terminal nodes.
                mu: Terminal-node mean values.
                tree_offset: Offsets delimiting each serialized tree.
                n_draws: Number of retained posterior forest draws.
                m: Number of trees per forest draw.
                p: Number of predictor columns expected at prediction time.

            Raises:
                RuntimeError: If array dimensions, node-array lengths, offsets, or tree
                    structure invariants are invalid.
            )pbdoc",
            py::arg("variable"),
            py::arg("value"),
            py::arg("left"),
            py::arg("right"),
            py::arg("mu"),
            py::arg("tree_offset"),
            py::arg("n_draws"),
            py::arg("m"),
            py::arg("p")
        )
        .def("draw_sums_row",
            &PackedForest::draw_sums_row,
            R"pbdoc(
            Evaluate all posterior forest draws for one feature row.

            Args:
                x: One-dimensional feature row with length equal to the training feature
                    count.

            Returns:
                A one-dimensional NumPy array of length n_draws. Each entry is the sum of
                all tree predictions for one retained posterior draw.

            Raises:
                RuntimeError: If x is not one-dimensional or has the wrong length.
            )pbdoc",
            py::arg("x"))
        .def("draw_sums_matrix",
            &PackedForest::draw_sums_matrix,
            R"pbdoc(
            Evaluate all posterior forest draws for a feature matrix.

            Args:
                X: Two-dimensional feature matrix with shape (n_rows, p).

            Returns:
                A NumPy array with shape (n_draws, n_rows). Entry (d, i) is the sum of all
                tree predictions for posterior draw d at row i.

            Raises:
                RuntimeError: If X is not two-dimensional or has the wrong number of
                    columns.
            )pbdoc",
            py::arg("X"));
}