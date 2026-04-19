#include<stdexcept>
#include<cstdint>
#include<vector>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>


namespace py = pybind11;


class BartBackfittingEngine {
public:
    BartBackfittingEngine(
        py::array_t<double, py::array::c_style | py::array::forcecast> X,
        int64_t m,
        uint64_t seed
    );

    // build m root trees with shared root rows_by_var caches
    void initialize_root_forest();

    // shared hot path
    bool draw_tree(
        int64_t j,
        py::array_t<double, py::array::c_style | py::array::forcecast> residuals,
        double sigma2,
        double sigma_mu2,
        double alpha,
        double beta,
        std::array<double, 4> move_probs
    );

    void draw_mu(
        int64_t j,
        py::array_t<double, py::array::c_style | py::array::forcecast> residuals,
        double sigma2,
        double sigma_mu2
    );

    void refresh_tree_training_predictions(
        int64_t j,
        py::array_t<double, py::array::c_style | py::array::forcecast> training_predictions,
        py::array_t<double, py::array::c_style | py::array::forcecast> fitted_sums
    );

    // posterior / serialization helpers
    py::tuple serialize_tree(int64_t j) const;
    py::tuple serialize_forest() const;

    // debug / testing helpers
    int64_t count_nodes(int64_t j) const;
    int64_t count_terminal_nodes(int64_t j) const;
    int64_t count_internal_nodes(int64_t j) const;
    void validate_tree(int64_t j) const;
    void validate_forest() const;
    py::dict debug_tree_summary(int64_t j) const;
};