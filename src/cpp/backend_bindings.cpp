#include <pybind11/pybind11.h>
#include "packed_forest.hpp"
#include "tree.hpp"
#include "backfitting_engine.hpp"

namespace py = pybind11;

/**
 * @brief Python extension module exposing the GenBART C++ backend.
 *
 * Registers the packed posterior forest, low-level tree testing interface, and
 * backfitting engine classes used by the Python BART estimators.
 *
 * @param m pybind11 module object populated with backend bindings.
 */

PYBIND11_MODULE(_backend, m) {
    m.doc() = R"pbdoc(
                Compiled backend for BART tree operations and posterior prediction.

                This module exposes low-level pybind11 bindings used by the Python BART
                estimators. It provides packed posterior-forest prediction, mutable tree
                debugging utilities, and the Bayesian backfitting engine.
                )pbdoc";
    bind_packed_forest(m);
    bind_tree(m);
    bind_backfitting_engine(m);
}