#include <pybind11/pybind11.h>
#include "packed_forest.hpp"
#include "tree.hpp"
#include "backfitting_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_backend, m) {
    bind_packed_forest(m);
    bind_tree(m);
    bind_backfitting_engine(m);
}