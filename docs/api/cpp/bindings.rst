pybind11 Bindings
=================

The C++ backend is exposed to Python through the ``genbart._backend`` extension
module.

The public Python estimators use this module internally. Most users should not
instantiate backend objects directly.

Binding functions
-----------------

.. doxygenfunction:: bind_tree

.. doxygenfunction:: bind_backfitting_engine

.. doxygenfunction:: bind_packed_forest