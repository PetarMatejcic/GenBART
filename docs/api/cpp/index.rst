C++ API
=======

The C++ backend implements the mutable tree representation, Bayesian
backfitting engine, posterior forest serialization, and fast prediction path used
by the Python estimators.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Backend Overview
      :link: overview
      :link-type: doc

      How the C++ backend fits into the GenBART training and prediction flow.

   .. grid-item-card:: Tree
      :link: tree
      :link-type: doc

      Mutable regression-tree structure and tree proposal operations.

   .. grid-item-card:: BackfittingEngine
      :link: backfitting_engine
      :link-type: doc

      MCMC backfitting loop, tree updates, and terminal-node draws.

   .. grid-item-card:: PackedForest
      :link: packed_forest
      :link-type: doc

      Compact posterior forest representation for fast prediction.

   .. grid-item-card:: pybind11 Bindings
      :link: bindings
      :link-type: doc

      Python extension bindings exposed through ``genbart._backend``.

.. toctree::
   :maxdepth: 1
   :caption: C++ API pages:

   overview
   tree
   backfitting_engine
   packed_forest
   bindings