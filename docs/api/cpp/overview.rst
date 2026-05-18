Backend Overview
================

The C++ backend is responsible for the performance-critical parts of GenBART:
tree mutation, Bayesian backfitting, forest serialization, and posterior
prediction.

Execution flow
--------------

During fitting, the Python estimator initializes model state and delegates tree
updates to the C++ backend.

.. code-block:: text

   RegBart / ProbitBart
        |
        v
   BaseBART
        |
        v
   _BackfittingEngine
        |
        v
   Tree proposal and update machinery
        |
        v
   Serialized posterior forests
        |
        v
   PackedForest
        |
        v
   Fast posterior prediction

Main backend classes
--------------------

``Tree``
   Stores one mutable regression tree during MCMC. It handles node storage,
   row partitions, valid split caches, proposal construction, accepted proposal
   application, serialization, and validation.

``BackfittingEngine``
   Owns the live ensemble of trees during MCMC. It runs the Bayesian
   backfitting sweep, proposes tree updates, draws terminal-node means, and
   serializes the current forest.

``PackedForest``
   Stores retained posterior trees in flat arrays after fitting. It is used for
   fast posterior prediction without keeping the mutable MCMC tree objects alive.

Python boundary
---------------

The backend is exposed through the ``genbart._backend`` pybind11 extension
module. Public users normally interact with ``RegBart`` and ``ProbitBart``;
the C++ objects are implementation details and developer-facing APIs.