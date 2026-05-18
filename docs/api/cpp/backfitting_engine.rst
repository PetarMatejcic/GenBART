BackfittingEngine
=================

``BackfittingEngine`` coordinates Bayesian backfitting updates for the live BART
forest.

It owns the current ensemble of trees, updates each tree conditionally on the
partial residuals, draws terminal-node means, and serializes posterior forests for
storage by the Python layer.

Class reference
---------------

.. doxygenclass:: BackfittingEngine
   :members:

Supporting structures
---------------------

If these are documented in your headers, include them:

.. doxygenstruct:: TerminalStat
   :members:

.. doxygenstruct:: InternalStat
   :members: