Tree
====

``Tree`` is the mutable regression-tree representation used during MCMC.

It stores the tree topology, split rules, terminal-node means, row membership,
valid split caches, and proposal state needed for grow, prune, change, and swap
moves.

Class reference
---------------

.. doxygenclass:: Tree
   :members:

Related structures
------------------

.. doxygenstruct:: Node
   :members:

.. doxygenstruct:: GrowProposalLite
   :members:

.. doxygenstruct:: PruneProposalLite
   :members:

.. doxygenstruct:: ChangeProposalLite
   :members:

.. doxygenstruct:: SwapProposalLite
   :members: