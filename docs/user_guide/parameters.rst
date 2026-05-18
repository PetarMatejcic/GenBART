Model Parameters
================

Number of trees: ``m``
----------------------

``m`` controls the number of trees in the BART ensemble.

Larger values usually give the model more flexibility, but also increase runtime
and posterior storage cost. Common starting values are ``50``, ``100``, and
``200``.

Tree prior: ``alpha`` and ``beta``
----------------------------------

``alpha`` and ``beta`` control the prior probability that a node splits as a
function of depth.

The default values are:

.. code-block:: python

   alpha = 0.95
   beta = 2.0

These defaults favor shallow trees while still allowing deeper trees when the
data support them.

Shrinkage: ``k``
----------------

``k`` controls the prior scale of terminal-node means.

Larger ``k`` means stronger shrinkage, so each individual tree has a smaller
effect. Smaller ``k`` allows individual trees to contribute more strongly.

MCMC settings: ``n_burn`` and ``n_samples``
-------------------------------------------

``n_burn`` controls how many initial MCMC iterations are discarded.

``n_samples`` controls how many posterior draws are retained after burn-in.

Regression-only parameters: ``nu`` and ``q``
--------------------------------------------

``RegBart`` uses ``nu`` and ``q`` to calibrate the prior for the observation
variance.

These parameters do not apply to ``ProbitBart``, where the latent probit error
variance is fixed.

Move distribution
-----------------

``move_distribution`` gives the proposal probabilities for the four tree
structure moves:

.. code-block:: python

   (grow, prune, change, swap)

The default is:

.. code-block:: python

   (0.25, 0.25, 0.40, 0.10)