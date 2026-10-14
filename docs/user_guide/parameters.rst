Model Parameters
================

GenBART exposes the main BART prior and MCMC hyperparameters directly through
:class:`genbart.reg_bart.RegBart` and
:class:`genbart.probit_bart.ProbitBart`.

The defaults are intended to provide a reasonable starting point. In most
applications, the parameters that are most useful to think about are the
number of trees ``m``, the tree-depth prior ``alpha`` and ``beta``, the
shrinkage parameter ``k``, and the number of retained posterior draws.


Number of trees: ``m``
----------------------

``m`` controls the number of trees in the BART ensemble.

BART represents the fitted function as a sum of many weak regression trees,

.. math::

   f(x) = \sum_{j=1}^{m} g(x; T_j, M_j).

Increasing ``m`` gives the model more trees over which to distribute the fitted
function. It also increases computation time and posterior storage.

The prior on each tree's terminal-node values depends on ``m``, so increasing
the number of trees also causes each individual tree to be more strongly
shrunk toward a small contribution.

The default is:

.. code-block:: python

   m = 200


Tree-depth prior: ``alpha`` and ``beta``
----------------------------------------

``alpha`` and ``beta`` control how likely a node is to split as tree depth
increases.

For a node at depth :math:`d`, the splitting probability is

.. math::

   p_{\mathrm{split}}(d)
   =
   \frac{\alpha}{(1+d)^{\beta}}.

The defaults are:

.. code-block:: python

   alpha = 0.95
   beta = 2.0

``alpha`` controls the overall tendency to split near the top of a tree.
Larger values make splitting more likely.

``beta`` controls how quickly the splitting probability decreases with depth.
Larger values therefore favor shallower trees more strongly.

Together, these parameters regularize the complexity of the individual trees.


Shrinkage: ``k``
----------------

``k`` controls the prior variance of terminal-node values and therefore how
strongly each individual tree is shrunk toward a small contribution.

For regression BART,

.. math::

   \sigma_{\mu}^{2}
   =
   \left(
   \frac{0.5}{k\sqrt{m}}
   \right)^2.

For probit BART,

.. math::

   \sigma_{\mu}^{2}
   =
   \left(
   \frac{3}{k\sqrt{m}}
   \right)^2.

The default is:

.. code-block:: python

   k = 2.0

Increasing ``k`` decreases the prior variance and produces stronger
regularization. Decreasing ``k`` allows individual trees to make larger
contributions to the fitted function.


MCMC settings: ``n_burn`` and ``n_samples``
-------------------------------------------

``n_burn`` is the number of initial MCMC iterations discarded before posterior
samples are retained.

``n_samples`` is the number of posterior forest draws retained after burn-in.

The defaults are:

.. code-block:: python

   n_burn = 200
   n_samples = 1000

Increasing ``n_samples`` gives more posterior draws for predictions, credible
intervals, partial dependence, and variable-importance summaries, but also
increases runtime and memory use.

``n_burn`` and ``n_samples`` control the sampling procedure rather than the
flexibility of the BART model itself.


Regression variance prior: ``nu`` and ``q``
-------------------------------------------

These parameters apply only to :class:`genbart.reg_bart.RegBart`.

Regression BART models Gaussian observation noise with an unknown variance
:math:`\sigma^2`. GenBART uses a scaled inverse-chi-squared prior for this
variance.

``nu`` controls the concentration of that prior. Larger values place more
prior mass close to the prior scale, while smaller values produce a more
diffuse prior.

The default is:

.. code-block:: python

   nu = 3.0

``q`` controls how the prior scale is calibrated from an initial residual
variance estimate. The scale is chosen so that this estimate is the ``q``
quantile of the prior distribution for :math:`\sigma^2`.

The default is:

.. code-block:: python

   q = 0.90

Thus, with the default value, approximately 90% of the prior mass lies below
the initial residual variance estimate.

These parameters do not apply to :class:`genbart.probit_bart.ProbitBart`,
where the latent Gaussian error variance is fixed to one.


Tree proposal distribution
--------------------------

``move_distribution`` specifies the probabilities of proposing the four tree
structure moves used by the MCMC sampler:

.. code-block:: text

   (grow, prune, change, swap)

The default is:

.. code-block:: python

   move_distribution = (0.25, 0.25, 0.40, 0.10)

The probabilities therefore correspond to:

- ``grow``: add a split,
- ``prune``: remove a split,
- ``change``: replace an existing splitting rule,
- ``swap``: exchange splitting rules between related internal nodes.

These values control which proposals are attempted; they do not directly
specify how often those proposals are accepted.


Reproducibility: ``random_state``
---------------------------------

``random_state`` seeds both the Python-side random number generator and the
compiled backfitting sampler.

Using the same data, hyperparameters, and seed produces the same retained
posterior draws.

The default is:

.. code-block:: python

   random_state = 0