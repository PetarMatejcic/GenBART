.. GenBART documentation master file, created by
   sphinx-quickstart on Mon May 18 18:29:55 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GenBART
=======

GenBART is a Bayesian Additive Regression Trees package with a Python interface
and a C++ backend.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Fit your first regression or binary classification BART model.
   
   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      Model parameters, recommended workflows, and practical usage notes.

   .. grid-item-card:: Python API
      :link: api/python/index
      :link-type: doc

      Public estimators for regression and binary classification.

   .. grid-item-card:: C++ Backend
      :link: api/cpp/index
      :link-type: doc

      Internal tree, backfitting, and prediction machinery.

   .. grid-item-card:: Developer Notes
      :link: developer/index
      :link-type: doc

      Architecture, implementation details, and contribution notes.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   user_guide/index
   api/index
   developer/index
