Architecture
============

GenBART is organized around a Python public API and a C++ backend.

The Python layer provides user-facing estimators, handles input validation,
stores posterior draws, and exposes prediction utilities.

The C++ backend handles tree operations, Bayesian backfitting, posterior forest
serialization, and fast prediction from packed forests.