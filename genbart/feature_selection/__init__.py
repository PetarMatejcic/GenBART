"""Feature-selection tools for BART models."""

from .results import (
    ThresholdResult,
    VariableSelectionResult,
    PredictiveSelectionResult,
)
from .thresholds import (
    VALID_THRESHOLD_METHODS,
    local_threshold,
    global_max_threshold,
    global_se_threshold,
    build_threshold_results,
)
from .utils import VALID_IMPORTANCE_KINDS
from .permutation_null import BartVariableSelector
from .predictive_degradation import (
    BartPredictiveSelector,
    VALID_TASKS,
    VALID_LOSSES,
)

__all__ = [
    "ThresholdResult",
    "VariableSelectionResult",
    "PredictiveSelectionResult",
    "VALID_THRESHOLD_METHODS",
    "VALID_IMPORTANCE_KINDS",
    "VALID_TASKS",
    "VALID_LOSSES",
    "local_threshold",
    "global_max_threshold",
    "global_se_threshold",
    "build_threshold_results",
    "BartVariableSelector",
    "BartPredictiveSelector",
]