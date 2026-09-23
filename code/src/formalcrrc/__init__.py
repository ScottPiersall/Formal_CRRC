"""FormalCRRC: counterfactual rubric response curves under formal thresholds.

Day 1 studies whether LLM judges reproduce the monotone counterfactual response
function implied by a rubric when the evaluated artifact is held byte-identical
and only a programmatically controlled decision threshold varies.

Ground truth in this package is computed by formal predicates, never by a model.
"""

__version__ = "0.1.0"

from formalcrrc import (  # noqa: F401
    bootstrap,
    config,
    dataset,
    day1_audit,
    day2,
    day2_bootstrap,
    day2_config,
    day3,
    day3_bootstrap,
    day3_config,
    metrics,
    predicates,
    prompts,
    provenance,
    scoring,
)

__all__ = [
    "__version__",
    "bootstrap",
    "config",
    "dataset",
    "day1_audit",
    "day2",
    "day2_bootstrap",
    "day2_config",
    "day3",
    "day3_bootstrap",
    "day3_config",
    "metrics",
    "predicates",
    "prompts",
    "provenance",
    "scoring",
]
