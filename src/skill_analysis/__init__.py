"""
skill_analysis package
----------------------
Post-recommendation skill-gap analysis and career insight helpers.
"""

from .analysis import (
    analyze_recommendations,
    build_graphviz_dot,
    build_skill_graph,
)
from .llm_explainer import explain_analysis, llm_is_configured

__all__ = [
    "analyze_recommendations",
    "build_graphviz_dot",
    "build_skill_graph",
    "explain_analysis",
    "llm_is_configured",
]
