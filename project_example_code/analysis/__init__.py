"""
Analysis module for Pareto front visualization and scenario selection.
"""

from analysis.pareto_analysis import (
    load_results,
    results_to_dataframe,
    plot_pareto_front_3d,
    plot_pareto_front_2d_projections,
    plot_convergence,
    plot_parameter_distributions,
    select_critical_scenarios,
    save_critical_scenarios_report,
    run_complete_analysis,
)

__all__ = [
    "load_results",
    "results_to_dataframe",
    "plot_pareto_front_3d",
    "plot_pareto_front_2d_projections",
    "plot_convergence",
    "plot_parameter_distributions",
    "select_critical_scenarios",
    "save_critical_scenarios_report",
    "run_complete_analysis",
]

