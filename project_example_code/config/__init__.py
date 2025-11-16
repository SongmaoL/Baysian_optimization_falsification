"""
Configuration module for multi-objective falsification framework.
"""

from config.search_space import (
    SEARCH_SPACE,
    WEATHER_BOUNDS,
    LEAD_VEHICLE_BOUNDS,
    INITIAL_CONDITION_BOUNDS,
    PLAUSIBILITY_CONSTRAINTS,
    PARAMETER_DESCRIPTIONS,
    get_parameter_bounds,
    get_random_parameters,
    validate_parameters,
    clip_parameters,
    print_search_space_summary,
)

__all__ = [
    "SEARCH_SPACE",
    "WEATHER_BOUNDS",
    "LEAD_VEHICLE_BOUNDS",
    "INITIAL_CONDITION_BOUNDS",
    "PLAUSIBILITY_CONSTRAINTS",
    "PARAMETER_DESCRIPTIONS",
    "get_parameter_bounds",
    "get_random_parameters",
    "validate_parameters",
    "clip_parameters",
    "print_search_space_summary",
]

