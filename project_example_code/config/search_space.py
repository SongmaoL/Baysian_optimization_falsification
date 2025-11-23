"""
Search Space Definition for Multi-Objective Falsification Framework

This module defines the parameter bounds and constraints for the environmental
parameters that will be searched during the falsification process.
"""

from typing import Dict, Any, Tuple
import numpy as np

# ============================================================================
# WEATHER PARAMETERS
# ============================================================================
# CARLA WeatherParameters documentation:
# https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters

WEATHER_BOUNDS = {
    # Fog density: 0-100 (percentage)
    # Higher values reduce visibility significantly
    "fog_density": (0.0, 80.0),  # Cap at 80 to maintain some visibility
    
    # Precipitation: 0-100 (percentage)
    # Rain intensity affecting road grip and visibility
    "precipitation": (0.0, 100.0),
    
    # Precipitation deposits: 0-100 (percentage)
    # Amount of water accumulated on surfaces
    "precipitation_deposits": (0.0, 100.0),
    
    # Wind intensity: 0-100 (percentage)
    # Affects vegetation and potentially vehicle stability
    "wind_intensity": (0.0, 50.0),  # Cap at 50 for realism
    
    # Sun altitude angle: -90 to 90 (degrees)
    # -90 = midnight, 0 = sunset/sunrise, 90 = noon
    "sun_altitude_angle": (-30.0, 90.0),  # Avoid very dark conditions
    
    # Cloudiness: 0-100 (percentage)
    "cloudiness": (0.0, 100.0),
}

# ============================================================================
# LEAD VEHICLE BEHAVIOR PARAMETERS
# ============================================================================

LEAD_VEHICLE_BOUNDS = {
    # Base throttle for lead vehicle (0-1)
    "lead_base_throttle": (0.2, 0.6),
    
    # Frequency of speed changes (Hz) - how often lead changes behavior
    "lead_behavior_frequency": (0.05, 0.3),  # Every 3-20 seconds
    
    # Amplitude of throttle variations (0-1)
    "lead_throttle_variation": (0.0, 0.4),
    
    # Brake probability (0-1) - probability of braking event
    "lead_brake_probability": (0.0, 0.3),
    
    # Brake intensity when braking (0-1)
    "lead_brake_intensity": (0.1, 0.8),
    
    # Brake duration (timesteps)
    "lead_brake_duration": (5, 30),  # 0.5-3 seconds at dt=0.1
}

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

INITIAL_CONDITION_BOUNDS = {
    # Initial distance between ego and lead vehicle (meters)
    "initial_distance": (15.0, 80.0),
    
    # Initial ego velocity (m/s)
    "initial_ego_velocity": (5.0, 20.0),
    
    # Initial lead velocity (m/s)
    "initial_lead_velocity": (5.0, 25.0),
}

# ============================================================================
# COMBINED SEARCH SPACE
# ============================================================================

# All parameters combined for Bayesian Optimization
SEARCH_SPACE = {
    **WEATHER_BOUNDS,
    **LEAD_VEHICLE_BOUNDS,
    **INITIAL_CONDITION_BOUNDS,
}

# Parameter descriptions for documentation
PARAMETER_DESCRIPTIONS = {
    "fog_density": "Fog density (0-80%)",
    "precipitation": "Rain intensity (0-100%)",
    "precipitation_deposits": "Water on surfaces (0-100%)",
    "wind_intensity": "Wind strength (0-50%)",
    "sun_altitude_angle": "Sun angle, -30=dusk, 90=noon (degrees)",
    "cloudiness": "Cloud coverage (0-100%)",
    "lead_base_throttle": "Lead vehicle base throttle (0-1)",
    "lead_behavior_frequency": "How often lead changes speed (Hz)",
    "lead_throttle_variation": "Magnitude of speed changes (0-1)",
    "lead_brake_probability": "Probability of braking (0-1)",
    "lead_brake_intensity": "Brake strength when braking (0-1)",
    "lead_brake_duration": "How long to brake (timesteps)",
    "initial_distance": "Starting distance between vehicles (m)",
    "initial_ego_velocity": "Ego starting speed (m/s)",
    "initial_lead_velocity": "Lead starting speed (m/s)",
}

# ============================================================================
# PLAUSIBILITY CONSTRAINTS
# ============================================================================

# Physical constraints for plausibility checking
PLAUSIBILITY_CONSTRAINTS = {
    # Maximum longitudinal acceleration (m/s²)
    # Passenger cars: typically < 10 m/s² (1g)
    # Emergency braking: < 20 m/s² (2g)
    # Extreme but still possible: < 25 m/s² (2.5g)
    "max_longitudinal_accel": 25.0,  # 2.5g (relaxed from 20.0 for better gradient)
    "comfortable_max_accel": 8.0,    # 0.8g (tighter comfortable range)
    
    # Maximum jerk (m/s³)
    # Comfortable: < 2 m/s³
    # Acceptable: < 5 m/s³
    # Extreme but possible: < 15 m/s³
    "max_jerk": 15.0,                 # Relaxed from 10.0 for better gradient
    "comfortable_max_jerk": 2.0,      # Keep same
    
    # Maximum lateral acceleration (m/s²)
    "max_lateral_accel": 10.0,  # 1g
    "comfortable_max_lateral_accel": 5.0,  # 0.5g
    
    # Minimum time-to-collision (seconds)
    "critical_ttc": 2.0,  # Below this is critical
    
    # Minimum safe distance (meters)
    "min_safe_distance": 4.0,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get all parameter bounds for Bayesian Optimization.
    
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    return SEARCH_SPACE.copy()


def get_random_parameters() -> Dict[str, float]:
    """
    Generate random parameters within bounds for initial exploration.
    
    Returns:
        Dictionary of parameter names to random values
    """
    params = {}
    for param_name, (min_val, max_val) in SEARCH_SPACE.items():
        params[param_name] = np.random.uniform(min_val, max_val)
    return params


def validate_parameters(params: Dict[str, float]) -> bool:
    """
    Validate that parameters are within bounds.
    
    Args:
        params: Dictionary of parameter values
        
    Returns:
        True if all parameters are within bounds
    """
    for param_name, value in params.items():
        if param_name not in SEARCH_SPACE:
            return False
        min_val, max_val = SEARCH_SPACE[param_name]
        if not (min_val <= value <= max_val):
            return False
    return True


def clip_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """
    Clip parameters to be within bounds.
    
    Args:
        params: Dictionary of parameter values
        
    Returns:
        Clipped parameters
    """
    clipped = {}
    for param_name, value in params.items():
        if param_name in SEARCH_SPACE:
            min_val, max_val = SEARCH_SPACE[param_name]
            clipped[param_name] = np.clip(value, min_val, max_val)
        else:
            clipped[param_name] = value
    return clipped


def print_search_space_summary():
    """Print a summary of the search space."""
    print("=" * 80)
    print("FALSIFICATION SEARCH SPACE")
    print("=" * 80)
    print(f"\nTotal parameters: {len(SEARCH_SPACE)}")
    print(f"\nWeather parameters: {len(WEATHER_BOUNDS)}")
    print(f"Lead vehicle parameters: {len(LEAD_VEHICLE_BOUNDS)}")
    print(f"Initial condition parameters: {len(INITIAL_CONDITION_BOUNDS)}")
    
    print("\n" + "-" * 80)
    print("Parameter Details:")
    print("-" * 80)
    for param_name, (min_val, max_val) in SEARCH_SPACE.items():
        desc = PARAMETER_DESCRIPTIONS.get(param_name, "No description")
        print(f"{param_name:30s} [{min_val:8.2f}, {max_val:8.2f}]  {desc}")
    print("=" * 80)


if __name__ == "__main__":
    # Print search space summary
    print_search_space_summary()
    
    # Generate and validate random parameters
    print("\nExample random parameters:")
    random_params = get_random_parameters()
    for key, value in random_params.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nValidation result: {validate_parameters(random_params)}")

