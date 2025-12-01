"""
Scenario Generator for Multi-Objective Falsification

This module generates CARLA scenarios from search space parameters.
It creates:
1. Weather configurations
2. Lead vehicle behavior profiles
3. Initial conditions for ego and lead vehicles
4. Complete scenario JSON files for simulation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

try:
    import carla
except ImportError:
    print("Warning: CARLA not available. Weather generation will be limited.")
    carla = None


# ============================================================================
# WEATHER GENERATION
# ============================================================================

def create_weather_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """
    Create CARLA WeatherParameters from search space parameters.
    
    Args:
        params: Dictionary with weather parameter values
        
    Returns:
        Dictionary of weather parameters for CARLA
    """
    weather_params = {
        'cloudiness': params.get('cloudiness', 50.0),
        'precipitation': params.get('precipitation', 0.0),
        'precipitation_deposits': params.get('precipitation_deposits', 0.0),
        'wind_intensity': params.get('wind_intensity', 0.0),
        'sun_altitude_angle': params.get('sun_altitude_angle', 45.0),
        'fog_density': params.get('fog_density', 0.0),
        'fog_distance': 0.0,  # Auto-calculated from fog_density
        'wetness': 0.0,  # Auto-calculated from precipitation
    }
    
    # Auto-calculate fog_distance from fog_density
    # fog_distance decreases as fog_density increases
    # fog_density 0% -> fog_distance = inf (no fog)
    # fog_density 100% -> fog_distance = ~10m (very dense)
    if weather_params['fog_density'] > 0:
        # Exponential relationship: higher density = shorter visibility
        weather_params['fog_distance'] = 1000.0 * np.exp(-weather_params['fog_density'] / 30.0)
    else:
        weather_params['fog_distance'] = 1000.0  # Clear visibility
    
    # Auto-calculate wetness from precipitation
    weather_params['wetness'] = min(weather_params['precipitation'] / 2.0, 100.0)
    
    return weather_params


def weather_to_carla_object(weather_params: Dict[str, float]):
    """
    Convert weather parameters dict to CARLA WeatherParameters object.
    
    Args:
        weather_params: Dictionary of weather parameters
        
    Returns:
        carla.WeatherParameters object (or None if CARLA not available)
    """
    if carla is None:
        return None
    
    weather = carla.WeatherParameters(
        cloudiness=weather_params['cloudiness'],
        precipitation=weather_params['precipitation'],
        precipitation_deposits=weather_params['precipitation_deposits'],
        wind_intensity=weather_params['wind_intensity'],
        sun_altitude_angle=weather_params['sun_altitude_angle'],
        fog_density=weather_params['fog_density'],
        fog_distance=weather_params['fog_distance'],
        wetness=weather_params['wetness'],
    )
    
    return weather


# ============================================================================
# LEAD VEHICLE BEHAVIOR GENERATION
# ============================================================================

def generate_lead_vehicle_actions(params: Dict[str, float], 
                                  num_timesteps: int = 200,
                                  dt: float = 0.1,
                                  seed: Optional[int] = None) -> List[Dict[str, float]]:
    """
    Generate lead vehicle action sequence from behavior parameters.
    
    The lead vehicle behavior is generated using:
    - Base throttle with sinusoidal variations
    - Random braking events based on probability
    - SMOOTH transitions between actions (exponential smoothing)
    
    Args:
        params: Dictionary with lead vehicle behavior parameters
        num_timesteps: Number of timesteps to generate
        dt: Time step duration
        seed: Random seed for reproducibility
        
    Returns:
        List of action dictionaries with throttle, brake, steer
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Extract parameters
    base_throttle = params.get('lead_base_throttle', 0.4)
    behavior_freq = params.get('lead_behavior_frequency', 0.1)  # Hz
    throttle_var = params.get('lead_throttle_variation', 0.2)
    brake_prob = params.get('lead_brake_probability', 0.1)
    brake_intensity = params.get('lead_brake_intensity', 0.5)
    brake_duration = int(params.get('lead_brake_duration', 10))
    
    # Smoothing parameters - prevents unrealistic step functions
    # Alpha=0.05 gives smoother transitions (~2 seconds to reach target)
    # This keeps both acceleration AND jerk within realistic bounds
    # Reduced from 0.1 to improve plausibility (reduce jerk)
    SMOOTHING_ALPHA = 0.05  # Lower = smoother (reduced to improve plausibility)
    
    actions = []
    time = 0.0
    braking = False
    brake_timer = 0
    
    # State for exponential smoothing
    current_throttle = base_throttle
    current_brake = 0.0
    
    for step in range(num_timesteps):
        # Check if we should start braking
        if not braking and np.random.random() < brake_prob * dt:
            braking = True
            brake_timer = brake_duration
        
        if braking:
            # Target: braking
            target_throttle = 0.0
            target_brake = brake_intensity
            brake_timer -= 1
            
            if brake_timer <= 0:
                braking = False
        else:
            # Target: normal throttle with sinusoidal variation
            omega = 2 * np.pi * behavior_freq
            variation = throttle_var * np.sin(omega * time)
            target_throttle = np.clip(base_throttle + variation, 0.0, 1.0)
            target_brake = 0.0
        
        # Exponential smoothing for realistic transitions
        # This prevents instant acceleration changes that violate physics
        current_throttle = SMOOTHING_ALPHA * target_throttle + (1 - SMOOTHING_ALPHA) * current_throttle
        current_brake = SMOOTHING_ALPHA * target_brake + (1 - SMOOTHING_ALPHA) * current_brake
        
        # Keep steering minimal (straight line)
        steer = 0.0
        
        actions.append({
            'throttle': float(current_throttle),
            'brake': float(current_brake),
            'steer': float(steer),
        })
        
        time += dt
    
    return actions


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

def generate_initial_conditions(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate initial vehicle positions and velocities.
    
    Uses fixed spawn locations from Town06 with configurable distances.
    
    Args:
        params: Dictionary with initial condition parameters
        
    Returns:
        Dictionary with ego and lead initial states
    """
    # Extract parameters
    initial_distance = params.get('initial_distance', 40.0)
    initial_ego_velocity = params.get('initial_ego_velocity', 10.0)
    initial_lead_velocity = params.get('initial_lead_velocity', 15.0)
    
    # Fixed starting position for lead vehicle (front)
    # These coordinates are for Town06 straight road
    lead_x = 484.88
    lead_y = 20.57
    lead_z = 0.294
    lead_yaw = 0.0
    
    # Ego vehicle spawns behind lead vehicle
    # Distance is along x-axis (assuming straight road)
    ego_x = lead_x + initial_distance  # Behind in positive x direction
    ego_y = lead_y
    ego_z = lead_z
    ego_yaw = lead_yaw
    
    initial_conditions = {
        'ego': {
            'position': {
                'x': ego_x,
                'y': ego_y,
                'z': ego_z,
            },
            'yaw': ego_yaw,
            'velocity': initial_ego_velocity,
        },
        'lead': {
            'position': {
                'x': lead_x,
                'y': lead_y,
                'z': lead_z,
            },
            'yaw': lead_yaw,
            'velocity': initial_lead_velocity,
        }
    }
    
    return initial_conditions


# ============================================================================
# COMPLETE SCENARIO GENERATION
# ============================================================================

def generate_scenario(params: Dict[str, float],
                     num_timesteps: int = 200,
                     dt: float = 0.1,
                     seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate complete scenario from search space parameters.
    
    Args:
        params: Dictionary with all search space parameters
        num_timesteps: Number of timesteps for simulation
        dt: Time step duration
        seed: Random seed for reproducibility
        
    Returns:
        Complete scenario dictionary with all components
    """
    # Generate all components
    weather = create_weather_parameters(params)
    lead_actions = generate_lead_vehicle_actions(params, num_timesteps, dt, seed)
    initial_conditions = generate_initial_conditions(params)
    
    # Combine into complete scenario
    scenario = {
        'weather': weather,
        'initial_conditions': initial_conditions,
        'ego': initial_conditions['ego'],
        'lead': initial_conditions['lead'],
        'ado_actions': lead_actions,  # Using 'ado_actions' to match existing format
        'parameters': params,  # Store original parameters for reference
        'config': {
            'num_timesteps': num_timesteps,
            'dt': dt,
            'seed': seed,
        }
    }
    
    return scenario


def save_scenario_json(scenario: Dict[str, Any], output_path: Path):
    """
    Save scenario to JSON file.
    
    Args:
        scenario: Scenario dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(scenario, f, indent=4)
    
    print(f"Saved scenario to {output_path}")


def load_scenario_json(input_path: Path) -> Dict[str, Any]:
    """
    Load scenario from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Scenario dictionary
    """
    with open(input_path, 'r') as f:
        scenario = json.load(f)
    
    return scenario


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_scenario_batch(parameter_list: List[Dict[str, float]],
                           output_dir: Path,
                           prefix: str = "scenario",
                           **kwargs) -> List[Path]:
    """
    Generate multiple scenarios from a list of parameters.
    
    Args:
        parameter_list: List of parameter dictionaries
        output_dir: Directory to save scenarios
        prefix: Prefix for scenario filenames
        **kwargs: Additional arguments for generate_scenario
        
    Returns:
        List of paths to generated scenario files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_paths = []
    
    for i, params in enumerate(parameter_list):
        scenario = generate_scenario(params, **kwargs)
        output_path = output_dir / f"{prefix}_{i:04d}.json"
        save_scenario_json(scenario, output_path)
        scenario_paths.append(output_path)
    
    return scenario_paths


# ============================================================================
# EXAMPLE / TESTING
# ============================================================================

if __name__ == "__main__":
    from config.search_space import get_random_parameters
    
    print("=" * 80)
    print("SCENARIO GENERATOR")
    print("=" * 80)
    
    # Generate random parameters
    print("\nGenerating random parameters...")
    params = get_random_parameters()
    
    print("\nParameter values:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate scenario
    print("\nGenerating scenario...")
    scenario = generate_scenario(params, seed=42)
    
    print("\nScenario components:")
    print(f"  Weather parameters: {len(scenario['weather'])} parameters")
    print(f"  Lead vehicle actions: {len(scenario['ado_actions'])} timesteps")
    print(f"  Initial ego position: ({scenario['ego']['position']['x']:.2f}, "
          f"{scenario['ego']['position']['y']:.2f})")
    print(f"  Initial lead position: ({scenario['lead']['position']['x']:.2f}, "
          f"{scenario['lead']['position']['y']:.2f})")
    
    # Save example
    output_path = Path("scenarios/example_scenario.json")
    save_scenario_json(scenario, output_path)
    
    print("\n" + "=" * 80)

