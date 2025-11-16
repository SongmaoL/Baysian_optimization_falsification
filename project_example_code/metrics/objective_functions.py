"""
Objective Functions for Multi-Objective Falsification

This module implements the three objective functions:
1. Safety Score - measures how unsafe a scenario is (minimize to find violations)
2. Plausibility Score - measures physical realism (maximize for realistic scenarios)
3. Comfort Score - measures passenger discomfort (minimize to find uncomfortable scenarios)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path

from config.search_space import PLAUSIBILITY_CONSTRAINTS


# ============================================================================
# HELPER FUNCTIONS FOR METRIC CALCULATION
# ============================================================================

def calculate_minimum_ttc(trace_df: pd.DataFrame) -> float:
    """
    Calculate minimum Time-to-Collision (TTC) from trace.
    
    TTC = distance / relative_velocity (when closing)
    
    Args:
        trace_df: DataFrame with columns: ego_velocity, lead_speed, distance_to_lead
        
    Returns:
        Minimum TTC in seconds (inf if never closing)
    """
    ttc_values = []
    
    for _, row in trace_df.iterrows():
        ego_vel = row['ego_velocity']
        lead_vel = row.get('lead_speed', 0)
        distance = row['distance_to_lead']
        
        # Relative velocity (positive when closing)
        rel_vel = ego_vel - lead_vel
        
        # Only calculate TTC when vehicles are closing (rel_vel > 0)
        if rel_vel > 0.1 and distance > 0:  # 0.1 m/s threshold
            ttc = distance / rel_vel
            ttc_values.append(ttc)
    
    return min(ttc_values) if ttc_values else float('inf')


def calculate_minimum_distance(trace_df: pd.DataFrame) -> float:
    """
    Calculate minimum distance between vehicles.
    
    Args:
        trace_df: DataFrame with column: distance_to_lead
        
    Returns:
        Minimum distance in meters
    """
    return trace_df['distance_to_lead'].min()


def calculate_maximum_jerk(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate maximum jerk (rate of change of acceleration).
    
    Args:
        trace_df: DataFrame with column: ego_velocity
        dt: Time step between measurements
        
    Returns:
        Maximum absolute jerk in m/s³
    """
    velocities = trace_df['ego_velocity'].values
    
    # Calculate acceleration (derivative of velocity)
    accelerations = np.diff(velocities) / dt
    
    # Calculate jerk (derivative of acceleration)
    if len(accelerations) > 1:
        jerks = np.diff(accelerations) / dt
        return np.max(np.abs(jerks)) if len(jerks) > 0 else 0.0
    
    return 0.0


def calculate_maximum_acceleration(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate maximum absolute acceleration.
    
    Args:
        trace_df: DataFrame with column: ego_velocity
        dt: Time step between measurements
        
    Returns:
        Maximum absolute acceleration in m/s²
    """
    velocities = trace_df['ego_velocity'].values
    accelerations = np.diff(velocities) / dt
    return np.max(np.abs(accelerations)) if len(accelerations) > 0 else 0.0


def calculate_total_jerk(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate total accumulated jerk (sum of absolute jerks).
    
    Args:
        trace_df: DataFrame with column: ego_velocity
        dt: Time step between measurements
        
    Returns:
        Total jerk in m/s³
    """
    velocities = trace_df['ego_velocity'].values
    accelerations = np.diff(velocities) / dt
    
    if len(accelerations) > 1:
        jerks = np.diff(accelerations) / dt
        return np.sum(np.abs(jerks))
    
    return 0.0


def count_hard_events(trace_df: pd.DataFrame, dt: float = 0.1, threshold: float = 3.0) -> int:
    """
    Count hard braking/acceleration events (> 0.3g = 2.94 m/s²).
    
    Args:
        trace_df: DataFrame with column: ego_velocity
        dt: Time step between measurements
        threshold: Acceleration threshold in m/s² (default: 3.0 m/s² ≈ 0.3g)
        
    Returns:
        Number of hard events
    """
    velocities = trace_df['ego_velocity'].values
    accelerations = np.diff(velocities) / dt
    
    # Count events exceeding threshold
    hard_events = np.sum(np.abs(accelerations) > threshold)
    
    return int(hard_events)


def check_collision(trace_df: pd.DataFrame) -> bool:
    """
    Check if a collision occurred (distance <= 0 or very close).
    
    Args:
        trace_df: DataFrame with column: distance_to_lead
        
    Returns:
        True if collision detected
    """
    min_distance = trace_df['distance_to_lead'].min()
    return min_distance < 0.5  # 0.5m threshold for collision


# ============================================================================
# OBJECTIVE FUNCTION 1: SAFETY SCORE
# ============================================================================

def calculate_safety_score(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate safety score (MINIMIZE to find unsafe scenarios).
    
    Lower score = more unsafe
    - Collision = very low score
    - Low TTC = low score
    - Small minimum distance = low score
    
    Args:
        trace_df: Simulation trace DataFrame
        dt: Time step
        
    Returns:
        Safety score (lower = more unsafe). Range: [0, 100]
    """
    # Check for collision (worst case)
    collision = check_collision(trace_df)
    if collision:
        return 0.0  # Lowest possible safety score
    
    # Calculate safety metrics
    min_ttc = calculate_minimum_ttc(trace_df)
    min_distance = calculate_minimum_distance(trace_df)
    
    # Normalize TTC (0 at critical, 100 at safe)
    critical_ttc = PLAUSIBILITY_CONSTRAINTS['critical_ttc']
    safe_ttc = 10.0  # 10 seconds is very safe
    if min_ttc == float('inf'):
        ttc_score = 100.0
    else:
        ttc_score = np.clip((min_ttc - critical_ttc) / (safe_ttc - critical_ttc) * 100, 0, 100)
    
    # Normalize distance (0 at critical, 100 at safe)
    critical_distance = PLAUSIBILITY_CONSTRAINTS['min_safe_distance']
    safe_distance = 30.0  # 30 meters is very safe
    distance_score = np.clip((min_distance - critical_distance) / (safe_distance - critical_distance) * 100, 0, 100)
    
    # Combined safety score (weighted average)
    # TTC is more important (60%) than distance (40%)
    safety_score = 0.6 * ttc_score + 0.4 * distance_score
    
    return float(safety_score)


# ============================================================================
# OBJECTIVE FUNCTION 2: PLAUSIBILITY SCORE
# ============================================================================

def calculate_plausibility_score(trace_df: pd.DataFrame, 
                                 lead_actions: Optional[Dict] = None,
                                 dt: float = 0.1) -> float:
    """
    Calculate plausibility score (MAXIMIZE for realistic scenarios).
    
    Higher score = more physically plausible
    - Excessive acceleration = low score
    - Excessive jerk = low score
    - Unrealistic lead vehicle behavior = low score
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional dict with lead vehicle behavior parameters
        dt: Time step
        
    Returns:
        Plausibility score (higher = more plausible). Range: [0, 100]
    """
    # Calculate dynamics metrics
    max_accel = calculate_maximum_acceleration(trace_df, dt)
    max_jerk = calculate_maximum_jerk(trace_df, dt)
    
    # Score acceleration plausibility
    max_acceptable_accel = PLAUSIBILITY_CONSTRAINTS['max_longitudinal_accel']
    comfortable_accel = PLAUSIBILITY_CONSTRAINTS['comfortable_max_accel']
    
    if max_accel > max_acceptable_accel:
        accel_score = 0.0  # Physically implausible
    elif max_accel > comfortable_accel:
        # Linearly decrease from 100 to 50 between comfortable and max
        accel_score = 100 - 50 * (max_accel - comfortable_accel) / (max_acceptable_accel - comfortable_accel)
    else:
        accel_score = 100.0  # Very plausible
    
    # Score jerk plausibility
    max_acceptable_jerk = PLAUSIBILITY_CONSTRAINTS['max_jerk']
    comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']
    
    if max_jerk > max_acceptable_jerk:
        jerk_score = 0.0  # Physically implausible
    elif max_jerk > comfortable_jerk:
        jerk_score = 100 - 50 * (max_jerk - comfortable_jerk) / (max_acceptable_jerk - comfortable_jerk)
    else:
        jerk_score = 100.0  # Very plausible
    
    # Combined plausibility score
    plausibility_score = 0.5 * accel_score + 0.5 * jerk_score
    
    return float(plausibility_score)


# ============================================================================
# OBJECTIVE FUNCTION 3: COMFORT SCORE
# ============================================================================

def calculate_comfort_score(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate passenger comfort score (MINIMIZE to find uncomfortable scenarios).
    
    Lower score = more uncomfortable
    - High jerk = low score
    - Many hard braking events = low score
    - High total accumulated jerk = low score
    
    Args:
        trace_df: Simulation trace DataFrame
        dt: Time step
        
    Returns:
        Comfort score (lower = more uncomfortable). Range: [0, 100]
    """
    # Calculate comfort metrics
    max_jerk = calculate_maximum_jerk(trace_df, dt)
    total_jerk = calculate_total_jerk(trace_df, dt)
    hard_events = count_hard_events(trace_df, dt, threshold=3.0)
    
    # Score maximum jerk
    comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']
    max_jerk_threshold = 10.0  # Very uncomfortable
    
    jerk_score = np.clip((max_jerk_threshold - max_jerk) / (max_jerk_threshold - comfortable_jerk) * 100, 0, 100)
    
    # Score total jerk (normalized by trace length)
    trace_length = len(trace_df)
    avg_jerk_per_step = total_jerk / max(trace_length, 1)
    comfortable_avg_jerk = 0.5
    max_avg_jerk = 5.0
    
    total_jerk_score = np.clip((max_avg_jerk - avg_jerk_per_step) / (max_avg_jerk - comfortable_avg_jerk) * 100, 0, 100)
    
    # Score hard events (normalized by trace length)
    hard_event_rate = hard_events / max(trace_length, 1) * 100  # Events per 100 timesteps
    max_acceptable_rate = 10.0  # 10 hard events per 100 timesteps
    
    hard_event_score = np.clip(100 - (hard_event_rate / max_acceptable_rate) * 100, 0, 100)
    
    # Combined comfort score (weighted average)
    comfort_score = 0.4 * jerk_score + 0.3 * total_jerk_score + 0.3 * hard_event_score
    
    return float(comfort_score)


# ============================================================================
# COMBINED EVALUATION
# ============================================================================

def evaluate_all_objectives(trace_df: pd.DataFrame, 
                           lead_actions: Optional[Dict] = None,
                           dt: float = 0.1) -> Dict[str, float]:
    """
    Evaluate all three objectives for a simulation trace.
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional lead vehicle behavior parameters
        dt: Time step
        
    Returns:
        Dictionary with all three objective scores
    """
    safety = calculate_safety_score(trace_df, dt)
    plausibility = calculate_plausibility_score(trace_df, lead_actions, dt)
    comfort = calculate_comfort_score(trace_df, dt)
    
    return {
        'safety': safety,
        'plausibility': plausibility,
        'comfort': comfort,
    }


def load_trace_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load simulation trace from CSV file.
    
    Args:
        csv_path: Path to CSV trace file
        
    Returns:
        DataFrame with trace data
    """
    return pd.read_csv(csv_path)


def evaluate_trace_file(csv_path: Path, dt: float = 0.1) -> Dict[str, float]:
    """
    Load and evaluate a trace file.
    
    Args:
        csv_path: Path to CSV trace file
        dt: Time step
        
    Returns:
        Dictionary with all objective scores
    """
    trace_df = load_trace_from_csv(csv_path)
    return evaluate_all_objectives(trace_df, dt=dt)


# ============================================================================
# PARETO DOMINANCE
# ============================================================================

def dominates(scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
    """
    Check if scores1 Pareto dominates scores2.
    
    For our objectives:
    - Safety: MINIMIZE (lower is better for finding unsafe scenarios)
    - Plausibility: MAXIMIZE (higher is better)
    - Comfort: MINIMIZE (lower is better for finding uncomfortable scenarios)
    
    Args:
        scores1: First set of objective scores
        scores2: Second set of objective scores
        
    Returns:
        True if scores1 dominates scores2
    """
    # Convert to tuple for comparison
    # For minimization: lower is better
    # For maximization: negate so lower is better
    s1 = (scores1['safety'], -scores1['plausibility'], scores1['comfort'])
    s2 = (scores2['safety'], -scores2['plausibility'], scores2['comfort'])
    
    # scores1 dominates if it's better in at least one and not worse in any
    better_in_at_least_one = any(s1[i] < s2[i] for i in range(3))
    not_worse_in_any = all(s1[i] <= s2[i] for i in range(3))
    
    return better_in_at_least_one and not_worse_in_any


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("OBJECTIVE FUNCTIONS MODULE")
    print("=" * 80)
    print("\nThis module provides three objective functions:")
    print("1. Safety Score (minimize): Lower = more unsafe scenarios")
    print("2. Plausibility Score (maximize): Higher = more realistic")
    print("3. Comfort Score (minimize): Lower = more uncomfortable")
    print("\nAll scores are in range [0, 100]")
    print("=" * 80)

