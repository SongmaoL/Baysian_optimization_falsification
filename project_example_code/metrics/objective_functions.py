"""
Objective Functions for Multi-Objective Falsification

This module implements the objective functions:
1. Safety Score - based on Min TTC (Time-to-Collision), a CONTINUOUS metric
   that provides better gradient information for Bayesian Optimization
   (minimize to find unsafe scenarios)
2. Plausibility Score - measures physical realism (maximize for realistic scenarios)

CHANGE LOG:
- Dec 2024: Changed Safety Score from binary (collision=0, safe=60) to 
  continuous Min TTC-based scoring. This allows BO to learn gradients
  and produces more diverse Pareto fronts.

Note: Comfort Score has been removed from the optimization as it wasn't providing
useful signal. The function is kept for legacy compatibility but not used.
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
        
        # Filter out artifacts at low speeds (< 2.0 m/s)
        # Instant stopping in sim causes infinite jerk/accel spikes
        # Use speed at start of interval (aligned with jerk array)
        speeds = velocities[:-2] 
        valid_mask = speeds > 2.0
        
        if np.any(valid_mask):
            return np.max(np.abs(jerks[valid_mask]))
    
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
    
    # Filter out artifacts at low speeds (< 2.0 m/s)
    # Instant stopping in sim causes infinite accel spikes
    speeds = velocities[:-1]
    valid_mask = speeds > 2.0
    
    if np.any(valid_mask):
        return np.max(np.abs(accelerations[valid_mask]))
        
    return 0.0


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
    Check if a collision occurred.
    
    Uses the 'collided' column if available (from simulator's collision sensor),
    otherwise falls back to distance heuristic.
    
    Args:
        trace_df: DataFrame with columns: distance_to_lead, optionally 'collided'
        
    Returns:
        True if collision detected
    """
    # Prefer actual collision flag from simulator (Fix #4)
    if 'collided' in trace_df.columns:
        return trace_df['collided'].max() > 0  # Any collision in trace
    
    # Fallback: distance-based heuristic
    min_distance = trace_df['distance_to_lead'].min()
    return min_distance < 0.5  # 0.5m threshold for collision


# ============================================================================
# OBJECTIVE FUNCTION 1: SAFETY SCORE (Min TTC based - CONTINUOUS)
# ============================================================================

def calculate_safety_score(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate safety score using Min TTC (MINIMIZE to find unsafe scenarios).
    
    This uses a CONTINUOUS metric based on minimum Time-to-Collision (TTC),
    which provides better gradient information for Bayesian Optimization
    compared to binary collision detection.
    
    Lower score = more unsafe (lower TTC = closer to collision)
    
    Scoring scale:
    - TTC = 0s (collision) -> Safety = 0
    - TTC < 1.5s (critical) -> Safety < 25
    - TTC 1.5-4s (dangerous) -> Safety 25-67
    - TTC > 4s (safe) -> Safety 67-100
    - TTC = inf (never closing) -> Safety = 100
    
    Args:
        trace_df: Simulation trace DataFrame
        dt: Time step
        
    Returns:
        Safety score (lower = more unsafe). Range: [0, 100]
    """
    # Check for actual collision first (from simulator sensor)
    collision = check_collision(trace_df)
    if collision:
        # Even with collision, use TTC to differentiate severity
        min_ttc = calculate_minimum_ttc(trace_df)
        # Collision with very low TTC = 0, with higher TTC = slight score
        # This handles cases where collision happened after a close call
        return float(np.clip(min_ttc * 5, 0, 10))  # Max 10 for collisions
    
    # Calculate minimum TTC (primary metric)
    min_ttc = calculate_minimum_ttc(trace_df)
    
    # Handle infinite TTC (vehicles never closing)
    if min_ttc == float('inf'):
        return 100.0  # Perfectly safe - never approached
    
    # Convert Min TTC to safety score (continuous mapping)
    # Using a piecewise linear function for interpretability:
    #   TTC <= 0.5s  -> Score 0-8   (imminent collision)
    #   TTC 0.5-1.5s -> Score 8-25  (critical)
    #   TTC 1.5-4.0s -> Score 25-67 (dangerous)
    #   TTC 4.0-8.0s -> Score 67-90 (caution)
    #   TTC > 8.0s   -> Score 90-100 (safe)
    
    if min_ttc <= 0.5:
        # Imminent collision zone: 0 -> 8
        safety_score = min_ttc / 0.5 * 8
    elif min_ttc <= 1.5:
        # Critical zone: 8 -> 25
        safety_score = 8 + (min_ttc - 0.5) / 1.0 * 17
    elif min_ttc <= 4.0:
        # Dangerous zone: 25 -> 67
        safety_score = 25 + (min_ttc - 1.5) / 2.5 * 42
    elif min_ttc <= 8.0:
        # Caution zone: 67 -> 90
        safety_score = 67 + (min_ttc - 4.0) / 4.0 * 23
    else:
        # Safe zone: 90 -> 100 (asymptotic)
        safety_score = 90 + 10 * (1 - np.exp(-(min_ttc - 8.0) / 4.0))
    
    return float(np.clip(safety_score, 0, 100))


# ============================================================================
# OBJECTIVE FUNCTION 2: PLAUSIBILITY SCORE
# ============================================================================

def calculate_lead_plausibility(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate plausibility of lead vehicle behavior.
    
    Checks if lead vehicle dynamics are physically realistic:
    - Maximum acceleration within tire/engine limits
    - Jerk within realistic bounds
    
    Args:
        trace_df: Simulation trace with 'lead_speed' column
        dt: Time step
        
    Returns:
        Lead plausibility score [0, 100]
    """
    if 'lead_speed' not in trace_df.columns:
        return 100.0  # Can't check, assume plausible
    
    lead_velocities = trace_df['lead_speed'].values
    
    # Calculate lead acceleration
    lead_accels = np.diff(lead_velocities) / dt
    max_lead_accel = np.max(np.abs(lead_accels)) if len(lead_accels) > 0 else 0.0
    
    # Calculate lead jerk
    if len(lead_accels) > 1:
        lead_jerks = np.diff(lead_accels) / dt
        max_lead_jerk = np.max(np.abs(lead_jerks))
    else:
        max_lead_jerk = 0.0
    
    # Lead vehicle thresholds (can be more aggressive than ego for realism)
    # Real vehicles: max braking ~10-12 m/s², max jerk ~30-50 m/s³ in emergency
    max_acceptable_lead_accel = 12.0  # m/s²
    max_acceptable_lead_jerk = 40.0   # m/s³
    
    # Score acceleration
    if max_lead_accel > max_acceptable_lead_accel:
        lead_accel_score = max(0, 100 - 50 * (max_lead_accel - max_acceptable_lead_accel) / 5.0)
    else:
        lead_accel_score = 100.0
    
    # Score jerk
    if max_lead_jerk > max_acceptable_lead_jerk:
        lead_jerk_score = max(0, 100 - 50 * (max_lead_jerk - max_acceptable_lead_jerk) / 20.0)
    else:
        lead_jerk_score = 100.0
    
    return 0.5 * lead_accel_score + 0.5 * lead_jerk_score


def calculate_plausibility_score(trace_df: pd.DataFrame, 
                                 lead_actions: Optional[Dict] = None,
                                 dt: float = 0.1) -> float:
    """
    Calculate plausibility score (MAXIMIZE for realistic scenarios).
    
    Higher score = more physically plausible
    - Excessive ego acceleration = low score
    - Excessive ego jerk = low score
    - Unrealistic lead vehicle behavior = low score (NEW)
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional dict with lead vehicle behavior parameters
        dt: Time step
        
    Returns:
        Plausibility score (higher = more plausible). Range: [0, 100]
    """
    # Calculate EGO dynamics metrics
    max_accel = calculate_maximum_acceleration(trace_df, dt)
    max_jerk = calculate_maximum_jerk(trace_df, dt)
    
    # Score ego acceleration plausibility
    max_acceptable_accel = PLAUSIBILITY_CONSTRAINTS['max_longitudinal_accel']
    comfortable_accel = PLAUSIBILITY_CONSTRAINTS['comfortable_max_accel']
    
    if max_accel > max_acceptable_accel:
        accel_score = 0.0  # Physically implausible
    elif max_accel > comfortable_accel:
        # Linearly decrease from 100 to 50 between comfortable and max
        accel_score = 100 - 50 * (max_accel - comfortable_accel) / (max_acceptable_accel - comfortable_accel)
    else:
        accel_score = 100.0  # Very plausible
    
    # Score ego jerk plausibility
    max_acceptable_jerk = PLAUSIBILITY_CONSTRAINTS['max_jerk']
    comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']
    
    if max_jerk > max_acceptable_jerk:
        jerk_score = 0.0  # Physically implausible
    elif max_jerk > comfortable_jerk:
        jerk_score = 100 - 50 * (max_jerk - comfortable_jerk) / (max_acceptable_jerk - comfortable_jerk)
    else:
        jerk_score = 100.0  # Very plausible
    
    # Calculate LEAD vehicle plausibility (NEW)
    lead_score = calculate_lead_plausibility(trace_df, dt)
    
    # Combined plausibility score
    # Weight: 40% ego accel, 30% ego jerk, 30% lead behavior
    plausibility_score = 0.4 * accel_score + 0.3 * jerk_score + 0.3 * lead_score
    
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
    
    # Score maximum jerk (adjusted for realistic emergency braking)
    # Real emergency braking: 20-50 m/s³, so we need wider thresholds
    comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']  # 2.0 m/s³
    noticeable_jerk = PLAUSIBILITY_CONSTRAINTS['noticeable_jerk']        # 5.0 m/s³
    uncomfortable_jerk = 15.0   # Uncomfortable (hard braking)
    emergency_jerk = PLAUSIBILITY_CONSTRAINTS['max_jerk']                # 25.0 m/s³
    
    if max_jerk <= comfortable_jerk:
        jerk_score = 100.0  # Very comfortable
    elif max_jerk <= noticeable_jerk:
        # Linear: 100 -> 80
        jerk_score = 100.0 - 20.0 * (max_jerk - comfortable_jerk) / (noticeable_jerk - comfortable_jerk)
    elif max_jerk <= uncomfortable_jerk:
        # Linear: 80 -> 50
        jerk_score = 80.0 - 30.0 * (max_jerk - noticeable_jerk) / (uncomfortable_jerk - noticeable_jerk)
    elif max_jerk <= emergency_jerk:
        # Linear: 50 -> 20 (emergency braking is uncomfortable but realistic)
        jerk_score = 50.0 - 30.0 * (max_jerk - uncomfortable_jerk) / (emergency_jerk - uncomfortable_jerk)
    else:
        # Beyond emergency: 20 -> 0
        jerk_score = max(0.0, 20.0 - 20.0 * (max_jerk - emergency_jerk) / 25.0)
    
    jerk_score = np.clip(jerk_score, 0.0, 100.0)
    
    # Score total jerk (normalized by trace length) - improved thresholds
    trace_length = len(trace_df)
    if trace_length == 0:
        return 50.0  # Default neutral score for empty traces
    
    avg_jerk_per_step = total_jerk / trace_length
    comfortable_avg_jerk = 0.3  # Lower threshold for comfortable
    acceptable_avg_jerk = 1.0   # Acceptable average
    very_uncomfortable_avg_jerk = 3.0  # Very uncomfortable
    
    if avg_jerk_per_step <= comfortable_avg_jerk:
        total_jerk_score = 100.0
    elif avg_jerk_per_step <= acceptable_avg_jerk:
        total_jerk_score = 100.0 - 30.0 * (avg_jerk_per_step - comfortable_avg_jerk) / (acceptable_avg_jerk - comfortable_avg_jerk)
    elif avg_jerk_per_step <= very_uncomfortable_avg_jerk:
        total_jerk_score = 70.0 - 50.0 * (avg_jerk_per_step - acceptable_avg_jerk) / (very_uncomfortable_avg_jerk - acceptable_avg_jerk)
    else:
        total_jerk_score = max(20.0, 20.0 - 10.0 * (avg_jerk_per_step - very_uncomfortable_avg_jerk) / 2.0)
    
    total_jerk_score = np.clip(total_jerk_score, 0.0, 100.0)
    
    # Score hard events (normalized by trace length) - improved formula
    hard_event_rate = hard_events / trace_length * 100  # Events per 100 timesteps
    comfortable_rate = 2.0   # 2 events per 100 timesteps is comfortable
    acceptable_rate = 5.0    # 5 events per 100 timesteps is acceptable
    uncomfortable_rate = 15.0  # 15 events per 100 timesteps is uncomfortable
    
    if hard_event_rate <= comfortable_rate:
        hard_event_score = 100.0
    elif hard_event_rate <= acceptable_rate:
        hard_event_score = 100.0 - 30.0 * (hard_event_rate - comfortable_rate) / (acceptable_rate - comfortable_rate)
    elif hard_event_rate <= uncomfortable_rate:
        hard_event_score = 70.0 - 50.0 * (hard_event_rate - acceptable_rate) / (uncomfortable_rate - acceptable_rate)
    else:
        hard_event_score = max(20.0, 20.0 - 10.0 * (hard_event_rate - uncomfortable_rate) / 10.0)
    
    hard_event_score = np.clip(hard_event_score, 0.0, 100.0)
    
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
    Evaluate objectives for a simulation trace.
    
    Objectives:
    - Safety: Lower = more unsafe (we want to find unsafe scenarios)
    - Plausibility: Higher = more realistic (we want realistic scenarios)
    
    Note: Comfort metric removed as it wasn't providing useful signal.
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional lead vehicle behavior parameters
        dt: Time step
        
    Returns:
        Dictionary with objective scores
    """
    safety = calculate_safety_score(trace_df, dt)
    plausibility = calculate_plausibility_score(trace_df, lead_actions, dt)
    
    return {
        'safety': safety,
        'plausibility': plausibility,
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
    
    Args:
        scores1: First set of objective scores
        scores2: Second set of objective scores
        
    Returns:
        True if scores1 dominates scores2
    """
    # Convert to tuple for comparison
    # For minimization: lower is better
    # For maximization: negate so lower is better
    s1 = (scores1['safety'], -scores1['plausibility'])
    s2 = (scores2['safety'], -scores2['plausibility'])
    
    # scores1 dominates if it's better in at least one and not worse in any
    better_in_at_least_one = any(s1[i] < s2[i] for i in range(2))
    not_worse_in_any = all(s1[i] <= s2[i] for i in range(2))
    
    return better_in_at_least_one and not_worse_in_any


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("OBJECTIVE FUNCTIONS MODULE (Min TTC Version)")
    print("=" * 80)
    print("\nThis module provides two objective functions:")
    print("1. Safety Score (minimize): Based on Min TTC (continuous)")
    print("   - TTC < 1.5s -> Score < 25 (critical)")
    print("   - TTC 1.5-4s -> Score 25-67 (dangerous)")
    print("   - TTC > 4s   -> Score > 67 (safe)")
    print("2. Plausibility Score (maximize): Higher = more realistic")
    print("\nAll scores are in range [0, 100]")
    print("\nThe Min TTC metric provides smooth gradients for BO optimization.")
    print("=" * 80)

