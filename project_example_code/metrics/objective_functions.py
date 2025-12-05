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
  
- Dec 2024: Improved Plausibility Score:
  * Uses 95th percentile instead of max (ignores numerical spikes)
  * Relaxed jerk thresholds: 25→40 m/s³ (ego), 40→60 m/s³ (lead)
  * Context-aware: higher jerk acceptable during emergencies (TTC < 2s)
  * Added weather consistency check (fog+sun, rain+clouds conflicts)
  * Updated weights: accel 35%, jerk 25%, lead 25%, weather 15%

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
# OBJECTIVE FUNCTION 2: PLAUSIBILITY SCORE (IMPROVED)
# ============================================================================
# Improvements made Dec 2024:
# 1. Relaxed jerk thresholds (25→40 ego, 40→60 lead) for realistic emergency braking
# 2. Use 95th percentile instead of max to ignore brief numerical spikes
# 3. Added weather consistency check
# 4. Context-aware scoring: higher jerk acceptable during emergencies

def calculate_weather_plausibility(weather_params: Optional[Dict] = None) -> float:
    """
    Check if weather conditions are physically consistent.
    
    Unrealistic combinations:
    - Heavy fog + bright sun
    - Heavy rain + no clouds
    - Extreme combinations
    
    Args:
        weather_params: Dict with fog_density, precipitation, sun_altitude_angle, cloudiness
        
    Returns:
        Weather plausibility score [0, 100]
    """
    if weather_params is None:
        return 100.0
    
    score = 100.0
    
    fog = weather_params.get('fog_density', 0)
    rain = weather_params.get('precipitation', 0)
    sun = weather_params.get('sun_altitude_angle', 45)
    clouds = weather_params.get('cloudiness', 50)
    
    # Heavy fog + bright midday sun is unrealistic
    if fog > 50 and sun > 60:
        penalty = min(30, (fog - 50) * 0.5 + (sun - 60) * 0.5)
        score -= penalty
    
    # Heavy rain + clear sky (no clouds) is unrealistic
    if rain > 50 and clouds < 30:
        penalty = min(30, (rain - 50) * 0.4 + (30 - clouds) * 0.4)
        score -= penalty
    
    # Extreme fog (>70%) + any significant rain is rare
    if fog > 70 and rain > 30:
        score -= 15
    
    return max(0, score)


def calculate_percentile_jerk(trace_df: pd.DataFrame, dt: float = 0.1, 
                               percentile: float = 95) -> float:
    """
    Calculate jerk at given percentile (instead of max) to ignore brief spikes.
    
    Args:
        trace_df: DataFrame with ego_velocity column
        dt: Time step
        percentile: Percentile to use (default 95th)
        
    Returns:
        Jerk value at specified percentile
    """
    velocities = trace_df['ego_velocity'].values
    
    if len(velocities) < 3:
        return 0.0
    
    # Calculate acceleration
    accelerations = np.diff(velocities) / dt
    
    # Calculate jerk
    jerks = np.diff(accelerations) / dt
    
    # Filter out low-speed artifacts
    speeds = velocities[:-2]
    valid_mask = speeds > 2.0
    
    if np.any(valid_mask):
        valid_jerks = np.abs(jerks[valid_mask])
        if len(valid_jerks) > 0:
            return np.percentile(valid_jerks, percentile)
    
    return 0.0


def calculate_lead_plausibility(trace_df: pd.DataFrame, dt: float = 0.1) -> float:
    """
    Calculate plausibility of lead vehicle behavior (IMPROVED).
    
    Uses 95th percentile and relaxed thresholds for realistic emergency braking.
    
    Args:
        trace_df: Simulation trace with 'lead_speed' column
        dt: Time step
        
    Returns:
        Lead plausibility score [0, 100]
    """
    if 'lead_speed' not in trace_df.columns:
        return 100.0
    
    lead_velocities = trace_df['lead_speed'].values
    
    if len(lead_velocities) < 2:
        return 100.0
    
    # Calculate lead acceleration
    lead_accels = np.diff(lead_velocities) / dt
    
    # Use 95th percentile instead of max to ignore brief spikes
    accel_95 = np.percentile(np.abs(lead_accels), 95) if len(lead_accels) > 0 else 0.0
    
    # Calculate lead jerk
    if len(lead_accels) > 1:
        lead_jerks = np.diff(lead_accels) / dt
        jerk_95 = np.percentile(np.abs(lead_jerks), 95) if len(lead_jerks) > 0 else 0.0
    else:
        jerk_95 = 0.0
    
    # RELAXED thresholds for lead vehicle (emergency braking is realistic)
    # Real vehicles: max braking ~10-12 m/s², emergency jerk ~40-80 m/s³
    max_acceptable_lead_accel = 14.0  # m/s² (relaxed from 12)
    max_acceptable_lead_jerk = 60.0   # m/s³ (relaxed from 40)
    comfortable_lead_accel = 8.0
    comfortable_lead_jerk = 30.0
    
    # Score acceleration (gradual penalty)
    if accel_95 > max_acceptable_lead_accel:
        lead_accel_score = max(0, 50 - 25 * (accel_95 - max_acceptable_lead_accel) / 5.0)
    elif accel_95 > comfortable_lead_accel:
        lead_accel_score = 100 - 50 * (accel_95 - comfortable_lead_accel) / (max_acceptable_lead_accel - comfortable_lead_accel)
    else:
        lead_accel_score = 100.0
    
    # Score jerk (gradual penalty)
    if jerk_95 > max_acceptable_lead_jerk:
        lead_jerk_score = max(0, 50 - 25 * (jerk_95 - max_acceptable_lead_jerk) / 30.0)
    elif jerk_95 > comfortable_lead_jerk:
        lead_jerk_score = 100 - 50 * (jerk_95 - comfortable_lead_jerk) / (max_acceptable_lead_jerk - comfortable_lead_jerk)
    else:
        lead_jerk_score = 100.0
    
    return 0.5 * lead_accel_score + 0.5 * lead_jerk_score


def calculate_plausibility_score(trace_df: pd.DataFrame, 
                                 lead_actions: Optional[Dict] = None,
                                 weather_params: Optional[Dict] = None,
                                 dt: float = 0.1) -> float:
    """
    Calculate plausibility score (IMPROVED - MAXIMIZE for realistic scenarios).
    
    Improvements:
    - Uses 95th percentile instead of max (ignores brief spikes)
    - Relaxed jerk thresholds for realistic emergency braking
    - Added weather consistency check
    - Context-aware: emergency situations allow higher dynamics
    
    Components:
    - Ego acceleration (35% weight)
    - Ego jerk (25% weight)
    - Lead vehicle dynamics (25% weight)
    - Weather consistency (15% weight)
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional lead vehicle behavior parameters
        weather_params: Optional weather parameters for consistency check
        dt: Time step
        
    Returns:
        Plausibility score (higher = more plausible). Range: [0, 100]
    """
    # Calculate EGO dynamics using 95th percentile
    max_accel = calculate_maximum_acceleration(trace_df, dt)
    jerk_95 = calculate_percentile_jerk(trace_df, dt, percentile=95)
    
    # Check if this is an emergency situation (low TTC)
    min_ttc = calculate_minimum_ttc(trace_df)
    is_emergency = min_ttc < 2.0  # TTC under 2 seconds = emergency
    
    # RELAXED thresholds (especially for jerk during emergencies)
    max_acceptable_accel = PLAUSIBILITY_CONSTRAINTS['max_longitudinal_accel']  # 12 m/s²
    comfortable_accel = PLAUSIBILITY_CONSTRAINTS['comfortable_max_accel']      # 5 m/s²
    
    # Jerk thresholds - RELAXED from original 25 to 40 m/s³
    # During emergency (TTC < 2s), allow even higher jerk
    if is_emergency:
        max_acceptable_jerk = 50.0   # Allow higher jerk in emergencies
        comfortable_jerk = 15.0
    else:
        max_acceptable_jerk = 40.0   # Relaxed from 25
        comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']  # 2 m/s³
    
    # Score ego acceleration
    if max_accel > max_acceptable_accel:
        accel_score = max(0, 50 - 25 * (max_accel - max_acceptable_accel) / 5.0)
    elif max_accel > comfortable_accel:
        accel_score = 100 - 50 * (max_accel - comfortable_accel) / (max_acceptable_accel - comfortable_accel)
    else:
        accel_score = 100.0
    
    # Score ego jerk (using 95th percentile)
    if jerk_95 > max_acceptable_jerk:
        jerk_score = max(0, 50 - 25 * (jerk_95 - max_acceptable_jerk) / 20.0)
    elif jerk_95 > comfortable_jerk:
        jerk_score = 100 - 50 * (jerk_95 - comfortable_jerk) / (max_acceptable_jerk - comfortable_jerk)
    else:
        jerk_score = 100.0
    
    # Lead vehicle plausibility
    lead_score = calculate_lead_plausibility(trace_df, dt)
    
    # Weather consistency plausibility
    weather_score = calculate_weather_plausibility(weather_params)
    
    # Combined plausibility score with updated weights
    # Reduced jerk weight since we're now more lenient
    plausibility_score = (
        0.35 * accel_score +      # Ego acceleration
        0.25 * jerk_score +       # Ego jerk (reduced from 30%)
        0.25 * lead_score +       # Lead vehicle
        0.15 * weather_score      # Weather consistency (NEW)
    )
    
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
                           weather_params: Optional[Dict] = None,
                           dt: float = 0.1) -> Dict[str, float]:
    """
    Evaluate objectives for a simulation trace.
    
    Objectives:
    - Safety: Lower = more unsafe (based on Min TTC, continuous metric)
    - Plausibility: Higher = more realistic (improved with 95th percentile,
                    relaxed jerk thresholds, weather consistency check)
    
    Args:
        trace_df: Simulation trace DataFrame
        lead_actions: Optional lead vehicle behavior parameters
        weather_params: Optional weather parameters for consistency check
        dt: Time step
        
    Returns:
        Dictionary with objective scores
    """
    safety = calculate_safety_score(trace_df, dt)
    plausibility = calculate_plausibility_score(trace_df, lead_actions, weather_params, dt)
    
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

