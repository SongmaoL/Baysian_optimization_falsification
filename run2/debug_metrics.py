"""
Debug: Check what values comfort and plausibility metrics are producing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'project_example_code'))

from config.search_space import PLAUSIBILITY_CONSTRAINTS

# Read a sample log
log_path = Path('logs/episode-scenario_0007.csv')
df = pd.read_csv(log_path)
dt = 0.1

print("=" * 80)
print("METRIC DEBUG - Scenario 0007")
print("=" * 80)

# Calculate acceleration
vel = df['ego_velocity'].values
accel = np.diff(vel) / dt
max_accel = np.max(np.abs(accel))

print(f"\n1. ACCELERATION")
print(f"   Max accel: {max_accel:.2f} m/s^2 ({max_accel/9.8:.2f}g)")
print(f"   Threshold (comfortable): {PLAUSIBILITY_CONSTRAINTS['comfortable_max_accel']} m/s^2")
print(f"   Threshold (max):         {PLAUSIBILITY_CONSTRAINTS['max_longitudinal_accel']} m/s^2")

# Calculate jerk
jerk = np.diff(accel) / dt
max_jerk = np.max(np.abs(jerk))
total_jerk = np.sum(np.abs(jerk))

print(f"\n2. JERK")
print(f"   Max jerk:   {max_jerk:.2f} m/s^3")
print(f"   Total jerk: {total_jerk:.2f} m/s^3")
print(f"   Threshold (comfortable): {PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']} m/s^3")
print(f"   Threshold (max):         {PLAUSIBILITY_CONSTRAINTS['max_jerk']} m/s^3")

# Count hard events (>3 m/s^2 = 0.3g)
hard_events = np.sum(np.abs(accel) > 3.0)
print(f"\n3. HARD EVENTS")
print(f"   Events > 0.3g: {hard_events}")
print(f"   Total timesteps: {len(accel)}")

# Calculate what the scores SHOULD be
print("\n" + "=" * 80)
print("PLAUSIBILITY SCORE BREAKDOWN")
print("=" * 80)

max_acceptable_accel = PLAUSIBILITY_CONSTRAINTS['max_longitudinal_accel']
comfortable_accel = PLAUSIBILITY_CONSTRAINTS['comfortable_max_accel']

if max_accel > max_acceptable_accel:
    accel_score = 0.0
    print(f"\nAccel score: 0 (max_accel {max_accel:.1f} > threshold {max_acceptable_accel})")
elif max_accel > comfortable_accel:
    accel_score = 100 - 50 * (max_accel - comfortable_accel) / (max_acceptable_accel - comfortable_accel)
    print(f"\nAccel score: {accel_score:.1f} (between comfortable and max)")
else:
    accel_score = 100.0
    print(f"\nAccel score: 100 (below comfortable threshold)")

max_acceptable_jerk = PLAUSIBILITY_CONSTRAINTS['max_jerk']
comfortable_jerk = PLAUSIBILITY_CONSTRAINTS['comfortable_max_jerk']

if max_jerk > max_acceptable_jerk:
    jerk_score = 0.0
    print(f"Jerk score: 0 (max_jerk {max_jerk:.1f} > threshold {max_acceptable_jerk})")
elif max_jerk > comfortable_jerk:
    jerk_score = 100 - 50 * (max_jerk - comfortable_jerk) / (max_acceptable_jerk - comfortable_jerk)
    print(f"Jerk score: {jerk_score:.1f} (between comfortable and max)")
else:
    jerk_score = 100.0
    print(f"Jerk score: 100 (below comfortable threshold)")

plausibility = 0.5 * accel_score + 0.5 * jerk_score
print(f"\nFinal plausibility: {plausibility:.1f}")

print("\n" + "=" * 80)
print("COMFORT SCORE BREAKDOWN")
print("=" * 80)

# Jerk score
max_jerk_threshold = 10.0
jerk_score_c = np.clip((max_jerk_threshold - max_jerk) / (max_jerk_threshold - comfortable_jerk) * 100, 0, 100)
print(f"\nMax jerk score: {jerk_score_c:.1f}")

# Total jerk score
trace_length = len(df)
avg_jerk_per_step = total_jerk / max(trace_length, 1)
comfortable_avg_jerk = 0.5
max_avg_jerk = 5.0
total_jerk_score = np.clip((max_avg_jerk - avg_jerk_per_step) / (max_avg_jerk - comfortable_avg_jerk) * 100, 0, 100)
print(f"Avg jerk: {avg_jerk_per_step:.2f}, Score: {total_jerk_score:.1f}")

# Hard event score
hard_event_rate = hard_events / max(trace_length, 1) * 100
max_acceptable_rate = 10.0
hard_event_score = np.clip(100 - (hard_event_rate / max_acceptable_rate) * 100, 0, 100)
print(f"Hard event rate: {hard_event_rate:.1f}%, Score: {hard_event_score:.1f}")

comfort = 0.4 * jerk_score_c + 0.3 * total_jerk_score + 0.3 * hard_event_score
print(f"\nFinal comfort: {comfort:.1f}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if max_jerk > max_acceptable_jerk:
    print("\n[!] JERK TOO HIGH - causing plausibility to drop")
    print(f"    max_jerk={max_jerk:.1f} but threshold={max_acceptable_jerk}")
    print("    Consider increasing max_jerk threshold in search_space.py")

if max_accel > max_acceptable_accel:
    print("\n[!] ACCEL TOO HIGH - causing plausibility to be 0")
    print(f"    max_accel={max_accel:.1f} but threshold={max_acceptable_accel}")
    
if hard_event_rate > 50:
    print("\n[!] TOO MANY HARD EVENTS - causing comfort to drop")
    print(f"    {hard_events} hard events out of {trace_length} timesteps")

if avg_jerk_per_step > max_avg_jerk:
    print("\n[!] AVG JERK TOO HIGH - comfort total_jerk_score = 0")
    print(f"    avg_jerk={avg_jerk_per_step:.2f} but max_avg={max_avg_jerk}")

print("=" * 80)

