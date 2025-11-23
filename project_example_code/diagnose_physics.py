"""
Diagnose: What are the actual acceleration and jerk values?
Are they physically realistic or is CARLA/calculation broken?
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

# Load results
with open('final_results.json') as f:
    data = json.load(f)

print("=" * 80)
print("PHYSICS DIAGNOSIS")
print("=" * 80)

accel_values = []
jerk_values = []
samples_checked = 0

for entry in data['evaluation_history'][:20]:  # Check first 20
    if 'metadata' not in entry or 'log_path' not in entry['metadata']:
        continue
    
    log_path = Path(entry['metadata']['log_path'])
    if not log_path.exists():
        continue
    
    try:
        df = pd.read_csv(log_path)
        dt = 0.1
        
        # Calculate acceleration
        vel = df['ego_velocity'].values
        accel = np.diff(vel) / dt
        max_accel = np.max(np.abs(accel))
        
        # Calculate jerk
        if len(accel) > 1:
            jerk = np.diff(accel) / dt
            max_jerk = np.max(np.abs(jerk))
        else:
            max_jerk = 0.0
        
        accel_values.append(max_accel)
        jerk_values.append(max_jerk)
        samples_checked += 1
        
        # Show scenarios exceeding limits
        if max_accel > 20.0 or max_jerk > 10.0:
            print(f"\nIter {entry['iteration']:03d}: EXCEEDS LIMITS")
            print(f"  Max accel: {max_accel:.1f} m/s² ({max_accel/9.8:.2f}g)")
            print(f"  Max jerk:  {max_jerk:.1f} m/s³")
            print(f"  Plausibility: {entry['objectives']['plausibility']:.1f}")
            
            # Show velocity profile around max accel
            idx_max = np.argmax(np.abs(accel))
            start = max(0, idx_max - 2)
            end = min(len(vel), idx_max + 3)
            print(f"  Velocity profile: {vel[start:end]}")
        
    except Exception as e:
        print(f"Error iter {entry['iteration']}: {e}")

if samples_checked == 0:
    print("\n❌ No log files found! Check paths in final_results.json")
    exit(1)

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Samples checked: {samples_checked}")
print(f"\nAcceleration:")
print(f"  Min:    {min(accel_values):.1f} m/s² ({min(accel_values)/9.8:.2f}g)")
print(f"  Max:    {max(accel_values):.1f} m/s² ({max(accel_values)/9.8:.2f}g)")
print(f"  Median: {np.median(accel_values):.1f} m/s² ({np.median(accel_values)/9.8:.2f}g)")
print(f"  Mean:   {np.mean(accel_values):.1f} m/s² ({np.mean(accel_values)/9.8:.2f}g)")

print(f"\nJerk:")
print(f"  Min:    {min(jerk_values):.1f} m/s³")
print(f"  Max:    {max(jerk_values):.1f} m/s³")
print(f"  Median: {np.median(jerk_values):.1f} m/s³")
print(f"  Mean:   {np.mean(jerk_values):.1f} m/s³")

# Check against thresholds
exceed_2g = sum(1 for a in accel_values if a > 20.0)
exceed_jerk = sum(1 for j in jerk_values if j > 10.0)

print(f"\n📊 Exceeding 2g (20 m/s²):  {exceed_2g}/{samples_checked} ({exceed_2g/samples_checked*100:.0f}%)")
print(f"📊 Exceeding 10 m/s³ jerk: {exceed_jerk}/{samples_checked} ({exceed_jerk/samples_checked*100:.0f}%)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

avg_accel_g = np.mean(accel_values) / 9.8

if max(accel_values) > 25.0:
    print("❌ PHYSICS VIOLATION: Acceleration > 2.5g is impossible for passenger cars")
    print("   → CARLA simulation is unrealistic (likely instant brake commands)")
    print("   → Options:")
    print("      1. Fix lead vehicle behavior (smooth out brake commands)")
    print("      2. Accept that 70% of scenarios are implausible")
    print("      3. Filter out implausible scenarios for analysis")
elif exceed_2g > samples_checked * 0.6:
    print("⚠️  MANY scenarios exceed 2g - CARLA may have aggressive physics")
    print("   → Lead car behavior params may cause instant braking")
    print("   → Check: lead_brake_intensity and brake transition smoothness")
else:
    print("✓ Most scenarios are within realistic bounds")
    print("  → Original 2g/10 m/s³ thresholds are appropriate")

print("=" * 80)

