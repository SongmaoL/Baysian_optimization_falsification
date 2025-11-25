"""
Analyze run2 results - 200 iterations
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_file = Path('results/final_results.json')
with open(results_file) as f:
    data = json.load(f)

print("=" * 80)
print("RUN2 ANALYSIS - 200 ITERATIONS")
print("=" * 80)

# Basic stats
n_evals = len(data['evaluation_history'])
print(f"\nTotal Evaluations: {n_evals}")
print(f"Parameter Dimensions: {len(data['parameter_bounds'])}")

# Extract objectives
safety = [e['objectives']['safety'] for e in data['evaluation_history']]
plausibility = [e['objectives']['plausibility'] for e in data['evaluation_history']]
comfort = [e['objectives']['comfort'] for e in data['evaluation_history']]

print("\n" + "=" * 80)
print("OBJECTIVE STATISTICS")
print("=" * 80)

print(f"\n1. SAFETY (minimize - lower = more unsafe)")
print(f"   Min:    {min(safety):.2f}")
print(f"   Max:    {max(safety):.2f}")
print(f"   Mean:   {np.mean(safety):.2f}")
print(f"   Std:    {np.std(safety):.2f}")

print(f"\n2. PLAUSIBILITY (maximize - higher = more realistic)")
print(f"   Min:    {min(plausibility):.2f}")
print(f"   Max:    {max(plausibility):.2f}")
print(f"   Mean:   {np.mean(plausibility):.2f}")
print(f"   Std:    {np.std(plausibility):.2f}")

print(f"\n3. COMFORT (minimize - lower = more uncomfortable)")
print(f"   Min:    {min(comfort):.2f}")
print(f"   Max:    {max(comfort):.2f}")
print(f"   Mean:   {np.mean(comfort):.2f}")
print(f"   Std:    {np.std(comfort):.2f}")

# Count non-zeros
non_zero_plaus = sum(1 for p in plausibility if p > 0)
non_zero_comfort = sum(1 for c in comfort if c > 0)

print("\n" + "=" * 80)
print("PLAUSIBILITY/COMFORT ISSUE CHECK")
print("=" * 80)
print(f"\nNon-zero plausibility: {non_zero_plaus}/{n_evals} ({non_zero_plaus/n_evals*100:.1f}%)")
print(f"Non-zero comfort:      {non_zero_comfort}/{n_evals} ({non_zero_comfort/n_evals*100:.1f}%)")

if non_zero_plaus / n_evals < 0.5:
    print("\n[!] STILL HAVE PLAUSIBILITY ISSUE - >50% scenarios are implausible")
else:
    print("\n[OK] Plausibility looks good - majority of scenarios are realistic")

if non_zero_comfort / n_evals < 0.5:
    print("[!] STILL HAVE COMFORT ISSUE - >50% scenarios have comfort=0")
else:
    print("[OK] Comfort looks good - majority of scenarios have non-zero comfort")

# Find best/worst scenarios
print("\n" + "=" * 80)
print("KEY SCENARIOS")
print("=" * 80)

best_safety_idx = safety.index(min(safety))
worst_safety_idx = safety.index(max(safety))

print(f"\n>> MOST UNSAFE (Iteration {data['evaluation_history'][best_safety_idx]['iteration']}):")
print(f"   Safety:       {safety[best_safety_idx]:.2f}")
print(f"   Plausibility: {plausibility[best_safety_idx]:.2f}")
print(f"   Comfort:      {comfort[best_safety_idx]:.2f}")

print(f"\n>> SAFEST (Iteration {data['evaluation_history'][worst_safety_idx]['iteration']}):")
print(f"   Safety:       {safety[worst_safety_idx]:.2f}")
print(f"   Plausibility: {plausibility[worst_safety_idx]:.2f}")
print(f"   Comfort:      {comfort[worst_safety_idx]:.2f}")

# Find best plausible unsafe scenario
plausible_unsafe = [(i, s, p, c) for i, (s, p, c) in enumerate(zip(safety, plausibility, comfort)) if p > 30]
if plausible_unsafe:
    plausible_unsafe.sort(key=lambda x: x[1])  # Sort by safety (ascending = more unsafe)
    best = plausible_unsafe[0]
    entry = data['evaluation_history'][best[0]]
    print(f"\n** BEST PLAUSIBLE UNSAFE (Iteration {entry['iteration']}):")
    print(f"   Safety:       {best[1]:.2f} (unsafe)")
    print(f"   Plausibility: {best[2]:.2f} (realistic)")
    print(f"   Comfort:      {best[3]:.2f}")
else:
    print("\n[X] No scenarios found with plausibility > 30")

# Convergence analysis - split into early/late
early = safety[:50]
late = safety[-50:]
print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS")
print("=" * 80)
print(f"\nFirst 50 iterations - Safety mean: {np.mean(early):.2f}")
print(f"Last 50 iterations  - Safety mean: {np.mean(late):.2f}")

if np.mean(late) < np.mean(early):
    print("[OK] BO is finding more unsafe scenarios over time (converging)")
else:
    print("[!] Not converging to more unsafe scenarios")

# Parameter analysis for unsafe scenarios
print("\n" + "=" * 80)
print("PARAMETER PATTERNS IN UNSAFE SCENARIOS")
print("=" * 80)

unsafe_entries = [e for e in data['evaluation_history'] if e['objectives']['safety'] < 30]
if len(unsafe_entries) >= 5:
    print(f"\nAnalyzing {len(unsafe_entries)} scenarios with safety < 30:")
    
    # Average parameters for unsafe scenarios
    params_unsafe = {key: np.mean([e['parameters'][key] for e in unsafe_entries]) 
                    for key in data['parameter_bounds'].keys()}
    
    print("\nKey parameter averages for unsafe scenarios:")
    print(f"   fog_density:          {params_unsafe['fog_density']:.1f}")
    print(f"   precipitation:        {params_unsafe['precipitation']:.1f}")
    print(f"   lead_brake_probability: {params_unsafe['lead_brake_probability']:.2f}")
    print(f"   lead_brake_intensity: {params_unsafe['lead_brake_intensity']:.2f}")
    print(f"   initial_distance:     {params_unsafe['initial_distance']:.1f} m")
    print(f"   initial_ego_velocity: {params_unsafe['initial_ego_velocity']:.1f} m/s")
else:
    print(f"\nOnly {len(unsafe_entries)} scenarios with safety < 30 (need more data)")

print("\n" + "=" * 80)
print("CONCLUSION & RECOMMENDATIONS")
print("=" * 80)

if non_zero_plaus / n_evals < 0.4:
    print("""
[!] PLAUSIBILITY ISSUE PERSISTS
   - Run2 was likely done BEFORE the smoothing fix was applied
   - 60%+ of scenarios still have plausibility=0
   - This means scenarios still violate physics (>2g acceleration)
   
   RECOMMENDATION:
   - Re-run falsification with the FIXED scenario_generator.py
   - The exponential smoothing fix (alpha=0.3) should solve this
""")
else:
    print("""
[OK] PLAUSIBILITY FIXED!
   - Majority of scenarios now have plausibility > 0
   - Smoothing fix is working
   
   Check comfort metric - if still low, investigate further
""")

print("=" * 80)

