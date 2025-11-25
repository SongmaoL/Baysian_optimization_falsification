# Run2 Analysis - 44 Iterations

## Summary

| Metric | Run1 (107 iter) | Run2 (44 iter) | Improvement |
|--------|-----------------|----------------|-------------|
| Plausibility non-zero | 29% | **81.8%** | ✅ +52.8% |
| Comfort non-zero | 30% | 2.3% | ❌ Worse |
| Safety range | 0-95.79 | 7.65-39.20 | Different |
| Max plausibility | 100 | 50 | ❌ Capped |

## Key Findings

### ✅ Plausibility FIXED (Partially)
- **81.8%** of scenarios now have non-zero plausibility (was 29%)
- Acceleration smoothing is working - max accel is ~12 m/s² (1.2g) ✓
- But plausibility caps at ~50 because **jerk is still too high**

### ❌ Comfort Still Broken
- Only **1 out of 44** scenarios has non-zero comfort (2.3%)
- Root cause identified: **JERK**

### Debug Output (Scenario 0007)
```
Acceleration: 11.67 m/s² (1.19g) ✓ Good!
Jerk:         90.43 m/s³        ❌ WAY TOO HIGH!
              (threshold: 10 m/s³)
```

## Root Cause: High Jerk

The smoothing fix (alpha=0.3) helps acceleration but not jerk:
- **Acceleration** = velocity change per timestep
- **Jerk** = acceleration change per timestep (derivative of acceleration)

Even with smoothing, the acceleration *rate of change* is still too fast:
- Current alpha=0.3 allows 30% of target change per timestep
- This creates jerk spikes during brake transitions

## Recommended Fix

### Option 1: Reduce Smoothing Alpha (Recommended)

Change in `scenario_generator.py`:
```python
# Current
SMOOTHING_ALPHA = 0.3  # Too fast

# Recommended
SMOOTHING_ALPHA = 0.1  # Much smoother
```

This will:
- Make transitions take ~1 second instead of ~0.3 seconds
- Reduce jerk from ~90 m/s³ to ~10-20 m/s³
- Keep scenarios realistic

### Option 2: Add Second-Order Smoothing

Apply smoothing twice (to both acceleration and jerk):
```python
# Smooth the target
smooth_target = ALPHA * target + (1-ALPHA) * current

# Smooth the smooth target (reduces jerk)
final_value = ALPHA2 * smooth_target + (1-ALPHA2) * current
```

### Option 3: Relax Jerk Threshold

If high jerk is acceptable for CARLA:
```python
# In config/search_space.py
"max_jerk": 50.0,  # Was 10.0
"comfortable_max_jerk": 10.0,  # Was 2.0
```

**Not recommended** - 90 m/s³ jerk is physically unrealistic.

## Best Plausible Unsafe Scenario

**Iteration 7:**
- Safety: 13.71 (unsafe - good for falsification!)
- Plausibility: 45.83 (realistic-ish)
- Comfort: 0.00 (jerk too high)

This is close to what we want! With the jerk fix, we'd get:
- Plausibility ~80-90
- Comfort ~50-70

## Next Steps

1. **Apply jerk fix** - Reduce SMOOTHING_ALPHA to 0.1
2. **Re-run falsification** - New run with fixed scenario generator
3. **Target 200+ iterations** - Get proper Pareto front
4. **Analyze results** - Should see ~70%+ non-zero for all metrics

## Files Created

- `run2/analyze_run2.py` - Analysis script
- `run2/debug_metrics.py` - Metric debugging script

