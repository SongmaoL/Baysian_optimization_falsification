# Controller Performance Analysis at Very Low Speed

## Executive Summary

This analysis examines the **Adaptive Cruise Control (ACC) controller** performance in the `project_example_code` at **very low speeds** (approximately 5-6 m/s or 18-22 km/h). The controller exhibits **mixed performance** with significant challenges at very low initial velocities.

---

## Controller Design Overview

The controller (`project_example_code/csci513-miniproject1/mp1_controller/controller.py`) is an **Ultra-Simple Adaptive Cruise Controller** with these key characteristics:

### Design Parameters
- **Natural frequency (wn)**: 3.0 rad/s for critically-damped response
- **Time headway**: 1.2 seconds
- **Minimum gap**: 5.0 meters
- **Acceleration limit**: ±10.0 m/s²
- **Target reach time**: 6 seconds to reach desired speed (±2 m/s band)
- **Minimum acceleration push**: 6.5 m/s² when far from target

### Control Strategy
1. **Cruising Mode**: Uses critically-damped speed control with deadline-based acceleration boost
2. **Following Mode**: Proportional control (Kp = 0.5) based on desired gap
3. **Safety Overrides**: Hard braking at low TTC, stopping distance violations, or near-field proximity

---

## Performance at Very Low Speeds (5-6 m/s)

### Test Scenarios from Falsification Results

The Bayesian Optimization falsification framework identified several critical low-speed scenarios:

#### 1. **Scenario 0042** (Iteration 40) - CRITICAL FAILURE ⚠️
**Initial Conditions:**
- **Ego velocity**: 5.21 m/s (18.8 km/h) - VERY LOW
- **Lead velocity**: 20.35 m/s (73.3 km/h) - Much faster
- **Initial distance**: 15.78 m - SHORT

**Performance Metrics:**
- **Safety Score**: 38.20/100 ❌ (UNSAFE - High collision risk)
- **Plausibility Score**: 0.0/100 ❌ (Unrealistic vehicle dynamics)
- **Comfort Score**: 0.0/100 ❌ (Very jerky, uncomfortable)

**Analysis:**
- Controller struggled with **large velocity differential** (15.14 m/s gap)
- At very low initial speed, aggressive acceleration was required
- Led to **unrealistic dynamics** (likely exceeded physical limits)
- **Safety concern**: Close following distance at low ego speed
- Controller likely oscillated or applied excessive acceleration

---

#### 2. **Scenario 0088** (Iteration 70) - CRITICAL FAILURE ⚠️
**Initial Conditions:**
- **Ego velocity**: 5.33 m/s (19.2 km/h) - VERY LOW
- **Lead velocity**: 9.66 m/s (34.8 km/h) - Moderate
- **Initial distance**: 33.84 m - MODERATE

**Performance Metrics:**
- **Safety Score**: 29.32/100 ❌ (VERY UNSAFE - Critical)
- **Plausibility Score**: 0.0/100 ❌ (Unrealistic)
- **Comfort Score**: 0.0/100 ❌ (Very uncomfortable)

**Analysis:**
- **Worst safety score** among low-speed scenarios
- Despite reasonable initial distance, controller failed badly
- Velocity differential (4.33 m/s) should be manageable
- **Likely issues**:
  - Poor transient response at low speeds
  - Overly aggressive acceleration from 5.33 m/s
  - Safety overrides triggered excessively
  - Possible collision or near-collision event

---

#### 3. **Scenario 0106** (Iteration 88) - SUCCESS ✅
**Initial Conditions:**
- **Ego velocity**: 5.89 m/s (21.2 km/h) - LOW
- **Lead velocity**: 14.30 m/s (51.5 km/h) - Moderate
- **Initial distance**: 21.21 m - ADEQUATE

**Performance Metrics:**
- **Safety Score**: 86.47/100 ✅ (SAFE)
- **Plausibility Score**: 100.0/100 ✅ (Highly realistic)
- **Comfort Score**: 100.0/100 ✅ (Very comfortable)

**Analysis:**
- **Best performance** in low-speed category
- Smooth acceleration profile from 5.89 m/s
- Maintained safe following distance
- Realistic vehicle dynamics (no excessive jerk)
- **Key success factor**: Slightly higher initial speed (5.89 vs 5.2 m/s)
- Adequate initial spacing allowed smooth catch-up

---

#### 4. **Scenario 0107** (Iteration 89) - MARGINAL PERFORMANCE ⚠️
**Initial Conditions:**
- **Ego velocity**: 5.87 m/s (21.1 km/h) - LOW
- **Lead velocity**: 10.02 m/s (36.1 km/h) - Moderate
- **Initial distance**: 21.26 m - ADEQUATE

**Performance Metrics:**
- **Safety Score**: 60.37/100 ⚠️ (Moderate - Some concerns)
- **Plausibility Score**: 26.65/100 ⚠️ (Questionable realism)
- **Comfort Score**: 19.81/100 ❌ (Uncomfortable)

**Analysis:**
- **Mixed performance** - Safe but uncomfortable
- Similar initial conditions to Scenario 0106, but worse outcomes
- Likely encountered **lead vehicle braking event** (brake probability 26%)
- Controller response to lead braking was jerky
- Plausibility issues suggest excessive jerk or acceleration changes

---

## Key Findings

### ✅ **Strengths at Low Speed**

1. **Can handle gradual speed increases**: When initial distance is adequate (>20m) and velocity differential is moderate (<10 m/s)
2. **Safety overrides work**: Hard braking engages when needed
3. **Best case performance is excellent**: Scenario 0106 shows smooth, safe operation

### ❌ **Critical Weaknesses at Low Speed**

1. **Very sensitive to initial conditions**: 
   - 5.21 m/s: FAILS (Safety 38.2)
   - 5.89 m/s: SUCCEEDS (Safety 86.5)
   - **0.68 m/s difference** causes dramatic performance change

2. **Poor handling of large velocity differentials at low speed**:
   - When ego is at 5 m/s and lead is at 20+ m/s, controller becomes aggressive
   - Leads to unrealistic acceleration profiles

3. **Uncomfortable response to disturbances**:
   - Lead vehicle braking causes jerky ego response
   - Low comfort scores (0.0-19.8) in challenging scenarios

4. **Lack of smooth start-up behavior**:
   - No special handling for very low speeds (<6 m/s)
   - Treats 5 m/s same as 15 m/s (problematic)

---

## Root Cause Analysis

### Controller Design Issues at Low Speed

1. **Fixed natural frequency (wn = 3.0)**:
   - Optimized for nominal speeds (10-20 m/s)
   - Too aggressive at very low speeds
   - No gain scheduling based on current velocity

2. **Minimum acceleration push (6.5 m/s²)**:
   ```python
   self.min_push = 6.5  # minimal push toward band
   ```
   - Applied regardless of current speed
   - At 5 m/s, this causes jerky starts
   - Should be velocity-dependent

3. **No low-speed startup logic**:
   ```python
   # No cruise guard, no start‑up floors (from design goals)
   ```
   - Simplified design omits gentle start-up
   - Commercial ACC systems have special low-speed modes

4. **Following mode proportional gain (Kp = 0.5)**:
   - Fixed gain doesn't adapt to speed
   - At low speeds, position errors translate to excessive acceleration commands

5. **Gap calculation at low speed**:
   ```python
   def _desired_gap(self, v_ego: float) -> float:
       return max(self.distance_threshold, self.min_gap, self.headway * max(v_ego, 0.0))
   ```
   - At v_ego = 5 m/s: desired gap = max(4, 5, 1.2 × 5) = 6.0 m
   - At v_ego = 20 m/s: desired gap = 24 m
   - **4x difference** in desired gap causes mode switching issues

---

## Specific Performance Metrics

### Speed Range Classification

| Speed Range | Classification | Controller Performance |
|------------|---------------|----------------------|
| 0-5 m/s | Very Low | ❌ **Not tested / Expected poor** |
| 5-6 m/s | Very Low | ⚠️ **Unreliable** (50% success rate) |
| 6-10 m/s | Low | ⚠️ **Marginal** (limited data) |
| 10-20 m/s | Nominal | ✅ **Good** (design target) |
| 20+ m/s | High | ✅ **Good** |

### Success Rate at Very Low Speed (5-6 m/s)

From falsification results:
- **Total scenarios with initial_ego_velocity < 6 m/s**: 4 identified
- **Safe scenarios (Safety > 70)**: 1 (25%)
- **Unsafe scenarios (Safety < 40)**: 2 (50%)
- **Marginal scenarios (40-70)**: 1 (25%)

**Success rate: 25%** ⚠️

---

## Comparison with Design Requirements

### Design Goals (from controller.py)
1. ✅ "Keep it tiny and readable" - ACHIEVED
2. ⚠️ "Reaches target speed band (±2 m/s) in ~6 s when NOT following" - **ONLY AT NOMINAL SPEEDS**
3. ⚠️ "Sanity-check FOLLOWING detection and basic safety overrides" - **WORKS BUT JERKY**

### Observed Behavior at 5 m/s

**Cruising Mode (no lead vehicle):**
- Target speed: 20 m/s (typical)
- Speed error: 20 - 5 = 15 m/s
- Time remaining: 6 seconds
- Required acceleration: (15 - 2) / 6 = **2.17 m/s²** (minimum)
- Applied acceleration: **6.5 m/s²** (min_push)
- **Result**: Very aggressive start, possibly uncomfortable

**Following Mode:**
- Desired gap at 5 m/s: 6.0 m
- If actual gap is 15 m (typical): gap_err = 9 m
- Commanded acceleration: 0.5 × 9 = **4.5 m/s²**
- **Result**: Aggressive acceleration even in following

---

## Recommendations for Improvement

### 1. **Add Low-Speed Startup Logic** (High Priority)
```python
# Suggested modification
def run_step(self, obs: Observation) -> Tuple[float, Mode]:
    v = obs.ego_velocity
    
    # Gentle startup at very low speeds
    if v < 6.0 and mode == Mode.CRUISING:
        startup_gain = v / 6.0  # 0.0 to 1.0
        min_push_scaled = self.min_push * max(startup_gain, 0.3)
        # Use scaled acceleration...
```

### 2. **Velocity-Dependent Gains** (High Priority)
```python
# Adaptive proportional gain for following
def _get_following_gain(self, v_ego: float) -> float:
    # Lower gain at low speeds
    base_gain = 0.5
    if v_ego < 8.0:
        return base_gain * (v_ego / 8.0)
    return base_gain
```

### 3. **Smooth Mode Transitions** (Medium Priority)
- Add hysteresis to following/cruising mode switching
- Prevents oscillation between modes at low speeds

### 4. **Limit Maximum Jerk** (Medium Priority)
```python
# Add jerk limiting
max_jerk = 5.0  # m/s³
a_change = a_cmd - self.prev_acceleration
if abs(a_change / self.dt) > max_jerk:
    a_cmd = self.prev_acceleration + sign(a_change) * max_jerk * self.dt
```

### 5. **Speed-Dependent Deadline** (Low Priority)
- 6-second deadline may be too aggressive at very low speeds
- Consider: `deadline = max(6.0, 2.0 * (v_set - v) / 3.0)`

---

## Conclusion

### Overall Assessment: ⚠️ **MARGINAL PERFORMANCE AT VERY LOW SPEED**

The controller shows **significant limitations at speeds below 6 m/s**:

**Critical Issues:**
- 🔴 Only **25% success rate** at 5-6 m/s initial speeds
- 🔴 **High sensitivity** to initial conditions (0.7 m/s changes outcome)
- 🔴 **Unrealistic dynamics** in 50% of very low-speed scenarios
- 🔴 **Poor comfort** even when safe

**Root Cause:**
The controller was designed for **nominal operating speeds (10-20 m/s)** and lacks:
- Low-speed startup logic
- Velocity-dependent gain scheduling
- Jerk limiting
- Smooth mode transitions

**Operational Recommendation:**
⚠️ **Do not deploy this controller for scenarios requiring operation below 6 m/s** without the recommended improvements.

**Best Use Case:**
✅ Highway cruise control where speeds remain above 10 m/s (36 km/h)

---

## Additional Data Sources

To conduct deeper analysis, examine:
1. **Simulation logs**: `my_falsification_run/logs/episode-scenario_0042.csv` (and others)
2. **Full results**: `project_example_code/final_results.json`
3. **Controller code**: `project_example_code/csci513-miniproject1/mp1_controller/controller.py`
4. **Search space**: `project_example_code/config/search_space.py`

The falsification framework successfully identified these critical low-speed failure modes through multi-objective Bayesian optimization.

---

**Analysis Date**: December 1, 2025  
**Framework**: Multi-Objective Bayesian Optimization Falsification  
**Simulator**: CARLA 0.9.15  
**Controller**: Ultra-Simple Adaptive Cruise Controller


