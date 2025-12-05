# Critical Issues Audit - Bayesian Optimization Falsification Framework

This document lists all identified issues where parameters are generated/configured but not actually used in the simulation.

---

## 🔴 CRITICAL ISSUES (Major Impact on Results)

### 1. ❌ Weather Parameters NOT Applied
**Status**: FIXED (with `simulator_sensor.py`)

**Location**: `mp1_simulator/simulator.py` line 94

**Problem**: Weather is hardcoded, ignoring scenario weather:
```python
self.world.set_weather(carla.WeatherParameters.ClearNoon)  # Always clear!
```

**Impact**: Weather optimization is useless - fog, rain, etc. have zero effect

**Solution**: Use `simulator_sensor.py` which applies weather from scenarios

---

### 2. ❌ Initial Velocities NOT Applied
**Status**: NOT FIXED ⚠️

**Location**: `mp1_simulator/simulator.py` `set_spawn_points()` method

**Problem**: Scenarios include `initial_ego_velocity` and `initial_lead_velocity`, but these are **never applied** to the vehicles:

```python
# scenario_generator.py generates:
'velocity': initial_ego_velocity,  # e.g., 15.0 m/s

# But simulator.py only uses position:
ego_location = initial_ego_state["position"]
ego_yaw = initial_ego_state["yaw"]
# velocity is IGNORED - vehicles start at 0 m/s!
```

**Impact**: 
- All simulations start with both vehicles at 0 velocity
- The "initial_ego_velocity" parameter in search space does NOTHING
- Controller behavior at "low speed" is actually behavior during acceleration from rest
- Distance parameter becomes less meaningful (vehicles don't move until they accelerate)

**Data Evidence**: In my low-speed analysis, scenarios showed ego starting at 5.21 m/s, but this was never actually applied!

---

### 3. ❌ Ground Truth Instead of Sensors  
**Status**: FIXED (with `simulator_sensor.py`)

**Location**: `mp1_simulator/simulator.py` `_get_obs()` method

**Problem**: Distance and velocity come from exact ground truth, not sensors

**Impact**: Weather cannot affect perception even if applied

---

## 🟡 MODERATE ISSUES (Partial Impact)

### 4. ⚠️ Desired Speed is Hardcoded
**Status**: NOT FIXED

**Location**: `mp1_simulator/simulator.py` line 39

**Problem**: Desired cruise speed is hardcoded to 20 m/s:
```python
CONFIG = {
    ...
    "desired_speed": 20,  # HARDCODED
    ...
}
```

**Impact**:
- Cannot test controller at different target speeds
- Missing from search space (could be valuable parameter)
- All scenarios use same target speed

**Recommended Fix**: Add `desired_speed` to search space or scenario files

---

### 5. ⚠️ Distance Threshold is Hardcoded
**Status**: NOT FIXED

**Location**: `mp1_simulator/simulator.py` line 40

**Problem**: Distance threshold is hardcoded to 4 meters:
```python
"distance_threshold": 4,  # HARDCODED
```

**Impact**: Cannot test controller with different safety margins

---

### 6. ⚠️ Lead Vehicle Initial Actions Ignore Weather
**Status**: NOT FIXED

**Location**: `scenario_generator.py` `generate_lead_vehicle_actions()`

**Problem**: Lead vehicle behavior doesn't react to weather conditions:
- In rain, a real driver would brake more cautiously
- In fog, a real driver would slow down
- The generated behavior is weather-independent

**Impact**: Unrealistic lead vehicle behavior in adverse conditions

---

## 🟢 MINOR ISSUES (Low Impact)

### 7. ℹ️ Random Vehicle Selection
**Status**: Known Limitation

**Location**: `simulator.py` `_create_vehicle_blueprint()`

**Problem**: Vehicle model is randomly selected from available blueprints
```python
bp = random.choice(blueprint_library)
```

**Impact**: 
- Slight variation in vehicle dynamics between runs
- Not reproducible without fixing the random seed

---

### 8. ℹ️ Town06 Only
**Status**: Known Limitation

**Location**: `simulator.py` line 36

**Problem**: Only Town06 straight road is used:
```python
"town": "Town06",
```

**Impact**: Limited scenario diversity (straight road only, no curves/intersections)

---

### 9. ℹ️ No Latency/Delay Modeling
**Status**: Not Implemented

**Problem**: Real ACC systems have:
- Sensor processing delay
- Control loop delay
- Actuator delay

**Impact**: Simulation is more optimistic than real-world performance

---

## 📊 Summary Table

| Issue | Parameter | Generated | Applied | Impact |
|-------|-----------|-----------|---------|--------|
| Weather | fog_density, precipitation, etc. | ✅ | ❌→✅ | **HIGH** |
| Initial Ego Velocity | initial_ego_velocity | ✅ | ❌ | **HIGH** |
| Initial Lead Velocity | initial_lead_velocity | ✅ | ❌ | **HIGH** |
| Sensor Realism | - | N/A | ❌→✅ | **HIGH** |
| Desired Speed | desired_speed | ❌ | ❌ | Medium |
| Distance Threshold | distance_threshold | ❌ | ❌ | Medium |
| Weather-aware Lead | - | ❌ | ❌ | Low |

---

## 🔧 Recommended Fixes

### Priority 1: Apply Initial Velocities

Add to `simulator.py` (or `simulator_sensor.py`):

```python
def set_spawn_points(self, initial_ego_state, initial_lead_state):
    # ... existing code ...
    
    # Store initial velocities for reset
    self.initial_ego_velocity = initial_ego_state.get('velocity', 0.0)
    self.initial_lead_velocity = initial_lead_state.get('velocity', 0.0)

def reset(self) -> Observation:
    # ... existing spawn code ...
    
    # Apply initial velocities
    if hasattr(self, 'initial_ego_velocity') and self.initial_ego_velocity > 0:
        # Get vehicle forward direction
        transform = self.ego.get_transform()
        forward = transform.get_forward_vector()
        
        # Set velocity (convert speed to velocity vector)
        velocity = carla.Vector3D(
            x=forward.x * self.initial_ego_velocity,
            y=forward.y * self.initial_ego_velocity,
            z=0.0
        )
        self.ego.set_target_velocity(velocity)
    
    if hasattr(self, 'initial_lead_velocity') and self.initial_lead_velocity > 0:
        transform = self.ado.get_transform()
        forward = transform.get_forward_vector()
        velocity = carla.Vector3D(
            x=forward.x * self.initial_lead_velocity,
            y=forward.y * self.initial_lead_velocity,
            z=0.0
        )
        self.ado.set_target_velocity(velocity)
    
    # Let physics settle
    self.world.tick()
    
    return self._get_obs()
```

### Priority 2: Add Desired Speed to Scenario

In `scenario_generator.py`:
```python
scenario = {
    ...
    'config': {
        'desired_speed': params.get('desired_speed', 20.0),
        'distance_threshold': params.get('distance_threshold', 4.0),
    }
}
```

In `__main__.py`:
```python
if 'config' in scenario_data:
    CONFIG.update(scenario_data['config'])
```

### Priority 3: Weather-Aware Lead Vehicle

In `scenario_generator.py`:
```python
def generate_lead_vehicle_actions(params, ...):
    # Reduce base throttle in bad weather
    fog = params.get('fog_density', 0)
    rain = params.get('precipitation', 0)
    
    weather_factor = 1.0 - 0.3 * (fog + rain) / 200.0
    base_throttle *= weather_factor
    
    # Increase brake probability in rain
    if rain > 20:
        brake_prob *= 1.5
```

---

## 📈 Impact on Existing Results

The `final_results.json` contains results that are **misleading** because:

1. **Weather scores are meaningless**: All runs used ClearNoon regardless of generated weather
2. **Initial velocities were ignored**: Vehicles always started at rest
3. **"Low speed" scenarios were mischaracterized**: What was analyzed as 5 m/s starting speed was actually 0 m/s

**Recommendation**: Re-run the falsification with fixes applied to get accurate results.

---

## ✅ What's Working Correctly

- Lead vehicle throttle/brake/steer actions ARE applied
- Initial positions ARE applied
- Collision detection works
- Objective function calculations are correct (given the data they receive)
- Bayesian optimization logic works correctly
- Pareto front tracking is correct

---

---

## 🔵 ADDITIONAL ISSUES FOUND

### 10. ⚠️ STL Evaluation Constants Mismatch
**Status**: NOT FIXED

**Location**: `mp1_evaluation/__main__.py`

**Problem**: The STL evaluation uses hardcoded constants that may not match controller/scenario:
```python
spec.declare_const("dsafe", "float", "4")  # Matches CONFIG, but not parameterized
spec.declare_const("followingDistThreshold", "float", "40.")  # MUCH larger than controller's 16m gate
spec.declare_const("closeEnough", "float", "2.0")  # Matches controller tolerance
```

**Impact**: 
- `followingDistThreshold=40` doesn't match controller's `d < 16.0` absolute near gate
- Evaluation may not catch actual mode switching issues

---

### 11. ⚠️ Suspicious STL Spec - `checkDontStopUnlessLeadStops`
**Status**: POTENTIAL BUG

**Location**: `mp1_evaluation/__main__.py` line 125

**Problem**: Strange time bounds in spec:
```python
spec.spec = "G[3.:3.] (not((lead_speed > reallySmallSpeed) until[0.:20.] ...))"
```

`G[3.:3.]` means "globally at exactly time 3.0 seconds" - this evaluates only at a single time point, not a range!

**Likely Intent**: Should be `G[3.:20.]` or `G[3.:]` to check from 3s onwards

---

### 12. ⚠️ Collision Detection Doesn't Use Actual Collision Sensor
**Status**: NOT FIXED

**Location**: `metrics/objective_functions.py` line 157-158

**Problem**: Collision is inferred from distance, not from actual collision sensor:
```python
min_distance = trace_df['distance_to_lead'].min()
return min_distance < 0.5  # 0.5m threshold for collision
```

But the simulator has an actual collision sensor that logs `collided_event`.

**Impact**: 
- Traces don't include actual collision flag
- Close approaches (0.5m) counted as collision even if no physical contact

---

### 13. ⚠️ Plausibility Score Doesn't Use Lead Actions
**Status**: PARTIAL

**Location**: `metrics/objective_functions.py` line 214-215

**Problem**: The function accepts `lead_actions` parameter but never uses it:
```python
def calculate_plausibility_score(trace_df: pd.DataFrame, 
                                 lead_actions: Optional[Dict] = None,  # NEVER USED!
                                 dt: float = 0.1) -> float:
```

The docstring says "Unrealistic lead vehicle behavior = low score" but this isn't implemented.

**Impact**: Lead vehicle plausibility is never checked

---

### 14. ℹ️ Trace Logs Missing Useful Data
**Status**: Known Limitation

**Location**: `mp1_simulator/__main__.py` line 97-106

**Problem**: Trace CSV doesn't include:
- Acceleration command from controller
- Throttle/brake applied
- Mode transitions
- Actual collision events
- Ground truth vs sensor (for sensor simulator)

**Impact**: Post-hoc analysis is limited

---

### 15. ⚠️ No Random Seed Control for Vehicle Blueprints
**Status**: NOT FIXED

**Location**: `simulator.py` line 172

**Problem**: Vehicle selection uses `random.choice` without seed:
```python
bp = random.choice(blueprint_library)
```

But `random.seed()` is never called, so it uses system time.

**Impact**: Vehicle dynamics vary non-reproducibly between runs

---

### 16. ℹ️ Original Controller is Just a Placeholder
**Status**: Known Limitation

**Location**: `mp1_controller/controller_original.py`

**Problem**: The "original" controller just returns maximum brake:
```python
return (-10.0, Mode.FOLLOWING)  # Always full brake, always following
```

This is a template, not a real baseline controller.

---

### 17. ⚠️ No Timeout Handling in Simulation
**Status**: NOT FIXED

**Location**: `falsification_framework.py` line 94

**Problem**: Simulation has 2-minute timeout but no graceful handling:
```python
timeout=120  # 2 minute timeout
```

If timeout occurs, no partial results are saved.

**Impact**: Long scenarios lost entirely on timeout

---

### 18. ⚠️ Pareto Dominance Only Uses Three Objectives
**Status**: Design Limitation

**Location**: `multi_objective_bo.py` line 74

**Problem**: Pareto dominance hardcodes three objectives:
```python
v1 = np.array([s1['safety'], -s1['plausibility'], s1['comfort']])
```

Can't easily add new objectives (e.g., efficiency, smoothness).

---

## 📊 Updated Summary Table

| Issue # | Category | Severity | Fixed? |
|---------|----------|----------|--------|
| 1 | Weather | 🔴 Critical | ✅ Yes |
| 2 | Initial Velocity | 🔴 Critical | ✅ Yes |
| 3 | Ground Truth Sensors | 🔴 Critical | ✅ Yes |
| 4 | Desired Speed | 🟡 Medium | ✅ Yes |
| 5 | Distance Threshold | 🟡 Medium | ✅ Yes |
| 6 | Weather-aware Lead | 🟡 Medium | ✅ Yes |
| 7 | Random Vehicle | 🟢 Low | ✅ Yes |
| 8 | Town06 Only | 🟢 Low | ❌ No (Design) |
| 9 | Latency Modeling | 🟢 Low | ❌ No (Design) |
| 10 | STL Constants | 🟡 Medium | ✅ Yes |
| 11 | STL Time Bounds | 🟡 Medium | ✅ Yes |
| 12 | Collision Detection | 🟡 Medium | ✅ Yes |
| 13 | Lead Plausibility | 🟡 Medium | ✅ Yes |
| 14 | Trace Logging | 🟢 Low | ✅ Yes |
| 15 | Random Seed | 🟢 Low | ✅ Yes |
| 16 | Original Controller | 🟢 Low | N/A (Template) |
| 17 | Timeout Handling | 🟢 Low | ✅ Yes |
| 18 | Pareto Extensibility | 🟢 Low | ❌ No (Design) |

**Legend**: 🔴 Critical (affects results significantly) | 🟡 Medium (affects some scenarios) | 🟢 Low (minor impact)

---

## ✅ FIXES APPLIED (December 1, 2025)

### Fix Summary

| Issue | Fix Location | Description |
|-------|--------------|-------------|
| #4, #5 | `config/search_space.py`, `scenario_generator.py`, `__main__.py` | Added `desired_speed` and `distance_threshold` to search space and scenarios |
| #6 | `scenario_generator.py` | Lead vehicle now slows down in fog/rain (up to 40% throttle reduction) |
| #7, #15 | `simulator.py` | Added `random_seed` config option, seeds `random` and `numpy` |
| #10 | `mp1_evaluation/__main__.py` | Made STL constants configurable, fixed `followingDistThreshold` to 16m |
| #11 | `mp1_evaluation/__main__.py` | Fixed `G[3.:3.]` → `G[3.:20.]` for proper time range |
| #12 | `metrics/objective_functions.py` | `check_collision()` now uses actual collision flag if available |
| #13 | `metrics/objective_functions.py` | Added `calculate_lead_plausibility()`, included in plausibility score |
| #14 | `mp1_simulator/__main__.py` | Trace now includes `acceleration` and `collision` columns |
| #17 | `falsification_framework.py` | Timeout now tries to save partial results |

### New Search Space Parameters

The search space now includes **17 parameters** (was 15):

```python
CONTROLLER_CONFIG_BOUNDS = {
    "desired_speed": (15.0, 30.0),      # Target cruise speed (m/s)
    "distance_threshold": (3.0, 8.0),   # Safe following distance (m)
}
```

### New Trace Columns

Trace CSV files now include:

| Column | Type | Description |
|--------|------|-------------|
| `acceleration` | float | Controller's acceleration command (m/s²) |
| `collision` | int | 1 if collision occurred, 0 otherwise |

---

**Audit Date**: December 1, 2025
**Auditor**: Code Review

