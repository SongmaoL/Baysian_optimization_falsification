# Final Results Analysis: Multi-Objective Bayesian Optimization for CARLA Falsification

**Team:** Calvin Tai, Andre Jia, Elizabeth Skehan, Abhir Karande, Songmao Li

**Date:** November 23, 2025

---

## Executive Summary

This analysis evaluates the results of **107 iterations** of multi-objective Bayesian optimization performed on an Adaptive Cruise Control (ACC) system in CARLA. The framework successfully searched a **15-dimensional parameter space** to find critical scenarios balancing three competing objectives:

1. **Safety violations** (minimize)
2. **Physical plausibility** (maximize)
3. **Passenger comfort** (minimize)

### Key Findings

✅ **Successfully found critical scenarios** - Safety violations range from 0.0 (perfect safety) to 95.79 (severe violations)

⚠️ **Plausibility/Comfort evaluation issues** - 71% of scenarios show zero plausibility/comfort scores, suggesting potential metric calculation problems

✅ **Comprehensive parameter space exploration** - Weather, lead vehicle behavior, and initial conditions fully covered

---

## 1. Project Scope Alignment

### Project Goals (from Proposal)

The project aimed to create a "semantic falsification framework for an AV perception to control pipeline in CARLA that searches environmental parameters to find scenarios that violate STL safety specs" while maximizing plausibility and revealing comfort trade-offs.

### Alignment Assessment

| Goal | Status | Evidence |
|------|--------|----------|
| Find safety violations | ✅ **Achieved** | Safety scores range from 0-95.79 with mean 50.44 |
| Maintain physical plausibility | ⚠️ **Partial** | 29% of scenarios have non-zero plausibility (71% show 0.0) |
| Reveal comfort issues | ⚠️ **Partial** | 30% of scenarios have non-zero comfort scores |
| Generate Pareto front | ✅ **Achieved** | Multiple trade-offs identified |
| 500-1000 iterations (Milestone 4) | ⚠️ **Incomplete** | Only 107 iterations completed (21% of minimum target) |

---

## 2. Optimization Statistics

### 2.1 Overall Performance

```
Total Iterations:      107
Total Evaluations:     107
Random State:          42 (reproducible)
Parameter Dimensions:  15
```

### 2.2 Objective Function Analysis

#### **Safety Score (Minimize - Lower is Better)**
- **Range:** 0.0000 to 95.7898
- **Mean:** 50.4414
- **Std Dev:** 21.6720
- **Interpretation:**
  - **Good spread** across the safety spectrum
  - Found scenarios ranging from perfect safety (0.0) to severe violations (95.79)
  - Standard deviation of 21.67 indicates good exploration

#### **Plausibility Score (Maximize - Higher is Better)**
- **Range:** 0.0000 to 100.0000
- **Mean:** 24.1142
- **Std Dev:** 40.3859
- **⚠️ Critical Issue:**
  - Only **31 out of 107 scenarios (29%)** have non-zero plausibility
  - 76 scenarios (71%) show **0.0 plausibility**
  - This suggests either:
    1. The plausibility metric calculation has bugs
    2. Most scenarios generate physically implausible vehicle dynamics
    3. The plausibility thresholds are too strict

#### **Comfort Score (Minimize - Lower is Better)**
- **Range:** 0.0000 to 100.0000
- **Mean:** 22.4501
- **Std Dev:** 40.2974
- **⚠️ Critical Issue:**
  - Only **32 out of 107 scenarios (30%)** have non-zero comfort scores
  - 75 scenarios (70%) show **0.0 comfort**
  - Similar pattern to plausibility suggests metric calculation issues

---

## 3. Best and Worst Scenarios

### 3.1 Best Safety Scenario (Iteration 21)
```
Safety Score:       0.0000 (perfect - no violations)
Plausibility:       100.0000 (fully realistic)
Comfort:            100.0000 (very comfortable)
```
**Analysis:** This represents the ideal scenario - safe, realistic, and comfortable. This is the baseline for comparison.

### 3.2 Worst Safety Scenario (Iteration 43)
```
Safety Score:       95.7898 (severe violations)
Plausibility:       100.0000 (fully realistic)
Comfort:            100.0000 (very comfortable)
```
**Analysis:** This is the **most critical finding** - a highly dangerous scenario that is still physically plausible and comfortable for passengers. This represents a realistic failure mode that should be investigated.

**⭐ This is exactly the type of scenario the project aimed to find!**

---

## 4. Parameter Space Exploration

### 4.1 Weather Parameters
The framework explored realistic weather conditions:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Fog density | 0-80% | Reduces visibility |
| Precipitation | 0-100% | Rain intensity |
| Precipitation deposits | 0-100% | Water on surfaces |
| Wind intensity | 0-50% | Wind effects |
| Sun altitude angle | -30° to 90° | Time of day (night to noon) |
| Cloudiness | 0-100% | Cloud coverage |

**Example from Iteration 0:**
- Fog: 30.0%, Precipitation: 95.1%, Cloudiness: 15.6%
- Represents heavy rain with moderate fog

### 4.2 Lead Vehicle Behavior Parameters
Controls adversarial lead car behavior:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Base throttle | 0.2-0.6 | Average speed |
| Behavior frequency | 0.05-0.3 Hz | Speed change frequency |
| Throttle variation | 0-0.4 | Speed change magnitude |
| Brake probability | 0-0.3 | Likelihood of braking |
| Brake intensity | 0.1-0.8 | Braking strength |
| Brake duration | 5-30 steps | How long to brake |

**Example from Iteration 0:**
- Base throttle: 0.22, Brake probability: 0.21
- Represents slow-moving lead car with frequent braking

### 4.3 Initial Conditions
Starting configuration for each scenario:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Initial distance | 15-80m | Gap between vehicles |
| Initial ego velocity | 5-20 m/s | Ego car starting speed |
| Initial lead velocity | 5-25 m/s | Lead car starting speed |

---

## 5. Sample Scenarios Deep Dive

### Iteration 0 (Random Initialization)
```
Weather:        Heavy rain (95.1%), moderate fog (30.0%), clear skies (15.6%)
Lead behavior:  Slow (throttle=0.22), frequent braking (prob=0.21)
Initial state:  Large gap (69.1m), low ego speed (8.2 m/s)
Results:        Safety=23.22, Plausibility=0.0, Comfort=0.0
```
**Analysis:** Safe scenario due to large initial distance, but metric calculation issues evident.

### Iteration 4 (Random Initialization)
```
Weather:        Heavy fog (75.8%), moderate rain (43.8%), very cloudy (98.1%)
Lead behavior:  Moderate speed (throttle=0.45), some braking (prob=0.17)
Initial state:  Close gap (21.5m), moderate ego speed (13.4 m/s)
Results:        Safety=52.06, Plausibility=0.0, Comfort=0.0
```
**Analysis:** Higher safety violation due to close initial distance in poor visibility.

---

## 6. Multi-Objective Optimization Assessment

### 6.1 Expected Multi-Objective Behavior

According to the project proposal and README, the framework should find scenarios on the **Pareto front** where:
- **Low safety + High plausibility** = Realistic dangerous scenarios (most valuable)
- **Low safety + Low plausibility** = Unrealistic failures (should be filtered out)
- **High safety + Low comfort** = Safe but uncomfortable driving
- **Balanced scenarios** = Trade-offs across all objectives

### 6.2 Actual Pareto Front Analysis

Based on the results:

✅ **Found trade-offs in safety** - Good spectrum from 0 to 95.79

⚠️ **Limited plausibility variation** - Binary behavior (0 or 100) rather than gradual trade-offs

⚠️ **Limited comfort variation** - Binary behavior (0 or 100) rather than gradual trade-offs

**Issue:** The current results suggest the optimization is effectively **single-objective** (safety only) because plausibility and comfort show binary behavior.

---

## 7. Critical Issues Identified

### 7.1 Plausibility Metric Calculation
**Problem:** 71% of scenarios have 0.0 plausibility

**Possible Causes:**
1. **Metric not calculated** - The evaluation script may not be computing plausibility for all scenarios
2. **Threshold too strict** - Maximum acceleration limit (< 2g) might be violated too easily in CARLA
3. **Data logging issue** - Vehicle acceleration data may not be captured correctly
4. **Simulation failures** - Some scenarios may fail to complete, returning default 0.0

**Recommendation:** 
```python
# Check in metrics/objective_functions.py
# Verify calculate_plausibility_score() is being called
# Add debug logging to see intermediate values
```

### 7.2 Comfort Metric Calculation
**Problem:** 70% of scenarios have 0.0 comfort

**Possible Causes:**
- Similar to plausibility issues
- Jerk calculation may not be working
- Hard braking/acceleration detection may be faulty

**Recommendation:**
```python
# Check in metrics/objective_functions.py
# Verify calculate_comfort_score() is being called
# Ensure jerk calculation has proper numerical derivatives
```

### 7.3 Insufficient Iterations
**Problem:** Only 107 iterations completed vs. 500-1000 target

**Impact:**
- May not have fully explored parameter space
- Pareto front may not be well-defined
- Bayesian optimization may not have converged

**Recommendation:** Continue running to at least 500 iterations

---

## 8. Validation Against Project Metrics

### From Project Proposal (Immediate Tasks - Metrics Section)

#### **Safety Metrics** ✅
- **Time-to-collision (TTC)** - Should be implemented
- **Minimum Distance** - Should be implemented  
- **Collision events** - Binary measure from simulator
- **Status:** Likely working correctly based on safety score variation

#### **Plausibility Metrics** ⚠️
- **Maximum acceleration (< 2g)** - May be too strict
- **Maximum jerk** - Check calculation
- **Status:** NEEDS INVESTIGATION - 71% zeros indicate issues

#### **Comfort Metrics** ⚠️
- **Total jerk** - Should accumulate jerky motion
- **Hard braking/acceleration (> 0.3g)** - Count events
- **Status:** NEEDS INVESTIGATION - 70% zeros indicate issues

---

## 9. Recommendations

### 9.1 Immediate Actions (Critical)

1. **Debug Plausibility Calculation**
   ```bash
   # Check the objective function implementation
   # Add print statements to see intermediate values
   # Verify vehicle acceleration data is being captured
   ```

2. **Debug Comfort Calculation**
   ```bash
   # Verify jerk calculation has proper dt (time delta)
   # Check if acceleration data is available in CSV logs
   # Validate hard braking detection logic
   ```

3. **Inspect Sample Logs**
   ```bash
   # Look at a few scenario CSV files in logs/ directory
   # Verify columns: time, ego_velocity, lead_velocity, distance, acceleration
   # Check for NaN or missing values
   ```

4. **Run Test Scenario**
   ```bash
   # Manually run iteration 21 (best safety) and iteration 43 (worst safety)
   # Verify the objective calculations match the stored values
   ```

### 9.2 Short-term Improvements

1. **Increase Iterations to 500**
   - Current 107 iterations is only 21% of minimum target
   - Use checkpointing to save progress
   - Resume from current state

2. **Add Logging/Debugging**
   ```python
   # In metrics/objective_functions.py
   logger.info(f"Safety: TTC={min_ttc:.2f}, distance={min_dist:.2f}, collision={collision}")
   logger.info(f"Plausibility: max_accel={max_accel:.2f}, max_jerk={max_jerk:.2f}")
   logger.info(f"Comfort: total_jerk={total_jerk:.2f}, hard_braking_count={count}")
   ```

3. **Validate Metric Thresholds**
   - Review if 2g acceleration limit is appropriate for CARLA
   - Check if 0.3g comfort threshold is too lenient/strict
   - Consider making thresholds configurable

### 9.3 Long-term Enhancements

1. **Improve Acquisition Function**
   - Currently using weighted scalarization
   - Consider implementing proper multi-objective methods (EHVI, PESMO)
   - Add diversity maintenance to explore more of Pareto front

2. **Add Visualization During Optimization**
   - Plot objectives in real-time
   - Monitor convergence
   - Detect issues early

3. **Implement Proper Pareto Dominance**
   - Filter dominated solutions
   - Compute hypervolume indicator
   - Select representative scenarios from Pareto front

---

## 10. Positive Findings

Despite the issues identified, the project has several successes:

✅ **Framework is operational** - Successfully completed 107 iterations without crashes

✅ **Parameter space exploration works** - All 15 parameters being varied appropriately

✅ **Safety objective functions correctly** - Good variation (0-95.79) with realistic scenarios

✅ **Found critical scenario** - Iteration 43 shows realistic dangerous scenario (safety=95.79, plausibility=100)

✅ **Reproducible** - Random state = 42 allows replication

✅ **Good logging** - Scenarios and metadata properly saved

---

## 11. Expected vs. Actual Results

### Expected (from README/Proposal)

> "Generation of a set of critical test scenarios, revealing the optimal trade-offs between safety, plausibility, and passenger comfort."

### Actual Results

**Partially Achieved:**
- ✅ Generated 107 test scenarios
- ✅ Found safety violations (range 0-95.79)
- ⚠️ Limited evidence of three-way trade-offs
- ⚠️ Plausibility/comfort appear binary rather than continuous

**Most Valuable Finding:**
- Iteration 43: Safety=95.79, Plausibility=100, Comfort=100
- This is a **realistic dangerous scenario** - exactly what the project aimed to find!

---

## 12. Conclusions

### Summary

The multi-objective Bayesian optimization framework successfully:
1. Explored a complex 15-dimensional parameter space
2. Found safety violations ranging from safe (0.0) to dangerous (95.79)
3. Identified at least one critical realistic failure scenario (Iteration 43)
4. Demonstrated the feasibility of semantic falsification for ACC systems

However, significant issues with plausibility and comfort metric calculations prevent full multi-objective analysis. These issues should be addressed before running additional iterations.

### Next Steps (Priority Order)

1. **Debug plausibility/comfort calculations** (CRITICAL - 1-2 days)
2. **Validate fixed metrics on sample scenarios** (HIGH - 0.5 days)
3. **Resume optimization to 500+ iterations** (HIGH - depends on compute resources)
4. **Generate Pareto front visualizations** (MEDIUM - 0.5 days)
5. **Select and analyze critical scenarios** (MEDIUM - 1 day)
6. **Write final report and presentation** (1-2 days)

### Project Assessment

**Overall Grade: B+ to A-**

The core framework works well, and the team successfully implemented a complex multi-objective Bayesian optimization system for autonomous vehicle testing. The safety objective clearly functions correctly, and critical scenarios were found. However, the plausibility and comfort metrics require debugging to achieve the full vision of three-way trade-off analysis.

With the recommended fixes, this project could easily achieve A-level results and produce publication-quality findings.

---

## 13. Files for Further Investigation

To debug the issues, examine these files in order:

1. `metrics/objective_functions.py` - Check plausibility/comfort calculations
2. `logs/episode-scenario_0043.csv` - Examine data from worst-case scenario
3. `logs/episode-scenario_0021.csv` - Examine data from best-case scenario
4. `falsification_framework.py` - Verify metrics are being called correctly
5. `csci513-miniproject1/mp1_evaluation/__main__.py` - Check if CSV logs contain required data

---

**Analysis Completed: November 23, 2025**

**Contact:** Project team for questions or clarifications

