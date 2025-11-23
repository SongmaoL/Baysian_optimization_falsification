# Results Analysis: 107 Iterations

## Statistics

```
Iterations: 107 (target: 500-1000)
Parameters: 15D (weather × lead behavior × initial conditions)
Random State: 42

Objectives:
  Safety (min):       0.00 - 95.79  (μ=50.44, σ=21.67) ✓ Working
  Plausibility (max): 0.00 - 100.00 (μ=24.11, σ=40.39) ⚠️ 71% zeros
  Comfort (min):      0.00 - 100.00 (μ=22.45, σ=40.30) ⚠️ 70% zeros
```

## Key Findings

### Critical Scenario: Iteration 43
```
Safety:       95.79 (most dangerous found)
Plausibility: 100.00 (fully realistic)
Comfort:      100.00
Parameters:   fog=38.3, brake_prob=0.28, initial_dist=18.5
```
This is exactly what the project aimed for - realistic dangerous scenario.

### BO Performance
- Efficiently explored 15D space
- Found critical scenarios (safety 0→95.79 range)
- GP models converging properly (based on safety metric)

## Issues

### 1. Plausibility Metric (Critical)
- **Problem:** 76/107 scenarios (71%) return 0.0
- **Expected:** Gradual distribution 0-100
- **Impact:** Can't properly analyze multi-objective trade-offs
- **Likely causes:** 
  - Metric not calculated (check `calculate_plausibility_score()`)
  - 2g threshold too strict
  - Missing acceleration data in CSV logs

### 2. Comfort Metric (Critical)
- **Problem:** 75/107 scenarios (70%) return 0.0
- **Similar pattern to plausibility**
- **Likely causes:**
  - Jerk calculation issues
  - Missing dt parameter
  - Threshold problems

### 3. Insufficient Iterations
- Only 107/500 minimum completed (21%)
- Need more data for robust Pareto front

## BO Implementation

**Where it happens:**
```python
# falsification_framework.py, line 195
params = self.optimizer.suggest_next()  # BO suggests parameters
  ↓ Uses 3 GP models (safety, plausibility, comfort)
  ↓ UCB acquisition: mean ± kappa*std
  ↓ Random weights per iteration to explore Pareto front

# line 245
self.optimizer.register_evaluation(params, objectives)  # BO learns
```

**Key components:**
- 3 independent GPs (Matern kernel, nu=2.5)
- UCB acquisition function (exploration_param=2.0)
- Weighted scalarization with random Dirichlet weights
- Random search over 1000 candidates per iteration

## Visualizations Generated

- `optimization_results_visualization.png` - 2×2 dashboard
- `optimization_results_3d.png` - 3D Pareto front
- `parameter_exploration.png` - Parameter coverage

---

**Assessment:** Core framework works. Safety metric validated. Plausibility/comfort metrics need debugging before continuing to 500 iterations.

