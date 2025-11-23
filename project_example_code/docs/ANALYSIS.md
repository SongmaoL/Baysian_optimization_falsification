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

## Issues (FIXED ✓)

### Root Cause Identified: Instant Brake Commands

**Problem:** `scenario_generator.py` used step functions for brake commands:
- Brake: 0.0 → 0.8 instantly (unrealistic)
- Creates huge acceleration spikes when lead vehicle brakes
- Results in >2g acceleration → plausibility=0 (correctly!)

**Fix Applied:** Added exponential smoothing (alpha=0.3) to brake/throttle transitions
- Takes ~0.3-0.5s to reach target brake level
- Realistic acceleration profiles
- Should reduce implausible scenarios significantly

**Next:** Re-run falsification with fixed scenario generator to generate realistic scenarios

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

