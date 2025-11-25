# Bayesian Optimization: Simplifications & Improvements

## Current Issues

### 1. **Not Using the Imported Library!**
```python
# Line 17-22 imports bayes_opt but NEVER uses it!
from bayes_opt import BayesianOptimization  # ← Imported
from bayes_opt import UtilityFunction       # ← Imported
# ... then builds everything from scratch with sklearn
```

### 2. **Naive Acquisition Optimization**
```python
# Line 297-307: Random sampling instead of gradient optimization
for _ in range(n_random_samples):  # 1000 random samples
    candidate = np.random.random(self.n_params)
    acq_value = self._acquisition_function(candidate, weights)
```
This is **very inefficient** - L-BFGS-B would find better points faster.

### 3. **GP Models Refitted Every Iteration**
```python
# Line 291-292: Full refit on every suggest_next() call
self._fit_gp_models()  # O(n³) complexity!
```
With 200 iterations, this becomes slow.

### 4. **No Expected Hypervolume Improvement (EHVI)**
Using random scalarization weights is a basic approach. EHVI is the standard for multi-objective BO.

### 5. **Unused Parameters**
```python
init_points = 10  # Passed but never used to force exploration!
```

---

## Option 1: Use BoTorch (Recommended)

BoTorch is a PyTorch-based BO library with state-of-the-art multi-objective support.

```python
# multi_objective_bo_simple.py

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume

class SimpleMOBO:
    def __init__(self, bounds, ref_point):
        """
        Args:
            bounds: torch.Tensor of shape (2, d) with lower/upper bounds
            ref_point: Reference point for hypervolume (worst-case values)
        """
        self.bounds = bounds
        self.ref_point = ref_point
        self.X = []
        self.Y = []
    
    def suggest_next(self, n_candidates=1):
        if len(self.X) < 5:
            # Random exploration
            return torch.rand(n_candidates, self.bounds.shape[1]) * (
                self.bounds[1] - self.bounds[0]
            ) + self.bounds[0]
        
        # Fit GP models (one per objective)
        X = torch.stack(self.X)
        Y = torch.stack(self.Y)
        
        models = []
        for i in range(Y.shape[1]):
            gp = SingleTaskGP(X, Y[:, i:i+1])
            models.append(gp)
        model = ModelListGP(*models)
        
        # qEHVI acquisition function
        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            partitioning=...,  # Use default
        )
        
        # Optimize acquisition function with L-BFGS-B
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=n_candidates,
            num_restarts=10,
            raw_samples=512,
        )
        
        return candidates
    
    def register(self, x, y):
        self.X.append(x)
        self.Y.append(y)
```

**Benefits:**
- ✅ State-of-the-art qEHVI acquisition
- ✅ Proper gradient-based optimization
- ✅ GPU acceleration available
- ✅ Much less code

---

## Option 2: Simplify Current Code (Use bayes_opt properly)

If you want to keep dependencies minimal:

```python
# Simplified single-objective version using the imported library
from bayes_opt import BayesianOptimization, UtilityFunction

class SimpleBOWrapper:
    def __init__(self, pbounds, objective_func):
        self.optimizer = BayesianOptimization(
            f=objective_func,
            pbounds=pbounds,
            random_state=42,
            verbose=0
        )
        self.utility = UtilityFunction(kind="ucb", kappa=2.5)
    
    def suggest_next(self):
        return self.optimizer.suggest(self.utility)
    
    def register(self, params, target):
        self.optimizer.register(params=params, target=target)

# For multi-objective: Run 3 separate optimizers with different weights
def multi_objective_scalarize(safety, plausibility, comfort, weights):
    # Negate plausibility since we maximize it
    return weights[0] * (-safety) + weights[1] * plausibility + weights[2] * (-comfort)
```

---

## Option 3: Quick Fixes to Current Code

### Fix 1: Use L-BFGS-B for Acquisition Optimization

```python
from scipy.optimize import minimize

def suggest_next(self, n_random_samples: int = 10):
    if len(self.evaluation_history) < 3:
        return self._denormalize_parameters(np.random.random(self.n_params))
    
    self._fit_gp_models()
    weights = self._generate_random_weights()
    
    # Use L-BFGS-B instead of random sampling
    best_x = None
    best_acq = -np.inf
    
    # Multi-start optimization
    for _ in range(n_random_samples):
        x0 = np.random.random(self.n_params)
        
        result = minimize(
            lambda x: -self._acquisition_function(x, weights),  # Negate for minimization
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * self.n_params
        )
        
        if -result.fun > best_acq:
            best_acq = -result.fun
            best_x = result.x
    
    return self._denormalize_parameters(best_x)
```

### Fix 2: Warm Start GP Models

```python
def _fit_gp_models(self):
    """Fit GP models with warm starting."""
    if len(self.evaluation_history) == 0:
        return
    
    X = np.array([self._normalize_parameters(r.parameters) 
                 for r in self.evaluation_history])
    
    for obj_name in self.objective_names:
        y = np.array([r.objectives[obj_name] for r in self.evaluation_history])
        
        # Reuse previous kernel parameters if available
        if self.gp_models[obj_name] is not None:
            kernel = self.gp_models[obj_name].kernel_
        else:
            kernel = C(1.0) * Matern(length_scale=0.1, nu=2.5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # Reduced from 10
            alpha=1e-6,
            normalize_y=True,
            random_state=self.random_state
        )
        gp.fit(X, y)
        self.gp_models[obj_name] = gp
```

### Fix 3: Use init_points Properly

```python
def suggest_next(self):
    # Use init_points for initial random exploration
    if len(self.evaluation_history) < self.init_points:
        return self._denormalize_parameters(np.random.random(self.n_params))
    
    # Then use BO
    ...
```

### Fix 4: Batch GP Predictions

```python
def _acquisition_function_batch(self, X_candidates, weights):
    """Evaluate acquisition on batch of candidates at once."""
    acq_values = np.zeros(len(X_candidates))
    
    for i, obj_name in enumerate(self.objective_names):
        gp = self.gp_models[obj_name]
        if gp is None:
            continue
        
        # Batch prediction is much faster
        means, stds = gp.predict(X_candidates, return_std=True)
        
        if self.maximize_objectives[i]:
            ucb = means + self.exploration_param * stds
        else:
            ucb = means - self.exploration_param * stds
        
        acq_values += weights[i] * ucb
    
    return acq_values
```

---

## Comparison

| Approach | Complexity | Performance | Multi-Obj Quality |
|----------|------------|-------------|-------------------|
| Current (random scalarization) | Simple | Slow | Basic |
| bayes_opt library | Simple | Fast | Basic |
| BoTorch qEHVI | Medium | Fast | Excellent |
| Custom L-BFGS-B | Medium | Medium | Good |

---

## Recommended Path Forward

### If keeping current structure:
1. Apply Fix 1 (L-BFGS-B) - Improves convergence
2. Apply Fix 3 (init_points) - Proper exploration phase
3. Apply Fix 4 (batch predictions) - 5-10x speedup

### If willing to refactor:
1. Switch to BoTorch with qEHVI
2. Much better Pareto front coverage
3. ~50% less code

### Minimal change:
Just use the `bayes_opt` library that's already imported:
```python
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(f=None, pbounds=bounds)
optimizer.suggest(utility)  # Returns best next point
optimizer.register(params, target)  # Update model
```

---

## Other Simplifications

### 1. Remove Redundant Parameter Normalization
The bayes_opt and BoTorch libraries handle normalization internally.

### 2. Use Dataclasses More
```python
@dataclass
class Scenario:
    weather: dict
    lead_actions: list
    initial_conditions: dict
    
    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
```

### 3. Simplify Metric Calculation
```python
# Instead of separate functions, use a single pipeline
def evaluate_scenario(trace_df, dt=0.1):
    accel = np.diff(trace_df['ego_velocity']) / dt
    jerk = np.diff(accel) / dt
    
    return {
        'safety': calculate_safety(trace_df),
        'plausibility': score_accel_jerk(accel.max(), jerk.max()),
        'comfort': score_comfort(jerk, accel),
    }
```

---

## Summary

**Quick wins (30 min each):**
1. Use L-BFGS-B for acquisition optimization
2. Use init_points properly
3. Batch GP predictions

**Medium effort (1-2 hours):**
1. Use the bayes_opt library that's already imported
2. Simplify metric calculations

**Best results (4-6 hours):**
1. Switch to BoTorch with qEHVI
2. Get proper multi-objective Pareto front coverage

