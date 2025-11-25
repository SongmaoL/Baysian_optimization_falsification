# Bayesian Optimization: Simplifications Applied

## ✅ REFACTORED: Now Uses bayes_opt Library Properly!

The code has been refactored to use the `bayes_opt` library correctly.

### What Changed

**Before (300+ lines):**
- Custom GP fitting with sklearn
- Custom acquisition function
- Custom random sampling for candidate selection
- Redundant normalization/denormalization

**After (~150 lines):**
- Uses `BayesianOptimization` class from bayes_opt
- Uses `acquisition.UpperConfidenceBound` (UCB) with kappa=2.5
- Library handles GP fitting and acquisition optimization internally
- Clean, maintainable code

### Key Code Changes

```python
# OLD (custom implementation):
from sklearn.gaussian_process import GaussianProcessRegressor
# ... 200+ lines of GP fitting, acquisition, candidate sampling

# NEW (using bayes_opt):
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

acq_func = acquisition.UpperConfidenceBound(kappa=2.5)
self.optimizer = BayesianOptimization(
    f=None,  # Manual registration
    pbounds=self.parameter_bounds,
    acquisition_function=acq_func,
    allow_duplicate_points=True
)

# Suggest next point
params = self.optimizer.suggest()

# Register result with scalarized objective
self.optimizer.register(params=parameters, target=scalar_score)
```

### Multi-Objective Strategy

We use **weighted scalarization** with Dirichlet-distributed random weights:

```python
def _scalarize_objectives(self, objectives, weights):
    score = 0.0
    for i, obj_name in enumerate(['safety', 'plausibility', 'comfort']):
        value = objectives[obj_name]
        # Negate minimization objectives (bayes_opt maximizes)
        if obj_name != 'plausibility':
            value = -value
        score += weights[i] * value
    return score
```

Each iteration samples random weights to explore different regions of the Pareto front.

---

## Future Improvements (Optional)

### 1. BoTorch with qEHVI

For state-of-the-art multi-objective BO, consider switching to BoTorch:

```python
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
```

**Benefits:**
- Proper multi-objective acquisition (qEHVI)
- Better Pareto front coverage
- GPU acceleration

**Trade-off:** More dependencies (PyTorch + BoTorch)

### 2. Adaptive Kappa (Exploration vs Exploitation)

```python
# Start exploratory, become exploitative
kappa = max(0.5, 5.0 * (1 - iteration / max_iterations))
```

### 3. Thompson Sampling

Alternative to UCB that naturally balances exploration/exploitation:

```python
# Sample from posterior instead of using UCB
sample = gp.sample_y(X_candidates, n_samples=1)
```

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Lines of code | ~300 | ~150 |
| GP fitting | Manual sklearn | Built-in |
| Acquisition optimization | Random sampling | L-BFGS-B (library) |
| Maintainability | Complex | Simple |

The refactored code is now:
- ✅ Using bayes_opt properly
- ✅ Clean and maintainable
- ✅ Fully compatible with existing framework
