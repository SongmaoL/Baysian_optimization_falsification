# Next Steps

## 1. Debug Metrics (1-2 days) - DO FIRST

### Check what's wrong:
```bash
# Examine log file structure
head logs/episode-scenario_0043.csv

# Check if acceleration/jerk columns exist
# If missing, need to calculate from velocity
```

### Fix plausibility calculation:
```python
# metrics/objective_functions.py

def calculate_plausibility_score(trace_df, dt=0.1):
    # Add: Calculate accel from velocity if missing
    if 'ego_acceleration' not in trace_df.columns:
        trace_df['ego_acceleration'] = trace_df['ego_velocity'].diff() / dt
    
    max_accel = trace_df['ego_acceleration'].abs().max()
    
    # Check threshold - 2g (20 m/s²) may be too strict
    MAX_ACCEL = 20.0  
    MAX_JERK = 10.0
    
    # Calculate jerk
    jerk = trace_df['ego_acceleration'].diff() / dt
    max_jerk = jerk.abs().max()
    
    # Score (tune these weights)
    accel_score = 100 if max_accel < MAX_ACCEL else 0
    jerk_score = 100 if max_jerk < MAX_JERK else 0
    
    return (accel_score + jerk_score) / 2
```

### Fix comfort calculation:
```python
def calculate_comfort_score(trace_df, dt=0.1):
    # Calculate jerk
    if 'ego_jerk' not in trace_df.columns:
        accel = trace_df['ego_acceleration']
        trace_df['ego_jerk'] = accel.diff() / dt
    
    total_jerk = trace_df['ego_jerk'].abs().sum()
    
    # Hard braking (> 0.3g)
    THRESHOLD = 3.0  # m/s²
    hard_events = (trace_df['ego_acceleration'].abs() > THRESHOLD).sum()
    
    # Normalize (tune these)
    jerk_score = max(0, 100 - total_jerk / 10)
    event_score = max(0, 100 - hard_events * 5)
    
    return (jerk_score + event_score) / 2
```

### Test fixes:
```python
# test_metrics.py
from metrics.objective_functions import evaluate_trace_file

for scenario in ['0021', '0043']:
    log = f'logs/episode-scenario_{scenario}.csv'
    obj = evaluate_trace_file(log, dt=0.1)
    print(f"{scenario}: safety={obj['safety']:.1f}, plaus={obj['plausibility']:.1f}, comfort={obj['comfort']:.1f}")

# Should see non-zero plausibility & comfort
```

## 2. Continue Optimization (3-5 days)

```bash
# Resume to 500 iterations
python falsification_framework.py \
    --carla-project csci513-miniproject1 \
    --output-dir my_falsification_run \
    --n-iterations 500 \
    --checkpoint-interval 25 \
    --resume-from my_falsification_run/results/checkpoint_0100.json
```

## 3. Final Analysis (1 day)

```bash
# Generate Pareto front
python analysis/pareto_analysis.py \
    my_falsification_run/results/final_results.json \
    --output-dir final_analysis

# Re-run visualizations
python visualize_results.py

# Select critical scenarios for testing/presentation
```

## 4. Documentation (1-2 days)

- Final report
- Presentation slides (include videos of critical scenarios)
- Findings about ACC controller failure modes

---

**Timeline:** ~7-10 days total

**Priority:** Fix metrics today → continue to 500 iterations → analyze final Pareto front

