# Next Steps

## ✓ FIXED: Root Cause Identified!

**Problem:** `scenario_generator.py` used instant step functions for brake commands
- Brake went 0→0.8 instantly → unrealistic >2g acceleration
- **71% plausibility=0 was CORRECT** - those scenarios violated physics!

**Fix:** Added exponential smoothing (alpha=0.3) to brake/throttle transitions
- Now takes ~0.3-0.5s to transition (realistic)
- Should generate physically plausible scenarios

---

## 1. (Optional) Diagnose Old Scenarios

If you want to confirm the issue with existing 107 scenarios:

```bash
cd project_example_code
python diagnose_physics.py
```

This will show that old scenarios had >2.5g acceleration (unrealistic).

## 2. Re-run Falsification with Fixed Scenarios (3-5 days) - CRITICAL

**Start fresh** with the fixed scenario generator:

```bash
# Option A: Start completely new run (recommended)
python falsification_framework.py \
    --carla-project csci513-miniproject1 \
    --output-dir falsification_fixed \
    --n-iterations 500 \
    --init-points 20 \
    --checkpoint-interval 25 \
    --random-state 43

# Option B: Continue from iteration 107 (if you want to keep history)
python falsification_framework.py \
    --carla-project csci513-miniproject1 \
    --output-dir my_falsification_run \
    --n-iterations 500 \
    --checkpoint-interval 25 \
    --resume-from my_falsification_run/results/checkpoint_0100.json
```

**Expected results:**
- Plausibility: Should see 60-80% non-zero (instead of 29%)
- Comfort: Should see 60-80% non-zero (instead of 30%)
- Realistic multi-objective trade-offs

---

## 3. Monitor Progress

```bash
# Resume to 500 iterations
python falsification_framework.py \
    --carla-project csci513-miniproject1 \
    --output-dir my_falsification_run \
    --n-iterations 500 \
    --checkpoint-interval 25 \
    --resume-from my_falsification_run/results/checkpoint_0100.json
```

## 4. Final Analysis (1 day)

```bash
# Generate Pareto front
python analysis/pareto_analysis.py \
    my_falsification_run/results/final_results.json \
    --output-dir final_analysis

# Re-run visualizations
python visualize_results.py

# Select critical scenarios for testing/presentation
```

## 5. Documentation (1-2 days)

- Final report
- Presentation slides (include videos of critical scenarios)
- Findings about ACC controller failure modes

---

**Timeline:** ~7-10 days total

**Status:**
- ✓ Root cause found: Instant brake commands in scenario_generator.py
- ✓ Fix applied: Exponential smoothing (alpha=0.3)
- ⏳ Next: Re-run falsification with fixed scenarios → 500 iterations
- ⏳ Then: Final analysis with realistic Pareto front

