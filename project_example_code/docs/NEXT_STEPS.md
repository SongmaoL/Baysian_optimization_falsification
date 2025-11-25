# Next Steps

## ✓ FIXED: Two Issues Identified!

### Issue 1: Instant Brake Commands (Fixed in run1→run2)
- **Problem:** Brake went 0→0.8 instantly → >2g acceleration
- **Fix:** Added exponential smoothing (alpha=0.3)
- **Result:** Plausibility improved 29% → 82% ✅

### Issue 2: High Jerk (Found in run2 analysis)
- **Problem:** alpha=0.3 still causes jerk ~90 m/s³ (threshold: 10 m/s³)
- **Fix:** Reduced alpha to 0.1 for smoother transitions (~1 second)
- **Expected:** Jerk drops to ~10-20 m/s³, comfort becomes non-zero

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
- ✓ Issue 1 fixed: Exponential smoothing added (run2 shows 82% plausibility)
- ✓ Issue 2 fixed: Reduced alpha 0.3→0.1 to fix jerk (comfort should improve)
- ⏳ Next: Run3 with both fixes → expect 70%+ for all three metrics
- ⏳ Then: Continue to 500 iterations → final Pareto analysis

