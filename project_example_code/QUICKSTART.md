# Quick Start Guide

This guide will get you running the falsification framework in under 10 minutes.

## Prerequisites Check

Before starting, ensure you have:

- [ ] Python 3.8+ installed
- [ ] CARLA 0.9.15 installed (or access to Docker)
- [ ] GPU available (or HPC access)
- [ ] Terminal access (2 terminals needed)

## Step 1: Install Dependencies (5 minutes)

```bash
# Navigate to project directory
cd project_example_code

# Install Python packages
pip install -r requirements.txt

# Install CARLA mini-project
cd csci513-miniproject1
pip install -e .
cd ..
```

## Step 2: Verify Installation (2 minutes)

```bash
# Test imports
python -c "import numpy, pandas, sklearn, matplotlib, bayes_opt; print('All packages installed!')"

# Test search space
python config/search_space.py

# Test scenario generator
python scenario_generator.py

# Test multi-objective BO
python multi_objective_bo.py
```

Expected output: Summary messages and "All packages installed!"

## Step 3: Start CARLA Server (1 minute)

**Terminal 1:**

```bash
# Option A: Direct installation
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen

# Option B: Docker
cd csci513-miniproject1
make run-carla
```

Wait for: "Listening on port 2000" or similar message

## Step 4: Run Mini Experiment (2 minutes)

**Terminal 2:**

```bash
# Run 10 iterations as a test
python falsification_framework.py \
    --n-iterations 10 \
    --init-points 5 \
    --output-dir test_run

# Expected time: ~5 minutes
```

Watch for progress messages. Each iteration should complete in ~30 seconds.

## Step 5: Analyze Results (1 minute)

```bash
# Generate visualizations
python analysis/pareto_analysis.py \
    test_run/results/final_results.json \
    --output-dir test_run/analysis

# View plots
ls test_run/analysis/*.png
```

Open the PNG files to see:
- `pareto_front_3d.png` - 3D Pareto front
- `convergence.png` - Objective convergence
- `pareto_front_2d.png` - 2D projections

## Step 6: Run Full Experiment

Once the test run works, scale up:

```bash
# Full experiment: 500 iterations (~4-6 hours)
python falsification_framework.py \
    --n-iterations 500 \
    --init-points 50 \
    --checkpoint-interval 25 \
    --output-dir falsification_500

# Run in background (Linux/Mac)
nohup python falsification_framework.py \
    --n-iterations 500 \
    --init-points 50 \
    --output-dir falsification_500 \
    > falsification.log 2>&1 &

# Monitor progress
tail -f falsification.log
```

## Troubleshooting

### "Connection refused" or "Cannot connect to CARLA"

**Problem**: CARLA server not running or wrong port

**Solution**:
```bash
# Check if CARLA is running
ps aux | grep CarlaUE4

# If not running, start it (Terminal 1)
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen

# Wait 30 seconds, then try again
```

### "ModuleNotFoundError: No module named 'X'"

**Problem**: Missing Python package

**Solution**:
```bash
# Install missing package
pip install X

# Or reinstall all requirements
pip install -r requirements.txt --upgrade
```

### "Simulation timed out"

**Problem**: Simulation taking too long or hung

**Solution**:
```bash
# Increase timeout in falsification_framework.py
# Line ~80: timeout=120  ->  timeout=300

# Or restart CARLA server
killall CarlaUE4
./CarlaUE4.sh -RenderOffScreen
```

### "No log file found"

**Problem**: Simulation failed silently

**Solution**:
```bash
# Run simulation directly to see error
cd csci513-miniproject1
python -m mp1_simulator test_data/scenario1_0.json

# Check for errors in output
```

## What's Next?

After your first successful run:

1. **Review Results**: Check `critical_scenarios.json` for interesting scenarios
2. **Visualize**: Open all plots in `analysis_output/`
3. **Customize**: Modify search space in `config/search_space.py`
4. **Scale Up**: Run 500-1000 iterations for research-quality results

## Quick Reference

### File Locations

- **Scenarios**: `falsification_output/scenarios/`
- **Logs**: `falsification_output/logs/`
- **Results**: `falsification_output/results/final_results.json`
- **Analysis**: `analysis_output/*.png`

### Common Commands

```bash
# Resume interrupted run
python falsification_framework.py \
    --resume-from falsification_output/results/checkpoint_0050.json

# Analyze any checkpoint
python analysis/pareto_analysis.py \
    falsification_output/results/checkpoint_0050.json

# Generate test scenario
python scenario_generator.py

# Test objective functions
python metrics/objective_functions.py
```

## Getting Help

1. Check `README.md` for detailed documentation
2. Review error messages carefully
3. Verify CARLA is running and accessible
4. Check Python package versions: `pip list`

## Success Checklist

After completion, you should have:

- [ ] Generated 10+ scenarios
- [ ] Simulation logs for each scenario
- [ ] Pareto front visualization (3D and 2D)
- [ ] Convergence plots showing improvement
- [ ] Critical scenarios report (JSON)
- [ ] Understanding of safety/plausibility/comfort trade-offs

**Congratulations!** You've successfully run the multi-objective falsification framework!

