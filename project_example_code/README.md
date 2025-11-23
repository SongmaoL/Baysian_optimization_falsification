# Multi-Objective Falsification Framework for CARLA

A semantic falsification framework for finding critical test scenarios in CARLA that reveal optimal trade-offs between **safety violations**, **physical plausibility**, and **passenger comfort**.

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running quickly
- **[Next Steps](docs/NEXT_STEPS.md)** - Current status and what to do next
- **[Analysis Results](docs/ANALYSIS.md)** - 107 iteration analysis
- **[Full Documentation Index](docs/)** - All documentation files

## Project Overview

This framework uses **Multi-Objective Bayesian Optimization** to efficiently search the space of environmental parameters (weather conditions, lead vehicle behavior) to find scenarios that:

1. **Violate safety specifications** - Near-misses, collisions, dangerous driving
2. **Remain physically plausible** - Realistic vehicle dynamics and weather conditions  
3. **Reveal comfort issues** - High jerk, uncomfortable accelerations

The goal is to find the **Pareto front** of scenarios that represent optimal trade-offs between these three competing objectives.

## Architecture

```
.
├── config/                          # Search space configuration
│   ├── __init__.py
│   └── search_space.py             # Parameter bounds for weather, vehicle behavior
│
├── metrics/                         # Objective function evaluation
│   ├── __init__.py
│   └── objective_functions.py      # Safety, plausibility, comfort metrics
│
├── analysis/                        # Pareto front analysis and visualization
│   ├── __init__.py
│   └── pareto_analysis.py          # Plotting and scenario selection tools
│
├── docs/                            # Documentation
│   ├── QUICKSTART.md               # Quick start guide
│   ├── NEXT_STEPS.md               # Current status and next actions
│   ├── ANALYSIS.md                 # Results analysis
│   └── ...                         # More documentation
│
├── scenario_generator.py           # Converts parameters to CARLA scenarios
├── multi_objective_bo.py           # Multi-objective Bayesian optimization
├── falsification_framework.py      # Main orchestration script
│
├── csci513-miniproject1/           # CARLA project (existing)
│   ├── mp1_controller/             # Adaptive cruise control
│   ├── mp1_simulator/              # CARLA simulation wrapper
│   └── mp1_evaluation/             # STL specification evaluation
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

### Prerequisites

1. **Python 3.8+**
2. **CARLA 0.9.15** - Follow [CARLA installation guide](https://carla.readthedocs.io/en/0.9.15/start_quickstart/)
3. **GPU** - Required for CARLA (or use HPC resources)

### Installation

1. Clone the repository and navigate to the project directory:

```bash
cd project_example_code
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install the CARLA mini-project:

```bash
cd csci513-miniproject1
pip install -e .
cd ..
```

4. Verify CARLA installation by starting the server:

```bash
# Direct installation
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen

# Docker installation
cd csci513-miniproject1
make run-carla
```

## Usage

### Quick Start

1. **Start CARLA server** (in terminal 1):

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen
```

2. **Run falsification** (in terminal 2):

```bash
python falsification_framework.py \
    --carla-project csci513-miniproject1 \
    --output-dir falsification_output \
    --n-iterations 100 \
    --init-points 10
```

3. **Analyze results**:

```bash
python analysis/pareto_analysis.py \
    falsification_output/results/final_results.json \
    --output-dir analysis_output
```

### Command Line Options

#### Falsification Framework (`falsification_framework.py`)

```bash
python falsification_framework.py [OPTIONS]

Options:
  --carla-project PATH       Path to CARLA project directory (default: csci513-miniproject1)
  --output-dir PATH          Directory to save outputs (default: falsification_output)
  --n-iterations INT         Total iterations (default: 100, recommend: 500-1000)
  --init-points INT          Random initial iterations (default: 10)
  --checkpoint-interval INT  Save checkpoint every N iterations (default: 10)
  --resume-from PATH         Resume from checkpoint file
  --render                   Render simulation visualization (slower)
  --random-state INT         Random seed (default: 42)
```

#### Analysis (`analysis/pareto_analysis.py`)

```bash
python analysis/pareto_analysis.py RESULTS_FILE [OPTIONS]

Arguments:
  RESULTS_FILE              Path to results JSON file

Options:
  --output-dir PATH         Directory to save analysis outputs (default: analysis_output)
```

### Example: Full Workflow

```bash
# Terminal 1: Start CARLA
./CarlaUE4.sh -RenderOffScreen

# Terminal 2: Run falsification (500 iterations)
python falsification_framework.py \
    --n-iterations 500 \
    --init-points 50 \
    --checkpoint-interval 25 \
    --output-dir falsification_500

# After completion, analyze results
python analysis/pareto_analysis.py \
    falsification_500/results/final_results.json \
    --output-dir falsification_500/analysis
```

## Search Space

The framework searches over the following parameters:

### Weather Parameters
- **Fog density** (0-80%): Reduces visibility
- **Precipitation** (0-100%): Rain intensity
- **Precipitation deposits** (0-100%): Water on surfaces
- **Wind intensity** (0-50%): Wind strength
- **Sun altitude angle** (-30° to 90°): Time of day
- **Cloudiness** (0-100%): Cloud coverage

### Lead Vehicle Behavior
- **Base throttle** (0.2-0.6): Average speed
- **Behavior frequency** (0.05-0.3 Hz): How often behavior changes
- **Throttle variation** (0-0.4): Magnitude of speed changes
- **Brake probability** (0-0.3): Likelihood of braking events
- **Brake intensity** (0.1-0.8): Strength of braking
- **Brake duration** (5-30 timesteps): How long to brake

### Initial Conditions
- **Initial distance** (15-80m): Starting gap between vehicles
- **Initial ego velocity** (5-20 m/s): Ego car starting speed
- **Initial lead velocity** (5-25 m/s): Lead car starting speed

## Objective Functions

### 1. Safety Score (Minimize)
**Goal**: Find unsafe scenarios

Metrics:
- **Time-to-Collision (TTC)**: Lower TTC = higher risk
- **Minimum distance**: Closer = more dangerous
- **Collision detection**: Binary collision flag

Score range: [0, 100] where 0 = collision, 100 = very safe

### 2. Plausibility Score (Maximize)  
**Goal**: Ensure physical realism

Metrics:
- **Maximum acceleration**: Must be < 2g (20 m/s²)
- **Maximum jerk**: Rate of acceleration change
- **Vehicle dynamics**: Realistic behavior constraints

Score range: [0, 100] where 100 = highly realistic, 0 = impossible

### 3. Comfort Score (Minimize)
**Goal**: Find uncomfortable scenarios

Metrics:
- **Total jerk**: Accumulated jerky motion
- **Hard braking/acceleration events**: Count of > 0.3g events
- **Smooth driving**: Inverse of motion variability

Score range: [0, 100] where 0 = very uncomfortable, 100 = smooth

## Output Files

### During Falsification

```
falsification_output/
├── scenarios/              # Generated scenario JSON files
│   ├── scenario_0000.json
│   ├── scenario_0001.json
│   └── ...
│
├── logs/                   # Simulation trace CSV files
│   ├── scenario_0000.csv
│   ├── scenario_0001.csv
│   └── ...
│
├── vids/                   # Simulation videos (if --render)
│   ├── scenario_0000.mp4
│   └── ...
│
└── results/                # Optimization state
    ├── checkpoint_0010.json
    ├── checkpoint_0020.json
    └── final_results.json  # Final Pareto front
```

### After Analysis

```
analysis_output/
├── pareto_front_3d.png            # 3D Pareto front visualization
├── pareto_front_2d.png            # 2D pairwise projections
├── convergence.png                # Objective convergence over time
├── parameter_distributions.png    # Parameter histograms
└── critical_scenarios.json        # Selected critical test scenarios
```

## Interpreting Results

### Pareto Front

The **Pareto front** contains scenarios where improving one objective requires sacrificing another:

- **Low safety + High plausibility**: Realistic dangerous scenarios (most valuable!)
- **Low safety + Low plausibility**: Unrealistic failures (filter out)
- **High safety + Low comfort**: Safe but uncomfortable driving
- **Balanced**: Scenarios with trade-offs across all objectives

### Critical Scenarios

The analysis tool automatically selects ~20 critical scenarios:
1. **Most unsafe** (lowest safety score)
2. **Most realistic** (highest plausibility score)
3. **Most uncomfortable** (lowest comfort score)
4. **Diverse samples** across the Pareto front

These scenarios should be used for:
- Testing the controller
- Understanding failure modes
- Setting safety requirements
- Training/validation data

## Customization

### Adding New Parameters

Edit `config/search_space.py`:

```python
SEARCH_SPACE = {
    # Add new parameter
    "new_parameter": (min_value, max_value),
    # ...
}
```

### Modifying Objectives

Edit `metrics/objective_functions.py`:

```python
def calculate_safety_score(trace_df, dt=0.1):
    # Modify safety calculation
    # ...
    return safety_score
```

### Changing Acquisition Strategy

Edit `multi_objective_bo.py`:

```python
def _acquisition_function(self, X_candidate, weights, exploration_param=2.0):
    # Modify acquisition function
    # Try different strategies: EI, UCB, PI, etc.
    # ...
```

## Troubleshooting

### CARLA Connection Issues

```bash
# Check if CARLA server is running
ps aux | grep CarlaUE4

# Restart CARLA server
killall CarlaUE4
./CarlaUE4.sh -RenderOffScreen
```

### Simulation Failures

- **Timeout**: Increase timeout in `falsification_framework.py` (line: `timeout=120`)
- **Render issues**: Run without `--render` flag
- **Memory issues**: Reduce `--n-iterations` or run in batches

### Analysis Errors

- **Empty Pareto front**: Need more iterations (try 100+)
- **Import errors**: Ensure all modules are installed: `pip install -r requirements.txt`
- **Visualization issues**: Update matplotlib: `pip install --upgrade matplotlib`

## Performance Tips

1. **Parallelization**: Run multiple CARLA instances on different ports
2. **HPC**: Use university HPC resources for large experiments (500-1000 iterations)
3. **Checkpoints**: Use `--checkpoint-interval` to save progress frequently
4. **Resume**: Use `--resume-from` to continue interrupted runs

## Folder Structure

## Citation

If you use this framework in your research, please cite:

```
@misc{falsification_framework_2024,
  title={Multi-Objective Falsification Framework for CARLA},
  author={[Your Team Names]},
  year={2024},
  howpublished={\\url{https://github.com/your-repo}}
}
```

## References

1. [CARLA Documentation](https://carla.readthedocs.io/)
2. [Bayesian Optimization](https://arxiv.org/pdf/1807.02811.pdf)
3. [Multi-Objective BO](https://arxiv.org/pdf/2109.10964)
4. [Falsification for CPS](https://vbn.aau.dk/ws/portalfiles/portal/698944696/Usage_aware_Falsification_for_Cyber_Physical_Systems.pdf)

## License

[Your License Here]

## Contact

For questions or issues, please open an issue on GitHub or contact the team.

