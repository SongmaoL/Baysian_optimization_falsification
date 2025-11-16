# Formal Verification of Perception Systems: Realistic Parameter Identification in Falsification Frameworks

**Team:** Calvin Tai, Andre Jia, Elizabeth Skehan, Abhir Karande, Songmao Li

## Purpose

Semantic falsification framework for an AV perception to control pipeline in CARLA that searches environmental parameters to find scenarios that violate STL safety specs

Maximize plausibility: We don't want to find failures that are unlikely to occur or physically impossible

A failure/falsification in ADAS, for example, isn't just a crash. It can be a near-miss, extreme passenger discomfort, traffic law violation, etc. Rather than getting unrealistic falsification scenarios, we can find the most critical set that is at the trade-off between each of our objectives (safety, plausibility, passenger comfort)

## Task Breakdown/Schedule/Assignment 

**Environment Setup and Baseline Implementation, Milestone 1 by Oct 24** (Calvin, Elizabeth)
- Setup Carla
- Baseline single safety objective falsification, Bayesian Optimization loop to find worst-case failure for single objective. 

**Define a multi-objective Bayesian Optimization, Milestone 2 by Oct 31** (Songmao, Andre, Abhir)
- Objective 1 is safety score, Objective 2 is plausibility (based on G-Forces/acceleration), Objective 3 is passenger discomfort. 
- Adapt surrogate model and acquisition functions

**Scenario and Objective Implementation, Milestone 3 by Nov 7** (Andre, Elizabeth, Calvin)
- Input parameter space for scenarios
- Objective Metric (i.e. Time-to-Collision for safety score, adversary vehicle acceleration for plausibility score, ego vehicle jerk and lateral acceleration for comfort score

**Experimentation and Analysis, Milestone 4 by Nov 21** (Abhir, Songmao, Andre, Elizabeth, Calvin) 
- Execution of Falsification runs using our multi-objective Bayesian Optimization framework for 500-1000 simulations
- Visualize the trade-offs between safety, plausibility, and comfort in order to choose a "Pareto front"

## Expected Outcome

Generation of a set of critical test scenarios, revealing the optimal trade-offs between safety, plausibility, and passenger comfort. 

## Resources Required

CARLA for environment parameters and simulation

## Literature Survey

- https://arxiv.org/pdf/2209.06735
- https://dl.acm.org/doi/10.1145/3126521
- https://vbn.aau.dk/ws/portalfiles/portal/698944696/Usage_aware_Falsification_for_Cyber_Physical_Systems.pdf
- https://ieeexplore.ieee.org/document/8666747 - impact of weather on ADAS systems

## Immediate Tasks

### Parameter Set for Search Space
- Weather: fog conditions, precipitation, sun angle, cloudiness
  - Reference: https://carla.readthedocs.io/en/latest/tuto_M_custom_weather_landscape/
- Lead vehicle dynamics: How often the lead car changes speed
- Other environmental parameters

### Metrics

**Safety** (continuous metrics):
- Time-to-collision: lower minimum TTC over a scenario indicates higher risk
- Minimum Distance: minimum distance between ego and lead vehicles
- Simulator provides `collided_event`, which is a direct binary measure

**Plausibility** (vehicle dynamics should not defy physical laws):
- Maximum acceleration: passenger car can't brake at 2g
- Maximum jerk: The rate of change of acceleration shouldn't exceed a threshold
- Steering rate: How fast the steering wheel is turned

**Passenger Comfort**:
- Jerk: high jerk values are uncomfortable. Minimize total jerk experienced
- Lateral Acceleration: High acceleration in turns can be uncomfortable
- Hard Braking/Acceleration Events: Count times when acceleration/deceleration exceeds comfort threshold (> 0.3g)

### Basic Measurements Added to MP1 Eval Script

Reference: https://pastebin.com/fY4gsf1x

Functions: `calculate_minimum_ttc()`, `calculate_minimum_distance()`, `calculate_maximum_jerk()`

## Implementation

- Carla (done)
- Bayesian optimization: https://github.com/bayesian-optimization/BayesianOptimization
  - Extend for multiple objectives: https://arxiv.org/pdf/2109.10964

