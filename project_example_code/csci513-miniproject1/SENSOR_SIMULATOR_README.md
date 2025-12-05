# Sensor-Based Simulator for ACC Testing

This document describes the **sensor-based simulator** that replaces ground truth measurements with realistic sensor models, enabling weather to actually affect controller performance.

## Overview

The original simulator (`simulator.py`) uses **ground truth** from CARLA for all measurements:
- Distance to lead vehicle: Computed from exact world positions
- Ego velocity: Read directly from physics engine

The new **sensor-based simulator** (`simulator_sensor.py`) uses:
- **RADAR sensor**: For distance measurement to lead vehicle
- **Simulated wheel encoders**: For ego velocity with realistic noise
- **Weather effects**: Applied to both CARLA world and sensor performance

## Key Differences

| Aspect | Original Simulator | Sensor-Based Simulator |
|--------|-------------------|----------------------|
| Distance Measurement | Ground truth (exact) | RADAR sensor (noisy, weather-affected) |
| Velocity Measurement | Ground truth (exact) | Wheel encoder simulation (noisy) |
| Weather | Hardcoded ClearNoon | Applied from scenario file |
| Weather Impact | None | Degrades sensor performance |
| Controller Testing | Ideal conditions | Realistic conditions |

## Weather Effects on Sensors

### Fog Effects
- **Range reduction**: Up to 60% reduction at 80% fog density
- **Noise increase**: Up to 2m additional measurement noise
- **Detection probability**: Decreases with fog density

### Rain Effects
- **Noise increase**: Up to 1.5m additional measurement noise
- **False detections**: Up to 10% probability at heavy rain
- **Velocity error**: Increased due to wheel slip on wet roads

### Example Weather Impact

| Weather Condition | Detection Prob | Range Reduction | Distance Noise |
|------------------|----------------|-----------------|----------------|
| Clear (0% fog, 0% rain) | 100% | 0% | ±0.5m |
| Light fog (30%) | 93% | 22% | ±1.1m |
| Heavy fog (80%) | 76% | 60% | ±2.1m |
| Heavy rain (100%) | 90% | 0% | ±2.0m |
| Fog + Rain (50%/50%) | 80% | 37% | ±2.5m |

## Usage

### Running with Sensor-Based Simulator

```bash
# Navigate to the project directory
cd csci513-miniproject1

# Run with sensors (weather will affect measurements)
python -m mp1_simulator.sensor test_data/*.json --render

# Compare with ground truth
python -m mp1_simulator.sensor test_data/*.json --use-ground-truth
```

### Command Line Options

```
--render                  Show visualization
--use-ground-truth        Use ground truth instead of sensors (for comparison)
--log-sensor-comparison   Log both sensor and ground truth values (default: True)
--log-dir PATH           Directory for trace logs
--vid-dir PATH           Directory for videos
```

### Output Files

The sensor simulator produces extended trace files with additional columns:

```csv
timestep,time_elapsed,ego_velocity,desired_speed,distance_to_lead,lead_speed,mode,gt_distance,sensor_distance,distance_error,gt_velocity,sensor_velocity,velocity_error,detection_probability
0,0.0,5.21,20.0,15.78,10.5,0,15.78,16.23,0.45,5.21,5.18,0.03,0.85
...
```

New columns:
- `gt_distance`: Ground truth distance (meters)
- `sensor_distance`: RADAR-measured distance (meters, -1 if no detection)
- `distance_error`: |gt_distance - sensor_distance|
- `gt_velocity`: Ground truth ego velocity (m/s)
- `sensor_velocity`: Wheel encoder velocity (m/s)
- `velocity_error`: |gt_velocity - sensor_velocity|
- `detection_probability`: Weather-adjusted probability of radar detection

## Integration with Falsification Framework

To use the sensor-based simulator with the multi-objective Bayesian optimization framework:

1. **Update `falsification_framework.py`** to use `SensorSimulator`:

```python
# In CARLASimulationRunner, change the command to use sensor module
cmd = [
    sys.executable, "-m", "mp1_simulator.sensor",  # Changed!
    str(scenario_path.resolve()),
    "--log-dir", str(self.log_dir.resolve()),
    "--vid-dir", str(self.vid_dir.resolve()),
]
```

2. Weather parameters are already included in scenario files and will now be applied!

## Sensor Configuration

The sensor parameters can be adjusted in `simulator_sensor.py`:

```python
CONFIG = {
    ...
    "use_ground_truth": False,  # Set True to disable sensors
    "radar_range": 100.0,       # Maximum detection range (meters)
    "radar_fov": 30.0,          # Field of view (degrees)
    "velocity_noise_std": 0.1,  # Wheel encoder noise (m/s)
}
```

## RADAR Sensor Details

The CARLA RADAR sensor is configured as:
- **Horizontal FOV**: 30° (typical for automotive ACC radar)
- **Vertical FOV**: 20°
- **Range**: 100m (adjustable)
- **Update rate**: Synchronized with simulation (10 Hz)
- **Mounting position**: Front bumper (x=2.5m, z=0.5m)

## Validation

To validate sensor accuracy vs ground truth:

```python
# After each step, get comparison data
comparison = sim.get_sensor_comparison()
print(f"Distance error: {comparison['distance_error']:.2f}m")
print(f"Velocity error: {comparison['velocity_error']:.3f}m/s")
print(f"Detection probability: {comparison['weather_detection_prob']:.2f}")
```

## Known Limitations

1. **RADAR sensor requires CARLA 0.9.15+**: Earlier versions may have different sensor behavior
2. **Simplified weather model**: Real weather effects are more complex
3. **No multi-path effects**: Real radar can have reflections from guardrails, etc.
4. **Velocity from ground truth with noise**: True wheel encoder simulation would require tire model

## Files

| File | Description |
|------|-------------|
| `simulator_sensor.py` | Sensor-based simulator class |
| `__main__sensor.py` | Command-line entry point |
| `sensor.py` | Module wrapper for `python -m` invocation |
| `simulator.py` | Original ground-truth simulator (unchanged) |

## Example: Weather Impact Analysis

```python
from mp1_simulator.simulator_sensor import SensorSimulator, WeatherEffects

# Create weather effects for analysis
weather = {
    'fog_density': 50.0,
    'precipitation': 30.0,
}

effects = WeatherEffects.from_weather_params(weather)
print(f"Detection probability: {effects.detection_probability:.1%}")
print(f"Range reduction: {effects.fog_range_reduction:.1%}")
print(f"Distance noise: ±{0.5 + effects.fog_noise_increase + effects.rain_noise_increase:.1f}m")
```

## References

- CARLA RADAR Sensor: https://carla.readthedocs.io/en/latest/ref_sensors/#radar-sensor
- CARLA Weather: https://carla.readthedocs.io/en/latest/python_api/#carlaweatherparameters
- Automotive RADAR fundamentals: https://www.ti.com/lit/an/swra553a/swra553a.pdf


