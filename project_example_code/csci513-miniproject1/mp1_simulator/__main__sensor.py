#!/usr/bin/env python
"""
Sensor-Based Simulation Runner

This module runs CARLA simulations using the sensor-based simulator that:
1. Uses RADAR for distance measurement (affected by weather)
2. Uses simulated wheel encoders for velocity (with noise)
3. Applies weather conditions from scenario files
4. Logs both ground truth and sensor measurements for comparison

Usage:
    python -m mp1_simulator.sensor test_data/*.json --render
    
Or to use with the original ground-truth simulator:
    python -m mp1_simulator test_data/*.json --render
"""

import argparse
import csv
import logging
from collections import deque
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Any
import json
import os

import numpy as np

from mp1_controller.controller import Controller
from mp1_simulator.simulator_sensor import SensorSimulator, CONFIG, Observation, Mode

logger = logging.getLogger("SENSOR_SIMULATION")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensor-Based Adaptive Cruise Control Simulation in CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "test_data",
        nargs="+",
        help="JSON files containing the test data (scenarios).",
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "--log-dir",
        help="Directory to store the simulation trace",
        type=lambda p: Path(p).absolute(),
        default=Path.cwd() / "logs",
    )

    parser.add_argument(
        "--vid-dir",
        help="Directory to store camera video of the simulation",
        type=lambda p: Path(p).absolute(),
        default=Path.cwd() / "vids",
    )

    parser.add_argument(
        "--render",
        help="Render the Pygame display",
        action="store_true"
    )
    
    parser.add_argument(
        "--use-ground-truth",
        help="Use ground truth instead of sensors (for comparison)",
        action="store_true"
    )
    
    parser.add_argument(
        "--log-sensor-comparison",
        help="Log both ground truth and sensor values for analysis",
        action="store_true",
        default=True
    )

    return parser.parse_args()


class TraceRow(NamedTuple):
    """Single row of simulation trace data."""
    ego_velocity: float
    target_speed: float
    distance_to_lead: float
    ado_velocity: float
    mode: Mode


class SensorTraceRow(NamedTuple):
    """Extended trace row with sensor comparison data."""
    ego_velocity: float
    target_speed: float
    distance_to_lead: float
    ado_velocity: float
    mode: Mode
    collided: bool  # Collision flag
    # Sensor comparison fields
    gt_distance: float  # Ground truth distance
    sensor_distance: float  # Sensor-measured distance
    distance_error: float  # Absolute error
    gt_velocity: float  # Ground truth velocity
    sensor_velocity: float  # Sensor-measured velocity
    velocity_error: float  # Absolute error
    detection_prob: float  # Weather-based detection probability


def observation_to_trace_row(obs: Observation, sim: SensorSimulator, mode: Mode) -> TraceRow:
    """Convert observation to basic trace row."""
    return TraceRow(
        ego_velocity=obs.ego_velocity,
        target_speed=obs.desired_speed,
        distance_to_lead=obs.distance_to_lead,
        ado_velocity=sim._get_ado_velocity(),
        mode=mode
    )


def observation_to_sensor_trace_row(obs: Observation, sim: SensorSimulator, mode: Mode) -> SensorTraceRow:
    """Convert observation to extended trace row with sensor comparison."""
    comparison = sim.get_sensor_comparison()
    return SensorTraceRow(
        ego_velocity=obs.ego_velocity,
        target_speed=obs.desired_speed,
        distance_to_lead=obs.distance_to_lead,
        ado_velocity=sim._get_ado_velocity(),
        mode=mode,
        collided=sim.collided_event,  # Include collision status
        gt_distance=comparison['ground_truth_distance'],
        sensor_distance=comparison['sensor_distance'],
        distance_error=comparison['distance_error'],
        gt_velocity=comparison['ground_truth_velocity'],
        sensor_velocity=comparison['sensor_velocity'],
        velocity_error=comparison['velocity_error'],
        detection_prob=comparison['weather_detection_prob'],
    )


def extract_weather_from_scenario(scenario_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract weather parameters from scenario data.
    
    Handles both old format (no weather) and new format (with weather dict).
    """
    if 'weather' in scenario_data:
        return scenario_data['weather']
    
    # Try to find weather in parameters (from falsification framework)
    if 'parameters' in scenario_data:
        params = scenario_data['parameters']
        weather = {}
        weather_keys = ['fog_density', 'precipitation', 'precipitation_deposits',
                       'wind_intensity', 'sun_altitude_angle', 'cloudiness']
        for key in weather_keys:
            if key in params:
                weather[key] = params[key]
        if weather:
            # Calculate derived parameters
            if 'fog_density' in weather and weather['fog_density'] > 0:
                weather['fog_distance'] = 1000.0 * np.exp(-weather['fog_density'] / 30.0)
            else:
                weather['fog_distance'] = 1000.0
            weather['wetness'] = min(weather.get('precipitation', 0) / 2.0, 100.0)
            return weather
    
    # Default: clear weather
    return {
        'cloudiness': 0.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 0.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 0.0,
        'fog_distance': 1000.0,
        'wetness': 0.0,
    }


def run_episode(sim: SensorSimulator, controller: Controller, *,
                log_file: Path, video_file: Path, 
                log_sensor_comparison: bool = True):
    """
    Run a single simulation episode.
    
    Args:
        sim: Sensor-based simulator instance
        controller: ACC controller
        log_file: Path to save trace CSV
        video_file: Path to save video
        log_sensor_comparison: Whether to log sensor vs ground truth comparison
    """
    if log_sensor_comparison:
        trace = deque()  # type: deque[SensorTraceRow]
    else:
        trace = deque()  # type: deque[TraceRow]
    
    row = sim.reset()
    
    if log_sensor_comparison:
        trace.append(observation_to_sensor_trace_row(row, sim, Mode.CRUISING))
    else:
        trace.append(observation_to_trace_row(row, sim, Mode.CRUISING))
    
    while True:
        action, mode = controller.run_step(row)
        row = sim.step(action)
        
        if log_sensor_comparison:
            trace.append(observation_to_sensor_trace_row(row, sim, mode))
        else:
            trace.append(observation_to_trace_row(row, sim, mode))

        if sim.completed:
            break

    sim.dump_video(str(video_file))
    
    # Write trace to CSV
    with open(log_file, "w", newline='') as flog:
        csv_stream = csv.writer(flog)
        
        if log_sensor_comparison:
            # Extended header with sensor comparison
            csv_stream.writerow([
                "timestep",
                "time_elapsed",
                "ego_velocity",
                "desired_speed",
                "distance_to_lead",
                "lead_speed",
                "mode",
                "collided",  # Include collision column
                "gt_distance",
                "sensor_distance", 
                "distance_error",
                "gt_velocity",
                "sensor_velocity",
                "velocity_error",
                "detection_probability"
            ])
            
            for i, row in enumerate(trace):
                csv_stream.writerow([
                    i,
                    sim.dt * i,
                    row.ego_velocity,
                    row.target_speed,
                    row.distance_to_lead,
                    row.ado_velocity,
                    row.mode.value,
                    1 if row.collided else 0,  # Log collision as 0/1
                    row.gt_distance,
                    row.sensor_distance if row.sensor_distance != float('inf') else -1,
                    row.distance_error if row.distance_error != float('inf') else -1,
                    row.gt_velocity,
                    row.sensor_velocity,
                    row.velocity_error,
                    row.detection_prob,
                ])
        else:
            # Standard header
            csv_stream.writerow([
                "timestep",
                "time_elapsed", 
                "ego_velocity",
                "desired_speed",
                "distance_to_lead",
                "lead_speed",
                "mode"
            ])

            for i, row in enumerate(trace):
                csv_stream.writerow([
                    i,
                    sim.dt * i,
                    row.ego_velocity,
                    row.target_speed,
                    row.distance_to_lead,
                    row.ado_velocity,
                    row.mode.value
                ])


def main():
    args = parse_args()
    log_dir: Path = args.log_dir
    vid_dir: Path = args.vid_dir
    test_files = args.test_data

    if log_dir.is_dir():
        logger.warning(
            "Log directory %s already exists. Existing logs may be overwritten.",
            str(log_dir),
        )
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    if vid_dir.is_dir():
        logger.warning(
            "Vid directory %s already exists. Existing vids may be overwritten.",
            str(vid_dir),
        )
    else:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SENSOR-BASED simulator
    sim = SensorSimulator(
        render=args.render,
        log_dir=log_dir,
        use_ground_truth=args.use_ground_truth,
    )
    
    if args.use_ground_truth:
        logger.info("Running with GROUND TRUTH (sensors disabled)")
    else:
        logger.info("Running with SENSOR-BASED measurements (radar + wheel encoder)")

    for test_file in test_files:
        with open(test_file, "r") as file:
            scenario_data = json.load(file)

        # Extract components from scenario
        initial_ego_state = scenario_data['ego']
        initial_lead_state = scenario_data['lead']
        ado_actions = scenario_data['ado_actions']
        
        # ========== APPLY WEATHER ==========
        weather_params = extract_weather_from_scenario(scenario_data)
        sim.set_weather(weather_params)
        
        logger.info(f"Weather: fog={weather_params.get('fog_density', 0):.1f}%, "
                   f"rain={weather_params.get('precipitation', 0):.1f}%")

        # Initialize controller
        controller = Controller(
            distance_threshold=CONFIG["distance_threshold"],
        )

        logger.info("Running test data: %s", test_file)

        out_file = os.path.splitext(os.path.basename(test_file))[0]
        episode_name = f"episode-{out_file}.csv"
        video_name = f"episode-{out_file}.mp4"

        # Set up scenario
        sim._set_ado_control(ado_actions)
        sim.set_spawn_points(initial_ego_state, initial_lead_state)
        
        # Run episode
        run_episode(
            sim, controller,
            log_file=(log_dir / episode_name),
            video_file=(vid_dir / video_name),
            log_sensor_comparison=args.log_sensor_comparison
        )

        logger.info("Episode saved: %s", (log_dir / episode_name))
        
        # Print sensor performance summary
        if not args.use_ground_truth:
            comparison = sim.get_sensor_comparison()
            logger.info(f"  Final distance error: {comparison['distance_error']:.2f}m")
            logger.info(f"  Final velocity error: {comparison['velocity_error']:.3f}m/s")


if __name__ == "__main__":
    main()

