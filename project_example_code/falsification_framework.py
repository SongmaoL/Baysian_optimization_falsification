"""
Main Falsification Framework

This script orchestrates the multi-objective falsification process:
1. Uses multi-objective BO to suggest parameters
2. Generates scenarios from parameters  
3. Runs CARLA simulations
4. Evaluates objectives (safety, plausibility)
5. Updates Pareto front
6. Repeats for specified iterations
"""

import argparse
import time
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional
import json
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.search_space import get_parameter_bounds, validate_parameters
from scenario_generator import generate_scenario, save_scenario_json
from multi_objective_bo import MultiObjectiveBayesianOptimization, OptimizationStrategy
from metrics.objective_functions import evaluate_trace_file


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class CARLASimulationRunner:
    """
    Wrapper for running CARLA simulations with generated scenarios.
    """
    
    def __init__(self, 
                 carla_project_dir: Path,
                 log_dir: Path,
                 vid_dir: Path,
                 render: bool = False,
                 use_sensors: bool = True):
        """
        Initialize simulation runner.
        
        Args:
            carla_project_dir: Path to csci513-miniproject1 directory
            log_dir: Directory to save simulation logs
            vid_dir: Directory to save videos
            render: Whether to render visualization
            use_sensors: If True, use sensor-based simulator with weather effects
        """
        self.carla_project_dir = Path(carla_project_dir)
        self.log_dir = Path(log_dir)
        self.vid_dir = Path(vid_dir)
        self.render = render
        self.use_sensors = use_sensors
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.vid_dir.mkdir(parents=True, exist_ok=True)
    
    def run_simulation(self, scenario_path: Path) -> Optional[Path]:
        """
        Run a single CARLA simulation with a scenario file.
        
        Args:
            scenario_path: Path to scenario JSON file
            
        Returns:
            Path to simulation log CSV file (None if failed)
        """
        # Choose simulator module based on configuration
        if self.use_sensors:
            simulator_module = "mp1_simulator.sensor"  # Sensor-based with weather effects
        else:
            simulator_module = "mp1_simulator"  # Ground truth
        
        # Build command
        cmd = [
            sys.executable, "-m", simulator_module,
            str(scenario_path),
            "--log-dir", str(self.log_dir.resolve()),
            "--vid-dir", str(self.vid_dir.resolve()),
        ]
        
        if self.render:
            cmd.append("--render")
        
        try:
            # Run simulation
            print(f"Running simulation: {scenario_path.name}")
            result = subprocess.run(
                cmd,
                cwd=self.carla_project_dir,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Simulation failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return None
            
            # Find the generated log file
            # Simulation creates logs with scenario name, often prepended with 'episode-'
            scenario_name = scenario_path.stem
            
            # Check for both possible filenames
            possible_names = [
                f"episode-{scenario_name}.csv",
                f"{scenario_name}.csv"
            ]
            
            log_file = None
            for name in possible_names:
                candidate = self.log_dir / name
                if candidate.exists():
                    log_file = candidate
                    break
            
            if log_file is None:
                # Try to find most recent log file
                log_files = sorted(self.log_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if log_files:
                    log_file = log_files[0]
                else:
                    print("No log file found after simulation")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    return None
            
            print(f"  Log saved to: {log_file}")
            return log_file
            
        except subprocess.TimeoutExpired:
            print("Simulation timed out")
            return None
        except Exception as e:
            print(f"Error running simulation: {e}")
            return None


# ============================================================================
# FALSIFICATION ORCHESTRATOR
# ============================================================================

class FalsificationOrchestrator:
    """
    Main orchestrator for the falsification framework.
    """
    
    def __init__(self,
                 carla_project_dir: Path,
                 output_dir: Path,
                 render: bool = False,
                 strategy: str = "multi_objective",
                 random_state: int = 42,
                 use_sensors: bool = True):
        """
        Initialize falsification orchestrator.
        
        Args:
            carla_project_dir: Path to csci513-miniproject1 directory
            output_dir: Directory to save all outputs
            render: Whether to render simulations
            strategy: Optimization strategy ("multi_objective", "single_objective_safety", "random_search")
            random_state: Random seed
            use_sensors: If True, use sensor-based simulator with weather effects (default: True)
        """
        self.carla_project_dir = Path(carla_project_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_sensors = use_sensors
        
        # Create subdirectories
        self.scenarios_dir = self.output_dir / "scenarios"
        self.logs_dir = self.output_dir / "logs"
        self.vids_dir = self.output_dir / "vids"
        self.results_dir = self.output_dir / "results"
        
        for d in [self.scenarios_dir, self.logs_dir, self.vids_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.optimizer = MultiObjectiveBayesianOptimization(
            parameter_bounds=get_parameter_bounds(),
            strategy=strategy,
            random_state=random_state
        )
        
        self.simulator = CARLASimulationRunner(
            carla_project_dir=carla_project_dir,
            log_dir=self.logs_dir,
            vid_dir=self.vids_dir,
            render=render,
            use_sensors=use_sensors  # Pass sensor flag
        )
        
        print(f"Simulator mode: {'SENSOR-BASED (weather affects perception)' if use_sensors else 'GROUND TRUTH'}")
        
        self.random_state = random_state
        np.random.seed(random_state)
    
    def run_iteration(self, iteration: int) -> bool:
        """
        Run a single falsification iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"ITERATION {iteration}")
        print("=" * 80)
        
        # 1. Get next parameters from optimizer
        print("\n[1/5] Suggesting parameters...")
        params = self.optimizer.suggest_next()
        
        print("Parameters:")
        for key, value in list(params.items())[:5]:  # Show first 5
            print(f"  {key}: {value:.4f}")
        print(f"  ... and {len(params) - 5} more")
        
        # Validate parameters
        if not validate_parameters(params):
            print("ERROR: Invalid parameters suggested!")
            return False
        
        # 2. Generate scenario
        print("\n[2/5] Generating scenario...")
        scenario = generate_scenario(
            params,
            num_timesteps=200,
            dt=0.1,
            seed=self.random_state + iteration
        )
        
        scenario_path = self.scenarios_dir / f"scenario_{iteration:04d}.json"
        save_scenario_json(scenario, scenario_path)
        
        # 3. Run simulation
        print("\n[3/5] Running CARLA simulation...")
        # Ensure we pass the absolute path to the scenario file
        log_path = self.simulator.run_simulation(scenario_path.resolve())
        
        if log_path is None:
            print("ERROR: Simulation failed!")
            return False
        
        # 4. Evaluate objectives
        print("\n[4/5] Evaluating objectives...")
        try:
            objectives = evaluate_trace_file(log_path, dt=0.1)
            print(f"  Safety:      {objectives['safety']:.2f}")
            print(f"  Plausibility: {objectives['plausibility']:.2f}")
            
            # Warn if plausibility is 0 (scenario violates physics)
            if objectives['plausibility'] == 0.0:
                print("  WARNING: Plausibility=0 (scenario violates physical constraints)")
        except Exception as e:
            print(f"ERROR: Failed to evaluate objectives: {e}")
            return False
        
        # 5. Register result with optimizer
        print("\n[5/5] Updating optimizer...")
        metadata = {
            'scenario_path': str(scenario_path),
            'log_path': str(log_path),
            'timestamp': time.time()
        }
        self.optimizer.register_evaluation(params, objectives, metadata)
        
        pareto_size = len(self.optimizer.get_pareto_front())
        print(f"  Pareto front size: {pareto_size}")
        
        return True
    
    def run_falsification(self, 
                         n_iterations: int,
                         init_points: int = 10,
                         checkpoint_interval: int = 10,
                         resume_from: Optional[Path] = None):
        """
        Run the complete falsification process.
        
        Args:
            n_iterations: Total number of iterations
            init_points: Number of random initial iterations
            checkpoint_interval: Save checkpoint every N iterations
            resume_from: Path to checkpoint file to resume from
        """
        start_time = time.time()
        
        # Resume from checkpoint if provided
        start_iteration = 0
        if resume_from and resume_from.exists():
            print(f"Resuming from checkpoint: {resume_from}")
            self.optimizer.load_state(resume_from)
            start_iteration = self.optimizer.iteration
        
        print("\n" + "=" * 80)
        print("STARTING FALSIFICATION")
        print("=" * 80)
        print(f"Total iterations: {n_iterations}")
        print(f"Initial random iterations: {init_points}")
        print(f"Starting from iteration: {start_iteration}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        successful_iterations = 0
        failed_iterations = 0
        
        for i in range(start_iteration, n_iterations):
            try:
                success = self.run_iteration(i)
                
                if success:
                    successful_iterations += 1
                else:
                    failed_iterations += 1
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    checkpoint_path = self.results_dir / f"checkpoint_{i+1:04d}.json"
                    self.optimizer.save_state(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                
                # Print progress
                elapsed = time.time() - start_time
                avg_time = elapsed / (i - start_iteration + 1)
                remaining = (n_iterations - i - 1) * avg_time
                
                print(f"\nProgress: {i+1}/{n_iterations} ({(i+1)/n_iterations*100:.1f}%)")
                print(f"Elapsed: {elapsed/60:.1f}min, Estimated remaining: {remaining/60:.1f}min")
                print(f"Success: {successful_iterations}, Failed: {failed_iterations}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving checkpoint...")
                checkpoint_path = self.results_dir / f"checkpoint_interrupted_{i:04d}.json"
                self.optimizer.save_state(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                break
            except Exception as e:
                print(f"\nERROR in iteration {i}: {e}")
                failed_iterations += 1
                continue
        
        # Final save
        final_path = self.results_dir / "final_results.json"
        self.optimizer.save_state(final_path)
        
        # Print summary
        print("\n" + "=" * 80)
        print("FALSIFICATION COMPLETE")
        print("=" * 80)
        print(f"Total iterations: {successful_iterations + failed_iterations}")
        print(f"Successful: {successful_iterations}")
        print(f"Failed: {failed_iterations}")
        print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Results saved to: {final_path}")
        
        self.optimizer.print_summary()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Objective Falsification Framework for CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--carla-project",
        type=Path,
        default=Path("csci513-miniproject1"),
        help="Path to CARLA project directory (csci513-miniproject1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("falsification_output"),
        help="Directory to save all outputs"
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Total number of falsification iterations"
    )
    
    parser.add_argument(
        "--init-points",
        type=int,
        default=15,
        help="Number of random initial iterations (recommended: 15-20 for 14D space)"
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N iterations"
    )
    
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["multi_objective", "single_objective_safety", "random_search"],
        default="multi_objective",
        help="Optimization strategy: multi_objective, single_objective_safety, or random_search"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render simulation visualization"
    )
    
    parser.add_argument(
        "--use-sensors",
        action="store_true",
        default=True,
        help="Use sensor-based simulator with weather effects (default: True)"
    )
    
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Use ground-truth simulator (disable sensors)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if CARLA project exists
    if not args.carla_project.exists():
        print(f"ERROR: CARLA project directory not found: {args.carla_project}")
        print("Please specify the correct path with --carla-project")
        sys.exit(1)
    
    # Determine sensor usage (--no-sensors overrides default)
    use_sensors = not args.no_sensors
    
    # Initialize orchestrator
    orchestrator = FalsificationOrchestrator(
        carla_project_dir=args.carla_project,
        output_dir=args.output_dir,
        render=args.render,
        strategy=args.strategy,
        random_state=args.random_state,
        use_sensors=use_sensors
    )
    
    # Run falsification
    orchestrator.run_falsification(
        n_iterations=args.n_iterations,
        init_points=args.init_points,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()

