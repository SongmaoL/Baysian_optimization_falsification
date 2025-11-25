"""
Multi-Objective Bayesian Optimization Framework

This module implements multi-objective Bayesian Optimization for the falsification
framework. It maintains separate Gaussian Process models for each objective and
uses a scalarization approach to balance exploration of the Pareto front.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, asdict
import pickle

try:
    from bayes_opt import BayesianOptimization
    from bayes_opt import UtilityFunction
    BAYESOPT_AVAILABLE = True
except ImportError:
    print("Warning: bayesian-optimization not installed. Install with: pip install bayesian-optimization")
    BAYESOPT_AVAILABLE = False

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EvaluationResult:
    """Stores results of a single evaluation."""
    parameters: Dict[str, float]
    objectives: Dict[str, float]  # safety, plausibility, comfort
    iteration: int
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


# ============================================================================
# PARETO FRONT MANAGEMENT
# ============================================================================

class ParetoFront:
    """
    Manages the Pareto-optimal set of solutions.
    
    For our objectives:
    - Safety: MINIMIZE (lower = more unsafe, which we want to find)
    - Plausibility: MAXIMIZE (higher = more realistic)
    - Comfort: MINIMIZE (lower = more uncomfortable, which we want to find)
    """
    
    def __init__(self):
        self.solutions: List[EvaluationResult] = []
    
    def _dominates(self, result1: EvaluationResult, result2: EvaluationResult) -> bool:
        """
        Check if result1 Pareto dominates result2.
        """
        s1 = result1.objectives
        s2 = result2.objectives
        
        # Convert to minimization (negate plausibility since we maximize it)
        v1 = np.array([s1['safety'], -s1['plausibility'], s1['comfort']])
        v2 = np.array([s2['safety'], -s2['plausibility'], s2['comfort']])
        
        # Dominates if better in at least one and not worse in any
        better_in_one = np.any(v1 < v2)
        not_worse_in_any = np.all(v1 <= v2)
        
        return better_in_one and not_worse_in_any
    
    def add(self, result: EvaluationResult) -> bool:
        """
        Add a result to Pareto front if non-dominated.
        
        Returns:
            True if result was added (is non-dominated)
        """
        # Check if any existing solution dominates this one
        for existing in self.solutions:
            if self._dominates(existing, result):
                return False  # Dominated, don't add
        
        # Remove any solutions dominated by this one
        self.solutions = [s for s in self.solutions if not self._dominates(result, s)]
        
        # Add the new solution
        self.solutions.append(result)
        return True
    
    def get_solutions(self) -> List[EvaluationResult]:
        """Get all Pareto-optimal solutions."""
        return self.solutions.copy()
    
    def __len__(self) -> int:
        return len(self.solutions)


# ============================================================================
# MULTI-OBJECTIVE BAYESIAN OPTIMIZATION
# ============================================================================

class MultiObjectiveBayesianOptimization:
    """
    Multi-objective Bayesian Optimization using weighted scalarization.
    
    Uses separate GP models for each objective and adaptively explores
    the Pareto front using different weight combinations.
    """
    
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 objective_names: List[str] = None,
                 maximize_objectives: List[bool] = None,
                 random_state: int = 42):
        """
        Initialize multi-objective BO.
        
        Args:
            parameter_bounds: Dict mapping parameter names to (min, max) tuples
            objective_names: Names of objectives (default: ['safety', 'plausibility', 'comfort'])
            maximize_objectives: Which objectives to maximize (default: [False, True, False])
            random_state: Random seed
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.n_params = len(self.parameter_names)
        
        # Objectives
        if objective_names is None:
            objective_names = ['safety', 'plausibility', 'comfort']
        self.objective_names = objective_names
        self.n_objectives = len(objective_names)
        
        # Which objectives to maximize (rest are minimized)
        if maximize_objectives is None:
            maximize_objectives = [False, True, False]  # [safety, plausibility, comfort]
        self.maximize_objectives = maximize_objectives
        
        # Storage
        self.evaluation_history: List[EvaluationResult] = []
        self.pareto_front = ParetoFront()
        
        # GP models - one per objective
        self.gp_models: Dict[str, Optional[GaussianProcessRegressor]] = {
            obj: None for obj in objective_names
        }
        
        # Random state
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Iteration counter
        self.iteration = 0
    
    def _normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for param_name in self.parameter_names:
            min_val, max_val = self.parameter_bounds[param_name]
            value = params[param_name]
            normalized.append((value - min_val) / (max_val - min_val))
        return np.array(normalized)
    
    def _denormalize_parameters(self, normalized: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] to original range."""
        params = {}
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_bounds[param_name]
            params[param_name] = normalized[i] * (max_val - min_val) + min_val
        return params
    
    def _fit_gp_models(self):
        """Fit GP models to current data with warm starting."""
        if len(self.evaluation_history) == 0:
            return
        
        # Prepare training data
        X = np.array([self._normalize_parameters(r.parameters) 
                     for r in self.evaluation_history])
        
        # Fit a GP for each objective
        for obj_name in self.objective_names:
            y = np.array([r.objectives[obj_name] for r in self.evaluation_history])
            
            # Warm start: reuse previous kernel if available
            if self.gp_models[obj_name] is not None:
                kernel = self.gp_models[obj_name].kernel_
                n_restarts = 3  # Fewer restarts when warm starting
            else:
                kernel = C(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=np.ones(self.n_params) * 0.1,
                    length_scale_bounds=(1e-2, 1e2),
                    nu=2.5
                )
                n_restarts = 5  # More restarts for initial fit
            
            # Create and fit GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                alpha=1e-6,
                normalize_y=True,
                random_state=self.random_state
            )
            gp.fit(X, y)
            
            self.gp_models[obj_name] = gp
    
    def _acquisition_function(self, 
                             X_candidate: np.ndarray, 
                             weights: np.ndarray,
                             exploration_param: float = 2.0) -> float:
        """
        Scalarized acquisition function using Upper Confidence Bound.
        
        Args:
            X_candidate: Normalized parameter vector
            weights: Weight vector for objectives
            exploration_param: Exploration parameter (higher = more exploration)
            
        Returns:
            Acquisition value (higher is better)
        """
        acquisition_value = 0.0
        
        for i, obj_name in enumerate(self.objective_names):
            gp = self.gp_models[obj_name]
            if gp is None:
                continue
            
            # Predict mean and std
            X_candidate_2d = X_candidate.reshape(1, -1)
            mean, std = gp.predict(X_candidate_2d, return_std=True)
            mean, std = mean[0], std[0]
            
            # UCB: mean + kappa * std (for maximization)
            # For minimization, use: mean - kappa * std
            if self.maximize_objectives[i]:
                ucb = mean + exploration_param * std
            else:
                ucb = mean - exploration_param * std
            
            # Weight and accumulate
            acquisition_value += weights[i] * ucb
        
        return acquisition_value
    
    def _generate_random_weights(self, focus_diversity: bool = True) -> np.ndarray:
        """
        Generate random weight vector for scalarization.
        
        Args:
            focus_diversity: If True, use more diverse weights
            
        Returns:
            Weight vector (sums to 1)
        """
        if focus_diversity:
            # Use Dirichlet distribution for more diverse weights
            weights = np.random.dirichlet(np.ones(self.n_objectives))
        else:
            # Uniform random weights
            weights = np.random.random(self.n_objectives)
            weights /= weights.sum()
        
        return weights
    
    def suggest_next(self, n_restarts: int = 10, init_points: int = 5) -> Dict[str, float]:
        """
        Suggest next parameter configuration to evaluate.
        
        Uses L-BFGS-B optimization of acquisition function (faster than random sampling).
        
        Args:
            n_restarts: Number of random restarts for L-BFGS-B
            init_points: Number of initial random explorations
            
        Returns:
            Parameter dictionary for next evaluation
        """
        from scipy.optimize import minimize
        
        # Initial random exploration phase
        if len(self.evaluation_history) < init_points:
            return self._denormalize_parameters(np.random.random(self.n_params))
        
        # Fit GP models
        self._fit_gp_models()
        
        # Generate random weight vector for this iteration
        weights = self._generate_random_weights(focus_diversity=True)
        
        # Multi-start L-BFGS-B optimization (much faster than random sampling)
        best_acquisition = -np.inf
        best_candidate = None
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.random(self.n_params)
            
            # Optimize acquisition function
            result = minimize(
                lambda x: -self._acquisition_function(x, weights),  # Negate for minimization
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * self.n_params,
                options={'maxiter': 50}
            )
            
            acq_value = -result.fun
            if acq_value > best_acquisition:
                best_acquisition = acq_value
                best_candidate = result.x
        
        return self._denormalize_parameters(best_candidate)
    
    def register_evaluation(self, 
                          parameters: Dict[str, float],
                          objectives: Dict[str, float],
                          metadata: Optional[Dict] = None):
        """
        Register a new evaluation result.
        
        Args:
            parameters: Parameter configuration that was evaluated
            objectives: Objective function values
            metadata: Optional additional information
        """
        result = EvaluationResult(
            parameters=parameters,
            objectives=objectives,
            iteration=self.iteration,
            metadata=metadata
        )
        
        self.evaluation_history.append(result)
        self.pareto_front.add(result)
        self.iteration += 1
    
    def get_pareto_front(self) -> List[EvaluationResult]:
        """Get current Pareto-optimal solutions."""
        return self.pareto_front.get_solutions()
    
    def get_best_by_objective(self, objective_name: str) -> EvaluationResult:
        """Get best solution for a single objective."""
        if len(self.evaluation_history) == 0:
            return None
        
        maximize = self.maximize_objectives[self.objective_names.index(objective_name)]
        
        if maximize:
            return max(self.evaluation_history, 
                      key=lambda r: r.objectives[objective_name])
        else:
            return min(self.evaluation_history,
                      key=lambda r: r.objectives[objective_name])
    
    def save_state(self, filepath: Path):
        """Save optimization state to file."""
        state = {
            'parameter_bounds': self.parameter_bounds,
            'objective_names': self.objective_names,
            'maximize_objectives': self.maximize_objectives,
            'evaluation_history': [r.to_dict() for r in self.evaluation_history],
            'iteration': self.iteration,
            'random_state': self.random_state,
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved optimization state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load optimization state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.parameter_bounds = state['parameter_bounds']
        self.objective_names = state['objective_names']
        self.maximize_objectives = state['maximize_objectives']
        self.iteration = state['iteration']
        self.random_state = state['random_state']
        
        # Reconstruct evaluation history
        self.evaluation_history = [
            EvaluationResult.from_dict(r) for r in state['evaluation_history']
        ]
        
        # Rebuild Pareto front
        self.pareto_front = ParetoFront()
        for result in self.evaluation_history:
            self.pareto_front.add(result)
        
        print(f"Loaded optimization state from {filepath}")
        print(f"  Evaluations: {len(self.evaluation_history)}")
        print(f"  Pareto front size: {len(self.pareto_front)}")
    
    def print_summary(self):
        """Print summary of optimization progress."""
        print("=" * 80)
        print("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"Iterations: {self.iteration}")
        print(f"Total evaluations: {len(self.evaluation_history)}")
        print(f"Pareto front size: {len(self.pareto_front)}")
        
        print("\nBest by each objective:")
        for obj_name in self.objective_names:
            best = self.get_best_by_objective(obj_name)
            if best:
                print(f"  {obj_name}: {best.objectives[obj_name]:.4f}")
        
        print("=" * 80)


# ============================================================================
# EXAMPLE / TESTING
# ============================================================================

if __name__ == "__main__":
    from config.search_space import get_parameter_bounds
    
    print("=" * 80)
    print("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION")
    print("=" * 80)
    
    # Initialize optimizer
    bounds = get_parameter_bounds()
    optimizer = MultiObjectiveBayesianOptimization(
        parameter_bounds=bounds,
        random_state=42
    )
    
    print(f"\nInitialized with {len(bounds)} parameters")
    print(f"Objectives: {optimizer.objective_names}")
    
    # Simulate a few evaluations
    print("\nSimulating evaluations...")
    for i in range(5):
        # Get next parameters to evaluate
        params = optimizer.suggest_next()
        
        # Simulate objective evaluations (random values for demo)
        objectives = {
            'safety': np.random.uniform(20, 80),
            'plausibility': np.random.uniform(40, 90),
            'comfort': np.random.uniform(30, 70),
        }
        
        # Register result
        optimizer.register_evaluation(params, objectives)
        print(f"  Iteration {i+1}: safety={objectives['safety']:.2f}, "
              f"plausibility={objectives['plausibility']:.2f}, "
              f"comfort={objectives['comfort']:.2f}")
    
    # Print summary
    print()
    optimizer.print_summary()

