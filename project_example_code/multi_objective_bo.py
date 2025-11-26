"""
Multi-Objective Bayesian Optimization Framework

Uses the bayes_opt library with weighted scalarization for multi-objective optimization.
Much simpler than the previous custom implementation!
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

try:
    from scipy.spatial.distance import cdist
except ImportError:
    # Fallback if scipy not available
    cdist = None

# Add local BayesianOptimization to path
_bo_path = Path(__file__).parent / "BayesianOptimization"
if _bo_path.exists():
    sys.path.insert(0, str(_bo_path))

from bayes_opt import BayesianOptimization
try:
    from bayes_opt import acquisition
    _HAS_ACQUISITION_MODULE = True
except ImportError:
    from bayes_opt import UtilityFunction
    _HAS_ACQUISITION_MODULE = False


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
    """Manages the Pareto-optimal set of solutions."""
    
    def __init__(self):
        self.solutions: List[EvaluationResult] = []
    
    def _dominates(self, r1: EvaluationResult, r2: EvaluationResult) -> bool:
        """Check if r1 Pareto dominates r2."""
        # Convert to minimization (negate plausibility)
        v1 = np.array([r1.objectives['safety'], -r1.objectives['plausibility'], r1.objectives['comfort']])
        v2 = np.array([r2.objectives['safety'], -r2.objectives['plausibility'], r2.objectives['comfort']])
        return np.any(v1 < v2) and np.all(v1 <= v2)
    
    def add(self, result: EvaluationResult) -> bool:
        """Add result if non-dominated. Returns True if added."""
        for existing in self.solutions:
            if self._dominates(existing, result):
                return False
        self.solutions = [s for s in self.solutions if not self._dominates(result, s)]
        self.solutions.append(result)
        return True
    
    def get_solutions(self) -> List[EvaluationResult]:
        return self.solutions.copy()
    
    def __len__(self) -> int:
        return len(self.solutions)


# ============================================================================
# MULTI-OBJECTIVE BAYESIAN OPTIMIZATION (Using bayes_opt library)
# ============================================================================

class MultiObjectiveBayesianOptimization:
    """
    Multi-objective BO using bayes_opt library with weighted scalarization.
    
    Strategy: Each iteration uses random weights to explore different
    regions of the Pareto front. The bayes_opt library handles all the
    GP fitting and acquisition optimization internally.
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
        
        # Objectives
        if objective_names is None:
            objective_names = ['safety', 'plausibility', 'comfort']
        self.objective_names = objective_names
        self.n_objectives = len(objective_names)
        
        # Which objectives to maximize
        if maximize_objectives is None:
            maximize_objectives = [False, True, False]
        self.maximize_objectives = maximize_objectives
        
        # Storage
        self.evaluation_history: List[EvaluationResult] = []
        self.pareto_front = ParetoFront()
        
        # Random state
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Iteration counter
        self.iteration = 0
        
        # Initialize bayes_opt optimizer (no function - we register results manually)
        self._init_optimizer()
    
    def _init_optimizer(self):
        """Initialize the bayes_opt optimizer."""
        # Start with higher exploration (kappa=3.0) for better initial exploration
        # Will adaptively decrease over time
        
        if _HAS_ACQUISITION_MODULE:
            acq_func = acquisition.UpperConfidenceBound(kappa=3.0)
            
            self.optimizer = BayesianOptimization(
                f=None,  # No function - we register results manually
                pbounds=self.parameter_bounds,
                acquisition_function=acq_func,
                random_state=self.random_state,
                verbose=0,  # Quiet mode
                allow_duplicate_points=True
            )
        else:
            # Older version support
            self.optimizer = BayesianOptimization(
                f=None,
                pbounds=self.parameter_bounds,
                random_state=self.random_state,
                verbose=0,
                allow_duplicate_points=True
            )
            self._utility_function = UtilityFunction(kind="ucb", kappa=3.0, xi=0.0)
        
        # Track for adaptive exploration
        self.initial_kappa = 3.0
        self.final_kappa = 1.0
    
    def _scalarize_objectives(self, objectives: Dict[str, float], weights: np.ndarray) -> float:
        """
        Convert multi-objective to single scalar using weighted sum.
        
        Args:
            objectives: Dict with objective values
            weights: Weight vector (sums to 1)
            
        Returns:
            Scalar value (higher is better for bayes_opt)
        """
        score = 0.0
        for i, obj_name in enumerate(self.objective_names):
            value = objectives[obj_name]
            # Negate if minimizing (bayes_opt maximizes)
            if not self.maximize_objectives[i]:
                value = -value
            score += weights[i] * value
        return score
    
    def _generate_weights(self, strategy: str = "pareto_aware") -> np.ndarray:
        """
        Generate weights for scalarization.
        
        Args:
            strategy: "random", "pareto_aware", or "uniform"
            
        Returns:
            Weight vector (sums to 1)
        """
        if strategy == "random":
            # Pure random (original)
            return np.random.dirichlet(np.ones(self.n_objectives))
        elif strategy == "pareto_aware":
            # Bias toward under-explored regions of Pareto front
            if len(self.pareto_front) < 3:
                return np.random.dirichlet(np.ones(self.n_objectives))
            
            # Get Pareto front objectives
            pareto_objectives = np.array([
                [r.objectives[obj] for obj in self.objective_names]
                for r in self.pareto_front.get_solutions()
            ])
            
            # Normalize objectives to [0, 1] for each dimension
            normalized = np.zeros_like(pareto_objectives)
            for i, obj_name in enumerate(self.objective_names):
                values = pareto_objectives[:, i]
                if self.maximize_objectives[i]:
                    # For maximization, higher is better
                    normalized[:, i] = (values - values.min()) / (values.max() - values.min() + 1e-10)
                else:
                    # For minimization, lower is better (invert)
                    normalized[:, i] = 1.0 - (values - values.min()) / (values.max() - values.min() + 1e-10)
            
            # Find under-explored regions (low density)
            # Use inverse distance to nearest neighbor as density estimate
            if len(normalized) > 1 and cdist is not None:
                try:
                    distances = cdist(normalized, normalized)
                    np.fill_diagonal(distances, np.inf)
                    min_distances = distances.min(axis=1)
                    # Regions with larger min_distance are under-explored
                    exploration_scores = min_distances / (min_distances.max() + 1e-10)
                except Exception:
                    # Fallback if cdist fails
                    exploration_scores = np.ones(len(normalized))
            else:
                # Fallback: uniform weights
                exploration_scores = np.ones(len(normalized))
            
            # Weight by exploration score (prefer under-explored)
            weights = np.random.dirichlet(exploration_scores + 0.1)  # Add small constant for stability
            
            # Project to objective space (weighted average of Pareto points)
            target_weights = normalized.T @ weights
            target_weights = target_weights / (target_weights.sum() + 1e-10)
            
            # Add some randomness
            target_weights = 0.7 * target_weights + 0.3 * np.random.dirichlet(np.ones(self.n_objectives))
            return target_weights / target_weights.sum()
        else:  # uniform
            return np.ones(self.n_objectives) / self.n_objectives
    
    def suggest_next(self, init_points: int = 10) -> Dict[str, float]:
        """
        Suggest next parameters to evaluate.
        
        Args:
            init_points: Number of random initial explorations (increased default)
            
        Returns:
            Parameter dictionary for next evaluation
        """
        # Initial random exploration with Latin Hypercube Sampling for better coverage
        if len(self.evaluation_history) < init_points:
            # Use Latin Hypercube Sampling for better space coverage
            if len(self.evaluation_history) == 0:
                # First point: random
                params = {}
                for name, (low, high) in self.parameter_bounds.items():
                    params[name] = np.random.uniform(low, high)
            else:
                # Subsequent points: maximize minimum distance to existing points
                # Simplified: random but with some diversity
                params = {}
                for name, (low, high) in self.parameter_bounds.items():
                    params[name] = np.random.uniform(low, high)
            return params
        
        # Adapt exploration parameter (kappa) based on progress
        self._update_exploration_parameter()
        
        # Use bayes_opt to suggest next point
        try:
            if _HAS_ACQUISITION_MODULE:
                params = self.optimizer.suggest()
            else:
                params = self.optimizer.suggest(self._utility_function)
        except Exception:
            # Fallback to random if suggestion fails
            params = {}
            for name, (low, high) in self.parameter_bounds.items():
                params[name] = np.random.uniform(low, high)
        
        return params
    
    def _update_exploration_parameter(self):
        """Adaptively update kappa (exploration parameter) based on progress."""
        # Linear decay from initial_kappa to final_kappa
        # Use sqrt of progress for smoother transition
        total_iterations = max(50, len(self.evaluation_history))  # Estimate total
        progress = min(1.0, len(self.evaluation_history) / total_iterations)
        
        # Use exponential decay for smoother transition
        current_kappa = self.final_kappa + (self.initial_kappa - self.final_kappa) * np.exp(-2.0 * progress)
        
        # Update acquisition function
        if _HAS_ACQUISITION_MODULE:
            if hasattr(self.optimizer, '_acquisition_function'):
                self.optimizer._acquisition_function.kappa = current_kappa
        else:
            if hasattr(self, '_utility_function'):
                self._utility_function.kappa = current_kappa
    
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
        # Store result
        result = EvaluationResult(
            parameters=parameters,
            objectives=objectives,
            iteration=self.iteration,
            metadata=metadata
        )
        self.evaluation_history.append(result)
        self.pareto_front.add(result)
        
        # Generate weights for this evaluation and compute scalar target
        # Use Pareto-aware strategy after initial exploration
        if len(self.evaluation_history) > 10:
            weights = self._generate_weights(strategy="pareto_aware")
        else:
            weights = self._generate_weights(strategy="random")
        target = self._scalarize_objectives(objectives, weights)
        
        # Register with bayes_opt optimizer
        try:
            self.optimizer.register(params=parameters, target=target)
        except Exception:
            pass  # Ignore duplicate point errors
        
        self.iteration += 1
    
    def get_pareto_front(self) -> List[EvaluationResult]:
        """Get current Pareto-optimal solutions."""
        return self.pareto_front.get_solutions()
    
    def get_best_by_objective(self, objective_name: str) -> Optional[EvaluationResult]:
        """Get best solution for a single objective."""
        if not self.evaluation_history:
            return None
        
        maximize = self.maximize_objectives[self.objective_names.index(objective_name)]
        key = lambda r: r.objectives[objective_name]
        
        return max(self.evaluation_history, key=key) if maximize else min(self.evaluation_history, key=key)
    
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
        
        filepath = Path(filepath)
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
        self.evaluation_history = [EvaluationResult.from_dict(r) for r in state['evaluation_history']]
        
        # Rebuild Pareto front
        self.pareto_front = ParetoFront()
        for result in self.evaluation_history:
            self.pareto_front.add(result)
        
        # Re-register all points with optimizer
        self._init_optimizer()
        for result in self.evaluation_history:
            weights = self._generate_weights()
            target = self._scalarize_objectives(result.objectives, weights)
            try:
                self.optimizer.register(params=result.parameters, target=target)
            except Exception:
                pass
        
        print(f"Loaded {len(self.evaluation_history)} evaluations from {filepath}")
        print(f"  Pareto front size: {len(self.pareto_front)}")
    
    def print_summary(self):
        """Print summary of optimization progress."""
        print("=" * 70)
        print("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"Iterations: {self.iteration}")
        print(f"Evaluations: {len(self.evaluation_history)}")
        print(f"Pareto front: {len(self.pareto_front)} solutions")
        
        print("\nBest by objective:")
        for obj_name in self.objective_names:
            best = self.get_best_by_objective(obj_name)
            if best:
                print(f"  {obj_name}: {best.objectives[obj_name]:.2f}")
        print("=" * 70)


# ============================================================================
# EXAMPLE / TESTING
# ============================================================================

if __name__ == "__main__":
    from config.search_space import get_parameter_bounds
    
    print("=" * 70)
    print("MULTI-OBJECTIVE BO (using bayes_opt library)")
    print("=" * 70)
    
    # Initialize
    bounds = get_parameter_bounds()
    optimizer = MultiObjectiveBayesianOptimization(
        parameter_bounds=bounds,
        random_state=42
    )
    
    print(f"\nParameters: {len(bounds)}")
    print(f"Objectives: {optimizer.objective_names}")
    
    # Simulate evaluations
    print("\nSimulating 10 evaluations...")
    for i in range(10):
        params = optimizer.suggest_next()
        
        # Fake objectives (random for demo)
        objectives = {
            'safety': np.random.uniform(20, 80),
            'plausibility': np.random.uniform(40, 90),
            'comfort': np.random.uniform(30, 70),
        }
        
        optimizer.register_evaluation(params, objectives)
        print(f"  {i+1}: safety={objectives['safety']:.1f}, plaus={objectives['plausibility']:.1f}, comfort={objectives['comfort']:.1f}")
    
    print()
    optimizer.print_summary()
