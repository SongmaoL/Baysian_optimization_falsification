"""
Pareto Front Analysis and Visualization Tools

This module provides functions to analyze and visualize the results
of multi-objective falsification, including Pareto front visualization,
convergence plots, and scenario selection.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports when running as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json

from multi_objective_bo import MultiObjectiveBayesianOptimization, EvaluationResult


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_results(results_path: Path) -> MultiObjectiveBayesianOptimization:
    """
    Load optimization results from file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        MultiObjectiveBayesianOptimization object with loaded state
    """
    from config.search_space import get_parameter_bounds
    
    optimizer = MultiObjectiveBayesianOptimization(
        parameter_bounds=get_parameter_bounds()
    )
    optimizer.load_state(results_path)
    
    return optimizer


def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    Convert evaluation results to pandas DataFrame.
    
    Args:
        results: List of evaluation results
        
    Returns:
        DataFrame with all results
    """
    data = []
    for result in results:
        row = {
            'iteration': result.iteration,
            'safety': result.objectives['safety'],
            'plausibility': result.objectives['plausibility'],
        }
        # Include comfort if present (backward compatibility)
        if 'comfort' in result.objectives:
            row['comfort'] = result.objectives['comfort']
        
        # Add parameters
        for key, value in result.parameters.items():
            row[f'param_{key}'] = value
        
        data.append(row)
    
    return pd.DataFrame(data)


# ============================================================================
# PARETO FRONT VISUALIZATION
# ============================================================================

def plot_pareto_front_3d(optimizer: MultiObjectiveBayesianOptimization,
                        save_path: Optional[Path] = None,
                        show: bool = True):
    """
    Plot Pareto front (2D now that comfort is removed).
    
    Args:
        optimizer: Optimization object with results
        save_path: Path to save figure
        show: Whether to display figure
    """
    # Get all results and Pareto front
    all_results = optimizer.evaluation_history
    pareto_results = optimizer.get_pareto_front()
    
    # Extract objective values
    all_safety = [r.objectives['safety'] for r in all_results]
    all_plausibility = [r.objectives['plausibility'] for r in all_results]
    
    pareto_safety = [r.objectives['safety'] for r in pareto_results]
    pareto_plausibility = [r.objectives['plausibility'] for r in pareto_results]
    
    # Check if we have comfort data (backward compatibility)
    has_comfort = all_results and 'comfort' in all_results[0].objectives
    
    if has_comfort:
        # Legacy 3D plot for old results with comfort
        all_comfort = [r.objectives['comfort'] for r in all_results]
        pareto_comfort = [r.objectives['comfort'] for r in pareto_results]
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(all_safety, all_plausibility, all_comfort,
                  c='lightgray', alpha=0.3, s=30, label='All evaluations')
        ax.scatter(pareto_safety, pareto_plausibility, pareto_comfort,
                  c='red', alpha=0.8, s=100, marker='*', label='Pareto front')
        
        ax.set_xlabel('Safety Score (minimize)', fontsize=12)
        ax.set_ylabel('Plausibility Score (maximize)', fontsize=12)
        ax.set_zlabel('Comfort Score (minimize)', fontsize=12)
    else:
        # New 2D plot for results without comfort
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(all_safety, all_plausibility,
                  c='lightgray', alpha=0.3, s=30, label='All evaluations')
        ax.scatter(pareto_safety, pareto_plausibility,
                  c='red', alpha=0.8, s=100, marker='*', label='Pareto front')
        
        ax.set_xlabel('Safety Score (minimize)', fontsize=12)
        ax.set_ylabel('Plausibility Score (maximize)', fontsize=12)
    ax.set_zlabel('Comfort Score (minimize)', fontsize=12)
    ax.set_title('3D Pareto Front Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D Pareto plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_front_2d_projections(optimizer: MultiObjectiveBayesianOptimization,
                                     save_path: Optional[Path] = None,
                                     show: bool = True):
    """
    Plot 2D projections of Pareto front (pairwise comparisons).
    
    Args:
        optimizer: Optimization object with results
        save_path: Path to save figure
        show: Whether to display figure
    """
    all_results = optimizer.evaluation_history
    pareto_results = optimizer.get_pareto_front()
    
    # Extract values
    all_df = results_to_dataframe(all_results)
    pareto_df = results_to_dataframe(pareto_results)
    
    # Create subplots for pairwise comparisons (comfort removed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pairs = [
        ('safety', 'plausibility', 'Safety vs Plausibility'),
        ('safety', 'plausibility', 'Safety vs Plausibility (zoomed)'),
    ]
    
    for idx, (x_obj, y_obj, title) in enumerate(pairs):
        ax = axes[idx]
        
        # Plot all points
        ax.scatter(all_df[x_obj], all_df[y_obj],
                  c='lightgray', alpha=0.3, s=30, label='All evaluations')
        
        # Plot Pareto front
        ax.scatter(pareto_df[x_obj], pareto_df[y_obj],
                  c='red', alpha=0.8, s=100, marker='*', label='Pareto front')
        
        ax.set_xlabel(x_obj.capitalize(), fontsize=11)
        ax.set_ylabel(y_obj.capitalize(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Zoom into Pareto region for second row
        if idx >= 3:
            if len(pareto_df) > 0:
                x_margin = (pareto_df[x_obj].max() - pareto_df[x_obj].min()) * 0.2
                y_margin = (pareto_df[y_obj].max() - pareto_df[y_obj].min()) * 0.2
                ax.set_xlim(pareto_df[x_obj].min() - x_margin, 
                           pareto_df[x_obj].max() + x_margin)
                ax.set_ylim(pareto_df[y_obj].min() - y_margin,
                           pareto_df[y_obj].max() + y_margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D projections to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def plot_convergence(optimizer: MultiObjectiveBayesianOptimization,
                    save_path: Optional[Path] = None,
                    show: bool = True):
    """
    Plot convergence of each objective over iterations.
    
    Args:
        optimizer: Optimization object with results
        save_path: Path to save figure
        show: Whether to display figure
    """
    df = results_to_dataframe(optimizer.evaluation_history)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    objectives = ['safety', 'plausibility']  # comfort removed
    colors = ['blue', 'green', 'orange']
    
    for ax, obj, color in zip(axes, objectives, colors):
        # Plot all evaluations
        ax.scatter(df['iteration'], df[obj], alpha=0.3, s=20, c='lightgray', label='Evaluations')
        ax.plot(df['iteration'], df[obj], alpha=0.2, c='gray', linewidth=0.5)
        
        # Plot running best
        if obj == 'plausibility':  # Maximize
            running_best = df[obj].cummax()
            best_label = 'Running Maximum'
        else:  # Minimize
            running_best = df[obj].cummin()
            best_label = 'Running Minimum'
        
        ax.plot(df['iteration'], running_best, color=color, linewidth=2, label=best_label)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(f'{obj.capitalize()} Score', fontsize=11)
        ax.set_title(f'{obj.capitalize()} Convergence', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# PARAMETER ANALYSIS
# ============================================================================

def plot_parameter_distributions(optimizer: MultiObjectiveBayesianOptimization,
                                 save_path: Optional[Path] = None,
                                 show: bool = True):
    """
    Plot distributions of parameters in Pareto front.
    
    Args:
        optimizer: Optimization object with results
        save_path: Path to save figure
        show: Whether to display figure
    """
    pareto_df = results_to_dataframe(optimizer.get_pareto_front())
    
    # Get parameter columns
    param_cols = [col for col in pareto_df.columns if col.startswith('param_')]
    
    # Select top parameters to visualize (limit to 12)
    n_params = min(len(param_cols), 12)
    param_cols = param_cols[:n_params]
    
    # Create subplots
    n_rows = (n_params + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param_col in enumerate(param_cols):
        ax = axes[idx]
        param_name = param_col.replace('param_', '')
        
        # Histogram
        ax.hist(pareto_df[param_col], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{param_name} Distribution in Pareto Front', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(param_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter distributions to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# SCENARIO SELECTION
# ============================================================================

def select_critical_scenarios(optimizer: MultiObjectiveBayesianOptimization,
                             n_scenarios: int = 10) -> List[EvaluationResult]:
    """
    Select most critical scenarios from Pareto front.
    
    Selects diverse scenarios covering different trade-offs:
    - Lowest safety (most unsafe)
    - Highest plausibility (most realistic)
    - Best trade-off (low safety + high plausibility)
    - Knee points (balanced trade-offs)
    
    Args:
        optimizer: Optimization object with results
        n_scenarios: Number of scenarios to select
        
    Returns:
        List of selected critical scenarios
    """
    pareto_results = optimizer.get_pareto_front()
    
    if len(pareto_results) == 0:
        return []
    
    if len(pareto_results) <= n_scenarios:
        return pareto_results
    
    selected = []
    
    # 1. Lowest safety (most unsafe)
    min_safety = min(pareto_results, key=lambda r: r.objectives['safety'])
    selected.append(min_safety)
    
    # 2. Highest plausibility (most realistic)
    max_plausibility = max(pareto_results, key=lambda r: r.objectives['plausibility'])
    if max_plausibility not in selected:
        selected.append(max_plausibility)
    
    # 3. Best trade-off: low safety + high plausibility
    # (comfort removed from optimization)
    best_tradeoff = min(pareto_results, 
                        key=lambda r: r.objectives['safety'] - r.objectives['plausibility'])
    if best_tradeoff not in selected:
        selected.append(best_tradeoff)
    
    # 4. Fill remaining with diverse samples
    remaining = n_scenarios - len(selected)
    if remaining > 0:
        # Remove already selected
        candidates = [r for r in pareto_results if r not in selected]
        
        # Sample uniformly across Pareto front
        if len(candidates) <= remaining:
            selected.extend(candidates)
        else:
            # Sort by safety and sample uniformly
            candidates.sort(key=lambda r: r.objectives['safety'])
            step = len(candidates) / remaining
            for i in range(remaining):
                idx = int(i * step)
                selected.append(candidates[idx])
    
    return selected


def save_critical_scenarios_report(optimizer: MultiObjectiveBayesianOptimization,
                                   output_path: Path,
                                   n_scenarios: int = 20):
    """
    Generate a report of critical scenarios.
    
    Args:
        optimizer: Optimization object with results
        output_path: Path to save report JSON
        n_scenarios: Number of scenarios to include
    """
    critical = select_critical_scenarios(optimizer, n_scenarios)
    
    report = {
        'summary': {
            'total_evaluations': len(optimizer.evaluation_history),
            'pareto_front_size': len(optimizer.get_pareto_front()),
            'n_critical_scenarios': len(critical),
        },
        'critical_scenarios': []
    }
    
    for i, result in enumerate(critical):
        scenario_info = {
            'rank': i + 1,
            'iteration': result.iteration,
            'objectives': result.objectives,
            'parameters': result.parameters,
            'metadata': result.metadata,
            'notes': []
        }
        
        # Add notes about why this scenario is critical
        if result.objectives['safety'] < 30:
            scenario_info['notes'].append('Very unsafe (low safety score)')
        if result.objectives['plausibility'] > 80:
            scenario_info['notes'].append('Highly realistic (high plausibility)')
        # Note: comfort metric removed from optimization
        
        report['critical_scenarios'].append(scenario_info)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved critical scenarios report to {output_path}")
    return report


# ============================================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis(results_path: Path, output_dir: Path):
    """
    Run complete analysis pipeline and generate all plots.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save analysis outputs
    """
    print("=" * 80)
    print("PARETO FRONT ANALYSIS")
    print("=" * 80)
    
    # Load results
    print(f"\nLoading results from {results_path}...")
    optimizer = load_results(results_path)
    optimizer.print_summary()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    plot_pareto_front_3d(
        optimizer,
        save_path=output_dir / "pareto_front_3d.png",
        show=False
    )
    
    plot_pareto_front_2d_projections(
        optimizer,
        save_path=output_dir / "pareto_front_2d.png",
        show=False
    )
    
    plot_convergence(
        optimizer,
        save_path=output_dir / "convergence.png",
        show=False
    )
    
    plot_parameter_distributions(
        optimizer,
        save_path=output_dir / "parameter_distributions.png",
        show=False
    )
    
    # Generate critical scenarios report
    print("\nGenerating critical scenarios report...")
    save_critical_scenarios_report(
        optimizer,
        output_path=output_dir / "critical_scenarios.json",
        n_scenarios=20
    )
    
    print(f"\nAll analysis outputs saved to {output_dir}")
    print("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze falsification results")
    parser.add_argument("results_file", type=Path, help="Path to results JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"),
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    run_complete_analysis(args.results_file, args.output_dir)

