import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('final_results.json', 'r') as f:
    data = json.load(f)

# Extract data
iterations = [e['iteration'] for e in data['evaluation_history']]
safety = [e['objectives']['safety'] for e in data['evaluation_history']]
plausibility = [e['objectives']['plausibility'] for e in data['evaluation_history']]
comfort = [e['objectives']['comfort'] for e in data['evaluation_history']]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bayesian Optimization Results - 107 Iterations', fontsize=16, fontweight='bold')

# 1. Objective convergence over iterations
ax1 = axes[0, 0]
ax1.plot(iterations, safety, 'o-', label='Safety (minimize)', alpha=0.6, markersize=3)
ax1.plot(iterations, plausibility, 's-', label='Plausibility (maximize)', alpha=0.6, markersize=3)
ax1.plot(iterations, comfort, '^-', label='Comfort (minimize)', alpha=0.6, markersize=3)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Objective Score')
ax1.set_title('Objective Convergence Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 2D Pareto front projection (Safety vs Plausibility)
ax2 = axes[0, 1]
scatter1 = ax2.scatter(safety, plausibility, c=comfort, cmap='viridis', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Safety Score (minimize →)')
ax2.set_ylabel('Plausibility Score (← maximize)')
ax2.set_title('Safety vs. Plausibility (color = Comfort)')
ax2.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax2)
cbar1.set_label('Comfort Score')

# Add best/worst labels
best_safety_idx = safety.index(min(safety))
worst_safety_idx = safety.index(max(safety))
ax2.scatter(safety[best_safety_idx], plausibility[best_safety_idx], 
           color='green', s=200, marker='*', edgecolors='black', linewidth=2, 
           label=f'Best (iter {iterations[best_safety_idx]})', zorder=5)
ax2.scatter(safety[worst_safety_idx], plausibility[worst_safety_idx], 
           color='red', s=200, marker='X', edgecolors='black', linewidth=2,
           label=f'Worst (iter {iterations[worst_safety_idx]})', zorder=5)
ax2.legend()

# 3. Safety vs Comfort
ax3 = axes[1, 0]
scatter2 = ax3.scatter(safety, comfort, c=plausibility, cmap='plasma',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Safety Score (minimize →)')
ax3.set_ylabel('Comfort Score (minimize →)')
ax3.set_title('Safety vs. Comfort (color = Plausibility)')
ax3.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax3)
cbar2.set_label('Plausibility Score')

# 4. Distribution histograms
ax4 = axes[1, 1]
ax4.hist(safety, bins=20, alpha=0.5, label='Safety', color='red', edgecolor='black')
ax4.axvline(np.mean(safety), color='red', linestyle='--', linewidth=2, label=f'Safety mean={np.mean(safety):.1f}')
ax4.set_xlabel('Score')
ax4.set_ylabel('Frequency')
ax4.set_title('Objective Score Distributions')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add text box with statistics
stats_text = f"""Statistics:
Safety: {min(safety):.1f} - {max(safety):.1f} (μ={np.mean(safety):.1f})
Plausibility: {min(plausibility):.1f} - {max(plausibility):.1f} (μ={np.mean(plausibility):.1f})
Comfort: {min(comfort):.1f} - {max(comfort):.1f} (μ={np.mean(comfort):.1f})

Non-zero values:
Plausibility: {sum(1 for p in plausibility if p > 0)}/107 ({sum(1 for p in plausibility if p > 0)/107*100:.0f}%)
Comfort: {sum(1 for c in comfort if c > 0)}/107 ({sum(1 for c in comfort if c > 0)/107*100:.0f}%)"""

ax4.text(0.98, 0.5, stats_text, transform=ax4.transAxes,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         fontfamily='monospace', fontsize=8)

plt.tight_layout()
plt.savefig('optimization_results_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] Saved visualization to: optimization_results_visualization.png")

# Create second figure: 3D scatter plot
fig2 = plt.figure(figsize=(12, 9))
ax = fig2.add_subplot(111, projection='3d')

# Create 3D scatter
scatter = ax.scatter(safety, plausibility, comfort, 
                    c=iterations, cmap='coolwarm', 
                    s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Highlight best and worst
ax.scatter(safety[best_safety_idx], plausibility[best_safety_idx], comfort[best_safety_idx],
          color='green', s=300, marker='*', edgecolors='black', linewidth=2, label='Best Safety')
ax.scatter(safety[worst_safety_idx], plausibility[worst_safety_idx], comfort[worst_safety_idx],
          color='red', s=300, marker='X', edgecolors='black', linewidth=2, label='Worst Safety')

ax.set_xlabel('Safety (minimize →)', fontsize=10, labelpad=10)
ax.set_ylabel('Plausibility (← maximize)', fontsize=10, labelpad=10)
ax.set_zlabel('Comfort (minimize →)', fontsize=10, labelpad=10)
ax.set_title('3D Pareto Front - Multi-Objective Optimization\n(Color = Iteration Number)', 
             fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('Iteration', fontsize=10)
ax.legend(loc='upper left', fontsize=10)

# Adjust viewing angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('optimization_results_3d.png', dpi=300, bbox_inches='tight')
print("[OK] Saved 3D visualization to: optimization_results_3d.png")

# Create third figure: Parameter exploration heatmaps (sample)
fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
fig3.suptitle('Parameter Space Exploration (Sample Parameters)', fontsize=14, fontweight='bold')

# Extract some key parameters
param_names = ['fog_density', 'precipitation', 'lead_base_throttle', 
               'lead_brake_probability', 'initial_distance', 'initial_ego_velocity']
param_values = {name: [e['parameters'][name] for e in data['evaluation_history']] 
                for name in param_names}

for idx, (ax, param_name) in enumerate(zip(axes.flat, param_names)):
    values = param_values[param_name]
    ax.scatter(iterations, values, c=safety, cmap='RdYlGn_r', 
              s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(param_name.replace('_', ' ').title())
    ax.set_title(f'{param_name.replace("_", " ").title()} Over Time')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_exploration.png', dpi=300, bbox_inches='tight')
print("[OK] Saved parameter exploration to: parameter_exploration.png")

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
print("Generated files:")
print("  1. optimization_results_visualization.png - 2x2 grid of key plots")
print("  2. optimization_results_3d.png - 3D Pareto front")
print("  3. parameter_exploration.png - Parameter space exploration")

