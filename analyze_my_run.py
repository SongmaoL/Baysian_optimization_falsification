import json
import os
import math

# Configuration
RESULTS_DIR = 'my_falsification_run_11-25'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'results/final_results.json')

def calculate_mean(data):
    return sum(data) / len(data)

def analyze_results():
    print(f"Analyzing results from: {RESULTS_FILE}")
    
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: File not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    history = data['evaluation_history']
    iterations = [e['iteration'] for e in history]
    
    # Extract objectives
    safety = [e['objectives']['safety'] for e in history]
    plausibility = [e['objectives']['plausibility'] for e in history]
    comfort = [e['objectives']['comfort'] for e in history]
    
    # Combine into a list of dictionaries for easy sorting
    scenarios = []
    for i, e in enumerate(history):
        scenarios.append({
            'iteration': e['iteration'],
            'safety': e['objectives']['safety'],
            'plausibility': e['objectives']['plausibility'],
            'comfort': e['objectives']['comfort'],
            'parameters': e['parameters']
        })

    # ========================================================================
    # 1. TEXT SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Iterations: {len(iterations)}")
    print(f"\nSafety Score (Minimize -> 0 is worst/unsafe):")
    print(f"  Min: {min(safety):.2f} (Most unsafe)")
    print(f"  Max: {max(safety):.2f}")
    print(f"  Mean: {calculate_mean(safety):.2f}")
    
    print(f"\nPlausibility Score (Maximize -> 100 is best/realistic):")
    print(f"  Min: {min(plausibility):.2f}")
    print(f"  Max: {max(plausibility):.2f} (Most plausible)")
    print(f"  Mean: {calculate_mean(plausibility):.2f}")

    print(f"\nComfort Score (Minimize -> 0 is worst/uncomfortable):")
    print(f"  Min: {min(comfort):.2f} (Most uncomfortable)")
    print(f"  Max: {max(comfort):.2f}")
    print(f"  Mean: {calculate_mean(comfort):.2f}")

    # Identify Critical Scenarios
    # Definition: Safety < 15 AND Plausibility > 0 (at least somewhat physically valid)
    critical = [s for s in scenarios if s['safety'] < 15 and s['plausibility'] > 0]
    critical.sort(key=lambda x: x['safety'])
    
    print("\n" + "="*60)
    print("CRITICAL SCENARIOS (Safety < 15 & Plausibility > 0)")
    print("="*60)
    
    if len(critical) > 0:
        print(f"Found {len(critical)} critical scenarios:\n")
        print(f"{'Iter':<5} {'Safety':<10} {'Plausibility':<15} {'Comfort':<10} {'Description'}")
        print("-" * 70)
        
        for s in critical[:10]:  # Show top 10
            desc = f"Dist={s['parameters']['initial_distance']:.1f}, EgoV={s['parameters']['initial_ego_velocity']:.1f}"
            print(f"{s['iteration']:<5} {s['safety']:<10.2f} {s['plausibility']:<15.2f} {s['comfort']:<10.2f} {desc}")
            
        best_scenario = critical[0]
        print(f"\nMost critical scenario (Iter {best_scenario['iteration']}):")
        print(f"  Safety: {best_scenario['safety']:.2f}")
        print(f"  Plausibility: {best_scenario['plausibility']:.2f}")
        print(f"  Comfort: {best_scenario['comfort']:.2f}")
        print("  Key Parameters:")
        for p in ['fog_density', 'precipitation', 'lead_brake_intensity', 'initial_distance', 'initial_ego_velocity']:
            val = best_scenario['parameters'].get(p, 'N/A')
            if isinstance(val, float):
                print(f"    {p}: {val:.2f}")
            else:
                print(f"    {p}: {val}")
    else:
        print("No critical scenarios found matching criteria.")
        
    # Also find most plausible scenario
    most_plausible = sorted(scenarios, key=lambda x: x['plausibility'], reverse=True)[0]
    print(f"\nMost plausible scenario (Iter {most_plausible['iteration']}):")
    print(f"  Safety: {most_plausible['safety']:.2f}")
    print(f"  Plausibility: {most_plausible['plausibility']:.2f}")

    # ========================================================================
    # 3. PARETO FRONT ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("PARETO FRONT SOLUTIONS")
    print("="*60)
    print("These are solutions that represent optimal trade-offs.")
    print("No other solution is strictly 'better' in all objectives.")
    print("(Better = Lower Safety, Higher Plausibility, Lower Comfort)\n")

    pareto_front = []
    
    for i, cand in enumerate(scenarios):
        is_dominated = False
        c_safe = cand['safety']
        c_plaus = cand['plausibility']
        c_comf = cand['comfort']
        
        for j, other in enumerate(scenarios):
            if i == j:
                continue
                
            o_safe = other['safety']
            o_plaus = other['plausibility']
            o_comf = other['comfort']
            
            # Check if 'other' dominates 'cand'
            # Conditions for dominance (finding failures):
            # 1. other_safety <= cand_safety (More or equal unsafe)
            # 2. other_plausibility >= cand_plausibility (More or equal plausible)
            # 3. other_comfort <= cand_comfort (More or equal uncomfortable)
            # AND at least one is strictly better
            
            better_eq_safe = o_safe <= c_safe
            better_eq_plaus = o_plaus >= c_plaus
            better_eq_comf = o_comf <= c_comf
            
            strictly_better = (o_safe < c_safe) or (o_plaus > c_plaus) or (o_comf < c_comf)
            
            if better_eq_safe and better_eq_plaus and better_eq_comf and strictly_better:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(cand)

    # Sort by Safety for easier reading
    pareto_front.sort(key=lambda x: x['safety'])
    
    print(f"{'Iter':<5} {'Safety':<10} {'Plausibility':<15} {'Comfort':<10} {'Type'}")
    print("-" * 70)
    
    for s in pareto_front:
        # Classify the type of result
        if s['safety'] < 15 and s['plausibility'] > 30:
            res_type = "CRITICAL (Valid Failure)"
        elif s['safety'] < 10 and s['plausibility'] == 0:
            res_type = "Artifact (Invalid)"
        elif s['plausibility'] == 50:
            res_type = "Realistic"
        else:
            res_type = "Trade-off"
            
        print(f"{s['iteration']:<5} {s['safety']:<10.2f} {s['plausibility']:<15.2f} {s['comfort']:<10.2f} {res_type}")

if __name__ == "__main__":
    analyze_results()
