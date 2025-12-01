"""
Three-Way Comparison: Single-Objective BO vs Multi-Objective BO vs Random Search
"""

import json
import math

def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_objectives(data):
    """Extract objectives into a list of dictionaries."""
    history = data['evaluation_history']
    objectives = []
    for eval_result in history:
        obj = {
            'iteration': eval_result['iteration'],
            'safety': eval_result['objectives']['safety'],
            'plausibility': eval_result['objectives']['plausibility'],
            'comfort': eval_result['objectives']['comfort'],
        }
        objectives.append(obj)
    return objectives

def calculate_statistics(objectives):
    """Calculate basic statistics."""
    safety = [o['safety'] for o in objectives]
    plausibility = [o['plausibility'] for o in objectives]
    comfort = [o['comfort'] for o in objectives]
    
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0
    
    def std(vals, m):
        if len(vals) < 2:
            return 0
        variance = sum((x - m) ** 2 for x in vals) / len(vals)
        return math.sqrt(variance)
    
    return {
        'count': len(objectives),
        'safety': {'mean': mean(safety), 'std': std(safety, mean(safety)), 'min': min(safety) if safety else 0, 'max': max(safety) if safety else 0},
        'plausibility': {'mean': mean(plausibility), 'std': std(plausibility, mean(plausibility)), 'min': min(plausibility) if plausibility else 0, 'max': max(plausibility) if plausibility else 0},
        'comfort': {'mean': mean(comfort), 'std': std(comfort, mean(comfort)), 'min': min(comfort) if comfort else 0, 'max': max(comfort) if comfort else 0},
    }

def find_critical_scenarios(objectives, safety_threshold=20.0, plausibility_threshold=50.0):
    """Find critical scenarios (low safety, high plausibility)."""
    critical = []
    for obj in objectives:
        if obj['safety'] < safety_threshold and obj['plausibility'] > plausibility_threshold:
            critical.append(obj)
    return sorted(critical, key=lambda x: x['safety'])

def calculate_pareto_front(objectives):
    """Calculate Pareto front."""
    pareto = []
    for i, obj_i in enumerate(objectives):
        is_dominated = False
        for j, obj_j in enumerate(objectives):
            if i == j:
                continue
            better_safety = obj_j['safety'] <= obj_i['safety']
            better_plaus = obj_j['plausibility'] >= obj_i['plausibility']
            better_comfort = obj_j['comfort'] <= obj_i['comfort']
            strictly_better = (obj_j['safety'] < obj_i['safety'] or 
                              obj_j['plausibility'] > obj_i['plausibility'] or 
                              obj_j['comfort'] < obj_i['comfort'])
            if better_safety and better_plaus and better_comfort and strictly_better:
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(obj_i)
    return pareto

def analyze_method(data, method_name):
    """Analyze a single method's results."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {method_name}")
    print(f"{'='*80}")
    
    objectives = extract_objectives(data)
    stats = calculate_statistics(objectives)
    
    print(f"\nTotal Iterations: {stats['count']}")
    print(f"\nObjective Statistics:")
    print(f"  Safety (minimize): Mean={stats['safety']['mean']:.2f}±{stats['safety']['std']:.2f}, Range=[{stats['safety']['min']:.2f}, {stats['safety']['max']:.2f}]")
    print(f"  Plausibility (maximize): Mean={stats['plausibility']['mean']:.2f}±{stats['plausibility']['std']:.2f}, Range=[{stats['plausibility']['min']:.2f}, {stats['plausibility']['max']:.2f}]")
    print(f"  Comfort (minimize): Mean={stats['comfort']['mean']:.2f}±{stats['comfort']['std']:.2f}, Range=[{stats['comfort']['min']:.2f}, {stats['comfort']['max']:.2f}]")
    
    best_safety = min(objectives, key=lambda x: x['safety'])
    best_plausibility = max(objectives, key=lambda x: x['plausibility'])
    best_comfort = min(objectives, key=lambda x: x['comfort'])
    
    print(f"\nBest Scenarios:")
    print(f"  Worst Safety: Safety={best_safety['safety']:.2f}, Plaus={best_safety['plausibility']:.2f}, Comfort={best_safety['comfort']:.2f} (Iter {best_safety['iteration']})")
    print(f"  Best Plausibility: Safety={best_plausibility['safety']:.2f}, Plaus={best_plausibility['plausibility']:.2f}, Comfort={best_plausibility['comfort']:.2f} (Iter {best_plausibility['iteration']})")
    print(f"  Worst Comfort: Safety={best_comfort['safety']:.2f}, Plaus={best_comfort['plausibility']:.2f}, Comfort={best_comfort['comfort']:.2f} (Iter {best_comfort['iteration']})")
    
    critical = find_critical_scenarios(objectives)
    print(f"\nCritical Scenarios (Safety < 20, Plausibility > 50): {len(critical)}")
    if len(critical) > 0:
        best_critical = critical[0]
        print(f"  Best Critical: Safety={best_critical['safety']:.2f}, Plausibility={best_critical['plausibility']:.2f}, Comfort={best_critical['comfort']:.2f} (Iter {best_critical['iteration']})")
    
    pareto = calculate_pareto_front(objectives)
    print(f"Pareto Front: {len(pareto)} solutions ({len(pareto)/len(objectives)*100:.1f}% of all)")
    
    return {
        'objectives': objectives,
        'stats': stats,
        'critical': critical,
        'pareto': pareto,
        'best_safety': best_safety,
        'best_plausibility': best_plausibility,
        'best_comfort': best_comfort,
    }

def compare_three_methods(single_analysis, multi_analysis, random_analysis):
    """Compare all three methods."""
    print(f"\n{'='*80}")
    print("THREE-WAY COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<45} {'Single-Obj BO':<20} {'Multi-Obj BO':<20} {'Random Search':<20} {'Winner':<15}")
    print("-"*120)
    
    # Iterations
    print(f"{'Total Iterations':<45} {single_analysis['stats']['count']:<20} {multi_analysis['stats']['count']:<20} {random_analysis['stats']['count']:<20} {'-':<15}")
    
    # Best Safety
    single_safety = single_analysis['best_safety']['safety']
    multi_safety = multi_analysis['best_safety']['safety']
    random_safety = random_analysis['best_safety']['safety']
    best_safety_val = min(single_safety, multi_safety, random_safety)
    winner_safety = "Single" if single_safety == best_safety_val else ("Multi" if multi_safety == best_safety_val else "Random")
    print(f"{'Best Safety (Lower = Worse)':<45} {single_safety:<20.2f} {multi_safety:<20.2f} {random_safety:<20.2f} {winner_safety:<15}")
    
    # Best Plausibility
    single_plaus = single_analysis['best_plausibility']['plausibility']
    multi_plaus = multi_analysis['best_plausibility']['plausibility']
    random_plaus = random_analysis['best_plausibility']['plausibility']
    best_plaus_val = max(single_plaus, multi_plaus, random_plaus)
    winner_plaus = "Single" if single_plaus == best_plaus_val else ("Multi" if multi_plaus == best_plaus_val else "Random")
    print(f"{'Best Plausibility (Higher = Better)':<45} {single_plaus:<20.2f} {multi_plaus:<20.2f} {random_plaus:<20.2f} {winner_plaus:<15}")
    
    # Best Comfort
    single_comfort = single_analysis['best_comfort']['comfort']
    multi_comfort = multi_analysis['best_comfort']['comfort']
    random_comfort = random_analysis['best_comfort']['comfort']
    best_comfort_val = min(single_comfort, multi_comfort, random_comfort)
    winner_comfort = "Single" if single_comfort == best_comfort_val else ("Multi" if multi_comfort == best_comfort_val else "Random")
    print(f"{'Best Comfort (Lower = Worse)':<45} {single_comfort:<20.2f} {multi_comfort:<20.2f} {random_comfort:<20.2f} {winner_comfort:<15}")
    
    # Critical Scenarios
    single_critical = len(single_analysis['critical'])
    multi_critical = len(multi_analysis['critical'])
    random_critical = len(random_analysis['critical'])
    best_critical_val = max(single_critical, multi_critical, random_critical)
    winner_critical = "Single" if single_critical == best_critical_val else ("Multi" if multi_critical == best_critical_val else "Random")
    print(f"{'Critical Scenarios Found':<45} {single_critical:<20} {multi_critical:<20} {random_critical:<20} {winner_critical:<15}")
    
    # Critical Scenario Efficiency
    single_eff = single_critical / single_analysis['stats']['count'] if single_analysis['stats']['count'] > 0 else 0
    multi_eff = multi_critical / multi_analysis['stats']['count'] if multi_analysis['stats']['count'] > 0 else 0
    random_eff = random_critical / random_analysis['stats']['count'] if random_analysis['stats']['count'] > 0 else 0
    best_eff_val = max(single_eff, multi_eff, random_eff)
    winner_eff = "Single" if single_eff == best_eff_val else ("Multi" if multi_eff == best_eff_val else "Random")
    print(f"{'Efficiency (Critical/Iteration)':<45} {single_eff:<20.4f} {multi_eff:<20.4f} {random_eff:<20.4f} {winner_eff:<15}")
    
    # Pareto Front
    single_pareto = len(single_analysis['pareto'])
    multi_pareto = len(multi_analysis['pareto'])
    random_pareto = len(random_analysis['pareto'])
    best_pareto_val = max(single_pareto, multi_pareto, random_pareto)
    winner_pareto = "Single" if single_pareto == best_pareto_val else ("Multi" if multi_pareto == best_pareto_val else "Random")
    print(f"{'Pareto Front Size':<45} {single_pareto:<20} {multi_pareto:<20} {random_pareto:<20} {winner_pareto:<15}")
    
    # Mean Safety
    single_mean_safety = single_analysis['stats']['safety']['mean']
    multi_mean_safety = multi_analysis['stats']['safety']['mean']
    random_mean_safety = random_analysis['stats']['safety']['mean']
    best_mean_safety = min(single_mean_safety, multi_mean_safety, random_mean_safety)
    winner_mean_safety = "Single" if single_mean_safety == best_mean_safety else ("Multi" if multi_mean_safety == best_mean_safety else "Random")
    print(f"{'Mean Safety (Lower = Worse)':<45} {single_mean_safety:<20.2f} {multi_mean_safety:<20.2f} {random_mean_safety:<20.2f} {winner_mean_safety:<15}")
    
    # Mean Plausibility
    single_mean_plaus = single_analysis['stats']['plausibility']['mean']
    multi_mean_plaus = multi_analysis['stats']['plausibility']['mean']
    random_mean_plaus = random_analysis['stats']['plausibility']['mean']
    best_mean_plaus = max(single_mean_plaus, multi_mean_plaus, random_mean_plaus)
    winner_mean_plaus = "Single" if single_mean_plaus == best_mean_plaus else ("Multi" if multi_mean_plaus == best_mean_plaus else "Random")
    print(f"{'Mean Plausibility (Higher = Better)':<45} {single_mean_plaus:<20.2f} {multi_mean_plaus:<20.2f} {random_mean_plaus:<20.2f} {winner_mean_plaus:<15}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Critical scenarios
    if multi_critical > single_critical and multi_critical > random_critical:
        insights.append(f"[+] Multi-Objective BO found the most critical scenarios ({multi_critical} vs {single_critical} vs {random_critical})")
    elif single_critical > multi_critical and single_critical > random_critical:
        insights.append(f"[+] Single-Objective BO found the most critical scenarios ({single_critical} vs {multi_critical} vs {random_critical})")
    elif random_critical > single_critical and random_critical > multi_critical:
        insights.append(f"[+] Random Search found the most critical scenarios ({random_critical} vs {single_critical} vs {multi_critical})")
    
    # Best safety
    if multi_safety < single_safety and multi_safety < random_safety:
        insights.append(f"[+] Multi-Objective BO found the most dangerous scenario (Safety={multi_safety:.2f} vs {single_safety:.2f} vs {random_safety:.2f})")
    elif single_safety < multi_safety and single_safety < random_safety:
        insights.append(f"[+] Single-Objective BO found the most dangerous scenario (Safety={single_safety:.2f} vs {multi_safety:.2f} vs {random_safety:.2f})")
    
    # Best plausibility
    if multi_plaus > single_plaus and multi_plaus > random_plaus:
        insights.append(f"[+] Multi-Objective BO found the most realistic scenario (Plausibility={multi_plaus:.2f} vs {single_plaus:.2f} vs {random_plaus:.2f})")
    elif single_plaus > multi_plaus and single_plaus > random_plaus:
        insights.append(f"[+] Single-Objective BO found the most realistic scenario (Plausibility={single_plaus:.2f} vs {multi_plaus:.2f} vs {random_plaus:.2f})")
    
    # Efficiency
    if multi_eff > single_eff and multi_eff > random_eff:
        insights.append(f"[+] Multi-Objective BO is most efficient at finding critical scenarios ({multi_eff:.4f} vs {single_eff:.4f} vs {random_eff:.4f})")
    elif single_eff > multi_eff and single_eff > random_eff:
        insights.append(f"[+] Single-Objective BO is most efficient at finding critical scenarios ({single_eff:.4f} vs {multi_eff:.4f} vs {random_eff:.4f})")
    
    # Pareto front
    if multi_pareto > single_pareto and multi_pareto > random_pareto:
        insights.append(f"[+] Multi-Objective BO found the largest Pareto front ({multi_pareto} vs {single_pareto} vs {random_pareto} solutions)")
    elif single_pareto > multi_pareto and single_pareto > random_pareto:
        insights.append(f"[+] Single-Objective BO found the largest Pareto front ({single_pareto} vs {multi_pareto} vs {random_pareto} solutions)")
    
    # Best critical scenario quality
    if len(multi_analysis['critical']) > 0:
        multi_best_critical = multi_analysis['critical'][0]
        insights.append(f"[+] Multi-Objective BO best critical: Safety={multi_best_critical['safety']:.2f}, Plausibility={multi_best_critical['plausibility']:.2f}")
    if len(single_analysis['critical']) > 0:
        single_best_critical = single_analysis['critical'][0]
        insights.append(f"[+] Single-Objective BO best critical: Safety={single_best_critical['safety']:.2f}, Plausibility={single_best_critical['plausibility']:.2f}")
    if len(random_analysis['critical']) > 0:
        random_best_critical = random_analysis['critical'][0]
        insights.append(f"[+] Random Search best critical: Safety={random_best_critical['safety']:.2f}, Plausibility={random_best_critical['plausibility']:.2f}")
    
    if not insights:
        insights.append("All methods performed similarly. More iterations may be needed to see differences.")
    
    for insight in insights:
        print(f"  {insight}")

def main():
    """Main analysis function."""
    print("="*80)
    print("THREE-WAY COMPARISON: Single-Objective BO vs Multi-Objective BO vs Random Search")
    print("="*80)
    
    # Load all three results - try multiple possible paths
    import os
    possible_paths = [
        "final_results_single.json",
        "project_example_code/final_results_single.json",
        "project_example_code/csci513-miniproject1/final_results_single.json",
    ]
    
    single_data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                single_data = load_results(path)
                print(f"Loaded single-objective results from: {path}")
                break
            except Exception as e:
                continue
    
    if single_data is None:
        print("Error: Could not find final_results_single.json in any expected location")
        return
    
    single_analysis = analyze_method(single_data, "Single-Objective Bayesian Optimization")
    
    possible_paths = [
        "final_results_multi.json",
        "project_example_code/final_results_multi.json",
        "project_example_code/csci513-miniproject1/final_results_multi.json",
    ]
    
    multi_data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                multi_data = load_results(path)
                print(f"Loaded multi-objective results from: {path}")
                break
            except Exception as e:
                continue
    
    if multi_data is None:
        print("Error: Could not find final_results_multi.json in any expected location")
        return
    
    multi_analysis = analyze_method(multi_data, "Multi-Objective Bayesian Optimization")
    
    possible_paths = [
        "final_results_random.json",
        "project_example_code/final_results_random.json",
        "project_example_code/csci513-miniproject1/final_results_random.json",
    ]
    
    random_data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                random_data = load_results(path)
                print(f"Loaded random search results from: {path}")
                break
            except Exception as e:
                continue
    
    if random_data is None:
        print("Error: Could not find final_results_random.json in any expected location")
        return
    
    random_analysis = analyze_method(random_data, "Random Search")
    
    # Compare all three
    compare_three_methods(single_analysis, multi_analysis, random_analysis)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()


