"""
Quick check: Do log files exist and what do they contain?
"""

import json
from pathlib import Path
import pandas as pd

# Load results
with open('final_results.json') as f:
    data = json.load(f)

print("=" * 80)
print("CHECKING LOG FILES")
print("=" * 80)

# Check first 3 scenarios
for i in range(min(3, len(data['evaluation_history']))):
    entry = data['evaluation_history'][i]
    iteration = entry['iteration']
    
    if 'metadata' not in entry or 'log_path' not in entry['metadata']:
        print(f"\n❌ Iter {iteration}: No log_path in metadata")
        continue
    
    log_path = Path(entry['metadata']['log_path'])
    print(f"\n📋 Iter {iteration}:")
    print(f"   Log path: {log_path}")
    print(f"   Exists: {log_path.exists()}")
    
    if log_path.exists():
        try:
            df = pd.read_csv(log_path)
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample velocity: {df['ego_velocity'].iloc[0]:.2f} m/s")
            
            # Check for required columns
            required = ['ego_velocity', 'distance_to_lead']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"   ⚠️  Missing columns: {missing}")
            else:
                print(f"   ✓ Has required columns")
                
        except Exception as e:
            print(f"   ❌ Error reading: {e}")

print("\n" + "=" * 80)

