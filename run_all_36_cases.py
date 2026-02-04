#!/usr/bin/env python3
"""
Run all 36 taxonomy test cases using the existing trilogy_app infrastructure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trilogy_app import TrilogyApp
from trilogy_config import TEST_CASES_36_TAXONOMY
import json
import time


def load_env_file(env_path: str) -> None:
    """Load uncommented KEY=VALUE pairs from a .env file into os.environ."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name:
                os.environ[name] = value


def main():
    """Run all 36 taxonomy test cases"""
    print("="*80)
    print("RUNNING ALL 36 TAXONOMY TEST CASES WITH REAL LLM")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Load .env values (uncommented only)
        load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

        # Initialize the trilogy app (uses .env configuration)
        app = TrilogyApp()
        
        print("[OK] Initialized TrilogyApp")
        print(f"[OK] LLM Provider: {type(app.llm_provider).__name__}")
        print(f"[OK] Model: {app.llm_provider.get_model_name()}")
        
        # Run all 36 test cases
        print(f"\nRunning all {len(TEST_CASES_36_TAXONOMY)} test cases...")
        
        results = app.run_test_suite(TEST_CASES_36_TAXONOMY)  # Pass all cases, not just first 10
        
        total_time = time.time() - start_time
        
        # Save results
        with open('all_36_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        print("\n" + "="*80)
        print("ALL 36 TAXONOMY TEST CASES COMPLETED")
        print("="*80)
        print(f"Total cases: {len(results)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: all_36_test_results.json")
        
        # Success/failure analysis
        successful = sum(1 for r in results if r.get('expected_fired') is not None)
        print(f"Successfully processed: {successful}/{len(results)}")
        
        # Decision breakdown
        decisions = {}
        for result in results:
            if result.get('comparison', {}).get('mechanisms_fired'):
                mech = result['comparison']['mechanisms_fired']
                for mechanism, fired in mech.items():
                    if fired:
                        decisions[mechanism] = decisions.get(mechanism, 0) + 1
        
        print(f"\nMechanisms fired:")
        for mechanism, count in decisions.items():
            print(f"  {mechanism}: {count} cases")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
