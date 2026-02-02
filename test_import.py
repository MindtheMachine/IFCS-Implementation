#!/usr/bin/env python3
"""
Test signal estimation import
"""

try:
    from signal_estimation import TrueSignalEstimator
    print("✅ TrueSignalEstimator imported successfully")
    
    estimator = TrueSignalEstimator()
    print("✅ TrueSignalEstimator instantiated successfully")
    
    # Test a simple method
    result = estimator.estimate_assertion_strength("You should definitely use this approach.")
    print(f"✅ estimate_assertion_strength returned: {result}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

try:
    from signal_estimation import signal_estimator
    print("✅ signal_estimator imported successfully")
    
    # Test the global instance
    result = signal_estimator.estimate_assertion_strength("You should definitely use this approach.")
    print(f"✅ Global signal_estimator works: {result}")
    
except ImportError as e:
    print(f"❌ Import error for signal_estimator: {e}")
except Exception as e:
    print(f"❌ Other error with signal_estimator: {e}")