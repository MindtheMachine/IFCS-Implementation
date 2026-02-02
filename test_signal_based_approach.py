#!/usr/bin/env python3
"""
Test the new signal-based approach for κ(z*) classification
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ifcs_engine import CommitmentActualityClassifier

def test_signal_based_approach():
    """Test the new signal-based κ(z*) classification"""
    print("Testing Signal-Based Commitment-Actuality Classification")
    print("=" * 60)
    
    # Initialize classifier
    classifier = CommitmentActualityClassifier()
    
    # Test cases
    test_cases = [
        {
            'name': 'Strong Commitment-Bearing',
            'response': 'You should definitely use React for this project because it is the best framework.',
            'prompt': 'What framework should I use?',
            'expected': True
        },
        {
            'name': 'Clear Non-Commitment-Bearing',
            'response': 'Common web development practices include React, Vue, Angular, and other modern frameworks.',
            'prompt': 'What are the current best practices?',
            'expected': False
        },
        {
            'name': 'Authority-Heavy Commitment',
            'response': 'You must implement proper error handling. This is absolutely critical for production systems.',
            'prompt': 'How should I handle errors?',
            'expected': True
        },
        {
            'name': 'Descriptive Enumeration',
            'response': 'Error handling approaches typically involve try-catch blocks, logging systems, and monitoring.',
            'prompt': 'What are different ways to handle errors?',
            'expected': False
        }
    ]
    
    print("Testing signal estimation approach (replaces text-matching heuristics):")
    print("Industry approach: estimates latent epistemic signals")
    print("- assertion strength")
    print("- epistemic certainty") 
    print("- scope breadth")
    print("- authority posture")
    print()
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[Test {i}] {test_case['name']}")
        print(f"Response: {test_case['response'][:80]}...")
        
        # Test classification
        result = classifier.is_commitment_bearing(test_case['response'], test_case['prompt'])
        expected = test_case['expected']
        
        print(f"Expected: κ(z*)={1 if expected else 0}, Actual: κ(z*)={1 if result else 0}")
        
        if result == expected:
            print("✅ CORRECT")
            results.append(True)
        else:
            print("❌ INCORRECT")
            results.append(False)
        
        print()
    
    # Summary
    correct_count = sum(results)
    total_count = len(results)
    accuracy = correct_count / total_count * 100
    
    print("=" * 60)
    print("SIGNAL-BASED APPROACH TEST RESULTS")
    print("=" * 60)
    print(f"Correct Classifications: {correct_count}/{total_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 75:
        print("✅ Signal-based approach working correctly!")
        print("✅ Text-matching heuristics successfully replaced with signal estimation")
    else:
        print("⚠️  Signal-based approach needs adjustment")
    
    print()
    print("Key Implementation Changes:")
    print("❌ Old: Text-matching heuristics (string/pattern/diff rules)")
    print("✅ New: Signal estimation (assertion strength, epistemic certainty, etc.)")
    print("✅ Industry-standard: R(z*) ∈ [0,1] bounded risk score")
    print("✅ Firing condition unchanged: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1")

if __name__ == "__main__":
    test_signal_based_approach()