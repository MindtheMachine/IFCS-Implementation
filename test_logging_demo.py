#!/usr/bin/env python3
"""
Demonstration of comprehensive κ(z*) logging functionality
Shows the logging features implemented for task 4.1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ifcs_engine import IFCSEngine
from trilogy_config import IFCSConfig

def demonstrate_logging():
    """Demonstrate the comprehensive logging functionality for κ(z*) decisions"""
    
    print("="*80)
    print("IFCS COMMITMENT-ACTUALITY GATE LOGGING DEMONSTRATION")
    print("="*80)
    
    # Initialize IFCS engine with minimal config
    ifcs_config = IFCSConfig()
    ifcs_engine = IFCSEngine(ifcs_config)
    
    # Test cases to demonstrate different logging scenarios
    test_cases = [
        {
            'name': 'Commitment-Bearing Example',
            'response': 'You should definitely use React for this project because it provides the best performance and scalability.',
            'prompt': 'What framework should I use for my web project?',
            'expected_kappa': 1
        },
        {
            'name': 'Non-Commitment-Bearing Example',
            'response': 'Popular web development frameworks include React, Vue, Angular, and Svelte. Each has different strengths and use cases.',
            'prompt': 'What are the current best practices for web development?',
            'expected_kappa': 0
        },
        {
            'name': 'Edge Case - Mixed Signals',
            'response': 'You might consider using React, though Vue and Angular are also good options depending on your specific requirements.',
            'prompt': 'What should I use for my project?',
            'expected_kappa': 0  # Hedging should reduce commitment
        },
        {
            'name': 'High Authority Example',
            'response': 'You must implement proper error handling. This is absolutely critical for production systems.',
            'prompt': 'How should I handle errors in my application?',
            'expected_kappa': 1
        },
        {
            'name': 'Descriptive Enumeration',
            'response': 'Common error handling approaches include try-catch blocks, error boundaries, logging systems, and user-friendly error messages.',
            'prompt': 'What are different ways to handle errors?',
            'expected_kappa': 0
        }
    ]
    
    print("\n🔍 TESTING CLASSIFICATION AND LOGGING:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Response: {test_case['response'][:100]}...")
        print(f"Expected κ(z*): {test_case['expected_kappa']}")
        
        # Test the shape_commitment method which includes comprehensive logging
        shaped_response, risk, debug_info = ifcs_engine.shape_commitment(
            response=test_case['response'],
            prompt=test_case['prompt'],
            context="",
            sigma=1.0  # High sigma to ensure σ ≥ τ condition is met
        )
        
        actual_kappa = debug_info['kappa']
        print(f"Actual κ(z*): {actual_kappa}")
        print(f"Classification: {'✅ CORRECT' if actual_kappa == test_case['expected_kappa'] else '❌ INCORRECT'}")
        print(f"κ(z*) Computation Time: {debug_info.get('kappa_computation_time_ms', 0):.2f}ms")
        print(f"Total Processing Time: {debug_info.get('total_processing_time_ms', 0):.2f}ms")
        
        if debug_info.get('latency_improvement_ms', 0) > 0:
            print(f"Latency Improvement: {debug_info['latency_improvement_ms']:.2f}ms (avoided IFCS processing)")
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    # Display performance report
    ifcs_engine.print_performance_report()
    
    print("\n📋 RECENT CLASSIFICATION LOGS (Last 3):")
    print("-" * 50)
    recent_logs = ifcs_engine.get_classification_logs(limit=3)
    for i, log in enumerate(recent_logs, 1):
        print(f"\n[Log {i}] {log.timestamp}")
        print(f"  κ(z*): {log.kappa_value} ({log.classification})")
        print(f"  Computation Time: {log.computation_time_ms:.2f}ms")
        print(f"  Context: {log.context_metadata['word_count']} words, {log.context_metadata['sentence_count']} sentences")
        print(f"  Decision Logic: {log.classification_reasoning['decision_logic']}")
    
    print("\n📈 NON-COMMITMENT-BEARING METRICS:")
    print("-" * 50)
    non_commitment_metrics = ifcs_engine.get_non_commitment_metrics()
    if non_commitment_metrics:
        for i, metric in enumerate(non_commitment_metrics[-2:], 1):  # Last 2
            print(f"\n[Metric {i}] {metric.timestamp}")
            print(f"  Rationale: {metric.classification_rationale}")
            print(f"  Score Difference: {metric.final_score_difference:.3f}")
            print(f"  Context Bias: {metric.context_bias:.3f}")
            if metric.hedging_penalty > 0:
                print(f"  Hedging Penalty: {metric.hedging_penalty:.3f}")
    else:
        print("  No non-commitment-bearing contexts in this test run.")
    
    print("\n💾 EXPORTING LOGS FOR EXTERNAL ANALYSIS:")
    print("-" * 50)
    
    # Export logs to JSON files
    ifcs_engine.export_classification_logs_json("classification_logs_demo.json")
    ifcs_engine.export_non_commitment_metrics_json("non_commitment_metrics_demo.json")
    
    print("\n✅ LOGGING DEMONSTRATION COMPLETE")
    print("="*80)
    print("Key Features Demonstrated:")
    print("• κ(z*) decisions logged with context metadata and reasoning")
    print("• Performance metrics for κ(z*) computation time")
    print("• Non-commitment-bearing contexts recorded with rationale")
    print("• Debug information added to existing IFCS debug output")
    print("• JSON export capability for external analysis")
    print("• Performance target validation (< 50ms)")
    print("="*80)

if __name__ == "__main__":
    demonstrate_logging()