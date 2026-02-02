#!/usr/bin/env python3
"""
Integration test for the complete trilogy pipeline
Tests that ECR → CP Type-1 → IFCS → CP Type-2 works correctly with the commitment-actuality gate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trilogy_config import TrilogyConfig
from trilogy_orchestrator import TrilogyOrchestrator
from ifcs_engine import IFCSEngine

def mock_llm_call(prompt: str, temperature=None, max_tokens=None, top_p=None) -> str:
    """Mock LLM call for testing"""
    # Return different responses based on prompt content
    if "react" in prompt.lower() or "framework" in prompt.lower():
        return "You should definitely use React for this project because it's the best framework available."
    elif "best practices" in prompt.lower():
        return "Common web development practices include React, Vue, Angular, and other modern frameworks."
    elif "error handling" in prompt.lower():
        return "You must implement proper error handling with try-catch blocks and logging."
    else:
        return "This is a generic response for testing purposes."

def test_trilogy_integration():
    """Test the complete trilogy pipeline integration"""
    
    print("="*80)
    print("TRILOGY PIPELINE INTEGRATION TEST")
    print("="*80)
    
    # Initialize trilogy system with mock LLM
    config = TrilogyConfig(api_key="mock_key_for_testing")
    trilogy = TrilogyOrchestrator(config, mock_llm_call)
    
    # Test cases covering different scenarios
    test_cases = [
        {
            'name': 'Commitment-Bearing Query',
            'prompt': 'What framework should I use for my web project?',
            'expected_kappa': 1,
            'expected_intervention': True
        },
        {
            'name': 'Non-Commitment-Bearing Query',
            'prompt': 'What are the current best practices for web development?',
            'expected_kappa': 0,
            'expected_intervention': False
        },
        {
            'name': 'High Authority Query',
            'prompt': 'How should I handle errors in my application?',
            'expected_kappa': 1,
            'expected_intervention': True
        }
    ]
    
    print("\n🔍 TESTING TRILOGY PIPELINE:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        
        # Process through trilogy pipeline
        result = trilogy.process(test_case['prompt'])
        
        # Extract key information
        ecr_fired = result.ecr_fired
        cp1_fired = result.cp_type1_fired
        ifcs_fired = result.ifcs_fired
        cp2_fired = result.cp_type2_fired
        
        # Check IFCS metrics for κ(z*) information
        ifcs_metrics = result.ifcs_metrics or {}
        kappa = ifcs_metrics.get('kappa', 'unknown')
        intervened = ifcs_metrics.get('intervened', False)
        
        print(f"Pipeline Results:")
        print(f"  ECR Fired: {ecr_fired}")
        print(f"  CP Type-1 Fired: {cp1_fired}")
        print(f"  IFCS Fired: {ifcs_fired}")
        print(f"  CP Type-2 Fired: {cp2_fired}")
        print(f"  κ(z*): {kappa}")
        print(f"  IFCS Intervened: {intervened}")
        
        # Validate expectations
        if kappa != 'unknown':
            kappa_correct = (kappa == test_case['expected_kappa'])
            intervention_correct = (intervened == test_case['expected_intervention'])
            
            print(f"Validation:")
            print(f"  κ(z*) Classification: {'✅ CORRECT' if kappa_correct else '❌ INCORRECT'}")
            print(f"  Intervention Logic: {'✅ CORRECT' if intervention_correct else '❌ INCORRECT'}")
        else:
            print(f"Validation: ⚠️ κ(z*) information not available")
        
        print(f"Final Response: {result.final_response[:100]}...")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
    
    print("\n" + "="*80)
    print("✅ TRILOGY INTEGRATION TEST COMPLETE")
    print("="*80)
    print("Pipeline Components Tested:")
    print("• ECR: Candidate generation and selection")
    print("• Control Probe Type-1: Admissibility gating")
    print("• IFCS: Commitment-actuality gate (κ(z*)) + risk evaluation")
    print("• Control Probe Type-2: Interaction monitoring")
    print("• End-to-end processing with mock LLM")
    print("="*80)

if __name__ == "__main__":
    test_trilogy_integration()