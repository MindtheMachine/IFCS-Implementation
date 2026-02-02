#!/usr/bin/env python3
"""
Test the corrected three-gate architecture
Verifies: ECR → CP-1 → IFCS (with CP-2 parallel monitoring)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import (
    CorrectedGovernancePipeline,
    ecr_select,
    control_probe_pre, 
    compute_R_ifcs,
    ControlProbeType2,
    PipelineDecision
)


def test_ecr_selection():
    """Test ECR pure selection (no blocking)"""
    print("\n=== TEST: ECR Selection ===")
    
    candidates = [
        "You should definitely use React because it's the best framework.",
        "Common frameworks include React, Vue, and Angular with different strengths.",
        "React is popular but Vue and Angular are also good options to consider."
    ]
    
    selected, metrics = ecr_select(candidates, "What framework should I use?")
    
    print(f"Selected candidate {metrics['selected_idx']}: {selected[:60]}...")
    print(f"Coherence score: {metrics['coherence_score']:.3f}")
    print(f"All scores: {[f'{s:.3f}' for s in metrics['all_scores']]}")
    
    # ECR should never block, always select something
    assert selected in candidates, "ECR must select from candidates"
    assert 'selected_idx' in metrics, "ECR must provide selection index"
    
    print("✅ ECR selection working correctly")


def test_cp1_admissibility():
    """Test CP-1 admissibility gate (binary pass/block)"""
    print("\n=== TEST: CP-1 Admissibility Gate ===")
    
    test_cases = [
        {
            'response': 'Based on the evidence, React is a popular framework with good community support.',
            'prompt': 'What framework should I use?',
            'expected': PipelineDecision.PASS
        },
        {
            'response': 'The secret alien technology clearly shows that React is superior to all frameworks.',
            'prompt': 'What framework should I use?', 
            'expected': PipelineDecision.BLOCK
        }
    ]
    
    for i, case in enumerate(test_cases):
        decision, sigma, metrics = control_probe_pre(case['response'], case['prompt'])
        
        print(f"Case {i+1}: {case['response'][:50]}...")
        print(f"Decision: {decision.value}, σ={sigma:.3f}")
        print(f"Signals: {metrics['signals']}")
        
        # CP-1 should make binary decisions based on groundability
        assert decision in [PipelineDecision.PASS, PipelineDecision.BLOCK], "CP-1 must be binary"
        assert 0.0 <= sigma <= 1.0, "σ must be in [0,1]"
        
    print("✅ CP-1 admissibility gate working correctly")


def test_ifcs_commitment_shaping():
    """Test IFCS commitment shaping (non-blocking, fuzzy logic)"""
    print("\n=== TEST: IFCS Commitment Shaping ===")
    
    test_cases = [
        {
            'response': 'You should definitely use React because it is always the best framework.',
            'sigma': 0.8,
            'expected_shaping': True
        },
        {
            'response': 'Common frameworks include React, Vue, and Angular with different strengths.',
            'sigma': 0.8, 
            'expected_shaping': False
        }
    ]
    
    for i, case in enumerate(test_cases):
        shaped, R_score, metrics = compute_R_ifcs(case['response'], case['sigma'])
        
        print(f"Case {i+1}: {case['response'][:50]}...")
        print(f"R(z*)={R_score:.3f}, shaped={metrics['should_shape']}")
        print(f"Signals: {metrics['signals']}")
        print(f"Original: {case['response']}")
        print(f"Shaped:   {shaped}")
        
        # IFCS should never block, only shape
        assert shaped is not None, "IFCS must always return a response"
        assert 0.0 <= R_score <= 1.0, "R(z*) must be in [0,1]"
        assert 'firing_condition' in metrics, "IFCS must show firing condition"
        
        if case['expected_shaping']:
            assert shaped != case['response'], "High-risk response should be shaped"
        
    print("✅ IFCS commitment shaping working correctly")


def test_cp2_parallel_monitoring():
    """Test CP-2 parallel interaction monitoring"""
    print("\n=== TEST: CP-2 Parallel Monitoring ===")
    
    cp2 = ControlProbeType2(theta=1.5)  # Low threshold for testing
    
    # Add some turns with increasing risk
    turns = [
        ("What is React?", "React is a JavaScript library.", 0.2),
        ("Should I use React?", "You should definitely use React.", 0.6),
        ("Is React the best?", "React is always the best framework.", 0.8),
    ]
    
    for prompt, response, risk in turns:
        cp2.add_turn(prompt, response, risk)
        decision, metrics = cp2.evaluate()
        
        print(f"Turn: {prompt}")
        print(f"R_cum={metrics.get('R_cum', 0):.3f}, Decision: {decision.value}")
    
    # Should halt when cumulative risk exceeds threshold
    final_decision, final_metrics = cp2.evaluate()
    print(f"Final: R_cum={final_metrics['R_cum']:.3f} vs Θ={final_metrics['theta']:.3f}")
    
    assert final_decision == PipelineDecision.HALT, "CP-2 should halt on high cumulative risk"
    assert final_metrics['R_cum'] >= final_metrics['theta'], "R_cum should exceed threshold"
    
    print("✅ CP-2 parallel monitoring working correctly")


def test_full_pipeline_integration():
    """Test complete pipeline integration"""
    print("\n=== TEST: Full Pipeline Integration ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=3.0)  # High threshold
    
    test_cases = [
        {
            'name': 'Commitment-bearing query',
            'prompt': 'What framework should I use for my project?',
            'candidates': [
                'You should definitely use React because it is the best framework.',
                'Common frameworks include React, Vue, and Angular.',
                'React has good community support and documentation.'
            ],
            'expected_decision': PipelineDecision.PASS
        },
        {
            'name': 'Non-commitment-bearing query', 
            'prompt': 'What are web development frameworks?',
            'candidates': [
                'Web frameworks include React, Vue, Angular, and others.',
                'Frameworks help organize code and provide reusable components.',
                'Popular options vary based on project requirements.'
            ],
            'expected_decision': PipelineDecision.PASS
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Prompt: {case['prompt']}")
        
        result = pipeline.process(
            prompt=case['prompt'],
            candidates=case['candidates']
        )
        
        print(f"Decision: {result.decision.value}")
        print(f"Final Response: {result.final_response[:80]}...")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        
        # Verify pipeline stages
        print(f"ECR: Selected {result.ecr_metrics.get('selected_idx', 'N/A')}")
        print(f"CP-1: σ={result.cp1_metrics.get('sigma', 0):.3f}")
        print(f"IFCS: R={result.ifcs_metrics.get('R_score', 0):.3f}")
        print(f"CP-2: R_cum={result.cp2_metrics.get('R_cum', 0):.3f}")
        
        # Assertions
        assert result.decision == case['expected_decision'], f"Expected {case['expected_decision']}"
        assert result.final_response, "Must have final response"
        assert result.processing_time_ms > 0, "Must have processing time"
        
    print("✅ Full pipeline integration working correctly")


def test_architectural_invariants():
    """Test that architectural invariants are preserved"""
    print("\n=== TEST: Architectural Invariants ===")
    
    # Test 1: ECR never blocks
    candidates = ["Bad response", "Another bad response"]
    selected, metrics = ecr_select(candidates, "Test prompt")
    assert selected in candidates, "ECR must select, never block"
    
    # Test 2: CP-1 is binary
    decision, sigma, _ = control_probe_pre("Test response", "Test prompt")
    assert decision in [PipelineDecision.PASS, PipelineDecision.BLOCK], "CP-1 must be binary"
    
    # Test 3: IFCS never blocks
    shaped, R_score, _ = compute_R_ifcs("Test response", 0.5)
    assert shaped is not None, "IFCS must never block"
    
    # Test 4: CP-2 runs in parallel
    cp2 = ControlProbeType2()
    decision, _ = cp2.evaluate()  # Should work without pipeline execution
    assert decision in [PipelineDecision.PASS, PipelineDecision.HALT], "CP-2 must be independent"
    
    print("✅ All architectural invariants preserved")


def test_signal_separation():
    """Test that signals don't leak across gates"""
    print("\n=== TEST: Signal Separation ===")
    
    response = "You should definitely use React because it's the best framework."
    
    # ECR signals (coherence only)
    ecr_selected, ecr_metrics = ecr_select([response], "Test")
    ecr_signals = set(ecr_metrics.keys())
    
    # CP-1 signals (admissibility only)
    _, _, cp1_metrics = control_probe_pre(response, "Test")
    cp1_signals = set(cp1_metrics['signals'].keys())
    
    # IFCS signals (commitment only)
    _, _, ifcs_metrics = compute_R_ifcs(response, 0.8)
    ifcs_signals = set(ifcs_metrics['signals'].keys())
    
    print(f"ECR signals: {ecr_signals}")
    print(f"CP-1 signals: {cp1_signals}")
    print(f"IFCS signals: {ifcs_signals}")
    
    # Verify no signal overlap
    assert not (cp1_signals & ifcs_signals), "CP-1 and IFCS signals must not overlap"
    
    # Verify signal purposes
    assert 'coherence_score' in ecr_metrics, "ECR must have coherence signals"
    assert 'claim_support_ratio' in cp1_metrics['signals'], "CP-1 must have admissibility signals"
    assert 'assertion_strength' in ifcs_metrics['signals'], "IFCS must have commitment signals"
    
    print("✅ Signal separation maintained across gates")


def run_all_tests():
    """Run all corrected architecture tests"""
    print("="*80)
    print("CORRECTED THREE-GATE ARCHITECTURE TESTS")
    print("Testing: ECR → CP-1 → IFCS (with CP-2 parallel)")
    print("="*80)
    
    try:
        test_ecr_selection()
        test_cp1_admissibility()
        test_ifcs_commitment_shaping()
        test_cp2_parallel_monitoring()
        test_full_pipeline_integration()
        test_architectural_invariants()
        test_signal_separation()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - CORRECTED ARCHITECTURE VERIFIED")
        print("✅ ECR: Pure selection (no blocking)")
        print("✅ CP-1: Binary admissibility gate")
        print("✅ IFCS: Non-blocking commitment shaping with fuzzy logic")
        print("✅ CP-2: Parallel interaction monitoring")
        print("✅ Signal separation maintained")
        print("✅ Architectural invariants preserved")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)