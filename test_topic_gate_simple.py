#!/usr/bin/env python3
"""
Simple focused test of CP-2 topic gating
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import CorrectedGovernancePipeline, PipelineDecision


def test_topic_gate_simple():
    """Simple test of topic gating behavior"""
    print("=== SIMPLE TOPIC GATE TEST ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=1.0)
    
    # Good candidates that pass CP-1 but have high commitment risk
    def get_good_candidates():
        return [
            "Based on research evidence, this approach is definitely recommended for your situation.",
            "Studies clearly show this is always the best method to use in these cases.",
            "You should certainly follow this proven approach for optimal results."
        ]
    
    print("\n1. First request (should pass)")
    result1 = pipeline.process("What should I do about React?", get_good_candidates())
    print(f"   Decision: {result1.decision.value}")
    print(f"   R_cum: {result1.cp2_metrics.get('R_cum', 0):.3f}")
    print(f"   Topic gate active: {pipeline.cp2.awaiting_new_topic}")
    
    print("\n2. Second request (should trigger CP-2 halt)")
    result2 = pipeline.process("Is React definitely the best?", get_good_candidates())
    print(f"   Decision: {result2.decision.value}")
    print(f"   R_cum: {result2.cp2_metrics.get('R_cum', 0):.3f}")
    print(f"   Topic gate active: {pipeline.cp2.awaiting_new_topic}")
    
    print("\n3. Same topic request (should be blocked by topic gate)")
    result3 = pipeline.process("Tell me more about React", get_good_candidates())
    print(f"   Decision: {result3.decision.value}")
    print(f"   Response: {result3.final_response[:80]}...")
    print(f"   Topic gate active: {pipeline.cp2.awaiting_new_topic}")
    
    # Check what the topic gate thinks
    should_block, message, decision = pipeline.cp2.should_block_prompt("Tell me more about React")
    print(f"   Topic gate says should_block: {should_block}")
    
    print("\n4. Different topic request (should pass)")
    result4 = pipeline.process("How do I cook pasta?", get_good_candidates())
    print(f"   Decision: {result4.decision.value}")
    print(f"   Response: {result4.final_response[:80]}...")
    print(f"   Topic gate active: {pipeline.cp2.awaiting_new_topic}")


if __name__ == "__main__":
    test_topic_gate_simple()