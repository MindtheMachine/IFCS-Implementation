#!/usr/bin/env python3
"""
Proper test of CP-2 topic gating with correct setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import CorrectedGovernancePipeline, PipelineDecision


def test_topic_gate_properly():
    """Test topic gating with proper setup"""
    print("=== PROPER TOPIC GATING TEST ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=0.8)  # Low threshold
    
    def get_high_risk_candidates():
        return [
            "You should definitely use this approach because it's always the best method.",
            "This is clearly the optimal solution that you must implement immediately.",
            "Based on evidence, this is certainly the correct way to proceed."
        ]
    
    # Phase 1: Build up cumulative risk naturally
    print("\n--- Phase 1: Building Cumulative Risk ---")
    
    result1 = pipeline.process("What should I do about React?", get_high_risk_candidates())
    print(f"1. '{result1.final_response[:50]}...' -> {result1.decision.value}")
    print(f"   R_cum: {result1.cp2_metrics.get('R_cum', 0):.3f}, Gate: {pipeline.cp2.awaiting_new_topic}")
    
    result2 = pipeline.process("Is React the best framework?", get_high_risk_candidates())
    print(f"2. '{result2.final_response[:50]}...' -> {result2.decision.value}")
    print(f"   R_cum: {result2.cp2_metrics.get('R_cum', 0):.3f}, Gate: {pipeline.cp2.awaiting_new_topic}")
    
    # Check if CP-2 has been triggered
    if pipeline.cp2.awaiting_new_topic:
        print(f"   ✅ CP-2 topic gate activated!")
        print(f"   Last topic tokens: {pipeline.cp2.last_topic_tokens}")
    else:
        print("   ⚠️  CP-2 not triggered yet, trying one more...")
        result3 = pipeline.process("Should I always use React?", get_high_risk_candidates())
        print(f"3. '{result3.final_response[:50]}...' -> {result3.decision.value}")
        print(f"   R_cum: {result3.cp2_metrics.get('R_cum', 0):.3f}, Gate: {pipeline.cp2.awaiting_new_topic}")
        print(f"   Last topic tokens: {pipeline.cp2.last_topic_tokens}")
    
    # Phase 2: Test same topic blocking
    print("\n--- Phase 2: Same Topic (Should Block) ---")
    
    same_topic_prompts = [
        "Tell me more about React benefits",
        "What about React performance issues?",
        "How do React components work?"
    ]
    
    for prompt in same_topic_prompts:
        result = pipeline.process(prompt, get_high_risk_candidates())
        print(f"'{prompt}' -> {result.decision.value}")
        
        # Check if it's the topic gate message
        if "limit for commitment-heavy responses" in result.final_response:
            print(f"   ✅ Blocked by topic gate")
        else:
            print(f"   ❌ Not blocked by topic gate: {result.final_response[:50]}...")
    
    # Phase 3: Test topic change
    print("\n--- Phase 3: Topic Change (Should Pass) ---")
    
    new_topic_prompts = [
        "How do I bake chocolate chip cookies?",
        "What are good exercises for fitness?",
        "Tell me about space exploration"
    ]
    
    for prompt in new_topic_prompts:
        # Check what topic gate thinks first
        should_block, message, decision = pipeline.cp2.should_block_prompt(prompt)
        print(f"Topic gate check for '{prompt}': should_block={should_block}")
        
        if not should_block:
            result = pipeline.process(prompt, get_high_risk_candidates())
            print(f"   -> {result.decision.value}")
            print(f"   Gate after: {pipeline.cp2.awaiting_new_topic}")
            
            if result.decision == PipelineDecision.PASS:
                print(f"   ✅ Topic change successful!")
                break
        else:
            print(f"   Still blocked - trying next topic...")


if __name__ == "__main__":
    test_topic_gate_properly()