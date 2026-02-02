#!/usr/bin/env python3
"""
Final test of topic change functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import CorrectedGovernancePipeline, PipelineDecision


def test_complete_topic_flow():
    """Test complete topic gating flow"""
    print("=== COMPLETE TOPIC GATING FLOW TEST ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=1.0)
    
    def get_candidates():
        return ["Based on evidence, this approach is recommended for optimal results."]
    
    # Phase 1: Build up to CP-2 halt
    print("\n--- Phase 1: Building to CP-2 Halt ---")
    
    # Add some history to trigger halt
    pipeline.cp2.add_turn("Previous", "High risk response", 0.6)
    pipeline.cp2.add_turn("Previous", "Another high risk", 0.5)
    
    result1 = pipeline.process("What should I do about React?", get_candidates())
    print(f"1. Decision: {result1.decision.value}, R_cum: {result1.cp2_metrics.get('R_cum', 0):.3f}")
    
    # Phase 2: Test same topic (should be blocked by topic gate)
    print("\n--- Phase 2: Same Topic (Should Block) ---")
    
    same_topic_tests = [
        "Tell me more about React",
        "What about React performance?",
        "Should I use React components?"
    ]
    
    for prompt in same_topic_tests:
        result = pipeline.process(prompt, get_candidates())
        print(f"'{prompt}' -> {result.decision.value}")
        
        # Check topic similarity manually
        current_tokens = pipeline.cp2._tokenize_prompt(prompt)
        overlap = len(current_tokens & pipeline.cp2.last_topic_tokens)
        union = len(current_tokens | pipeline.cp2.last_topic_tokens)
        similarity = overlap / union if union > 0 else 0.0
        print(f"  Tokens: {current_tokens}, Similarity: {similarity:.3f}")
    
    # Phase 3: Test very different topics
    print("\n--- Phase 3: Different Topics (Should Pass) ---")
    
    different_topics = [
        "How do I bake chocolate cookies?",
        "What exercises are good for fitness?", 
        "Tell me about astronomy and stars"
    ]
    
    for prompt in different_topics:
        result = pipeline.process(prompt, get_candidates())
        print(f"'{prompt}' -> {result.decision.value}")
        
        # Check topic similarity manually
        current_tokens = pipeline.cp2._tokenize_prompt(prompt)
        overlap = len(current_tokens & pipeline.cp2.last_topic_tokens)
        union = len(current_tokens | pipeline.cp2.last_topic_tokens)
        similarity = overlap / union if union > 0 else 0.0
        print(f"  Tokens: {current_tokens}, Similarity: {similarity:.3f}")
        
        if result.decision == PipelineDecision.PASS:
            print(f"  ✅ Topic change successful!")
            break
    
    print(f"\nFinal CP-2 state:")
    print(f"  awaiting_new_topic: {pipeline.cp2.awaiting_new_topic}")
    print(f"  last_topic_tokens: {pipeline.cp2.last_topic_tokens}")


if __name__ == "__main__":
    test_complete_topic_flow()