#!/usr/bin/env python3
"""
Debug CP-2 topic gating step by step
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import CorrectedGovernancePipeline, PipelineDecision


def debug_topic_detection():
    """Debug the topic detection algorithm"""
    print("=== DEBUGGING TOPIC DETECTION ===")
    
    pipeline = CorrectedGovernancePipeline()
    cp2 = pipeline.cp2
    
    # Set up reference topic
    reference = "What should I do about React framework?"
    print(f"Reference: {reference}")
    
    # Manually activate topic gate
    cp2._activate_topic_gate(reference, PipelineDecision.HALT)
    print(f"Reference tokens: {cp2.last_topic_tokens}")
    print(f"Topic gate active: {cp2.awaiting_new_topic}")
    
    # Test various prompts
    test_prompts = [
        "Tell me more about React",
        "What about React performance?", 
        "How do I cook pasta?",
        "What are the benefits of exercise?"
    ]
    
    for prompt in test_prompts:
        print(f"\n--- Testing: {prompt} ---")
        
        # Get tokens
        current_tokens = cp2._tokenize_prompt(prompt)
        print(f"Current tokens: {current_tokens}")
        
        # Calculate similarity
        overlap = len(current_tokens & cp2.last_topic_tokens)
        union = len(current_tokens | cp2.last_topic_tokens)
        similarity = overlap / union if union > 0 else 0.0
        
        print(f"Overlap: {overlap}, Union: {union}, Similarity: {similarity:.3f}")
        
        # Check if new topic
        is_new = cp2._is_new_topic(prompt)
        print(f"Is new topic: {is_new}")
        
        # Check if should block
        should_block, message, decision = cp2.should_block_prompt(prompt)
        print(f"Should block: {should_block}")
        if should_block:
            print(f"Block message: {message[:100]}...")


def debug_full_pipeline():
    """Debug the full pipeline flow"""
    print("\n=== DEBUGGING FULL PIPELINE ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=0.8)  # Low threshold
    
    # Mock candidates
    def get_candidates(prompt):
        return [f"You should definitely {prompt.lower()}", f"The best way is {prompt.lower()}"]
    
    # Step 1: Build up risk
    print("\n--- Step 1: Building Risk ---")
    result1 = pipeline.process("Should I use React?", get_candidates("use React"))
    print(f"Result 1: {result1.decision.value}, R_cum: {result1.cp2_metrics.get('R_cum', 0):.3f}")
    
    result2 = pipeline.process("Is React the best?", get_candidates("choose React"))  
    print(f"Result 2: {result2.decision.value}, R_cum: {result2.cp2_metrics.get('R_cum', 0):.3f}")
    
    # Check CP-2 state
    print(f"\nCP-2 State:")
    print(f"  History length: {len(pipeline.cp2.history)}")
    print(f"  Awaiting new topic: {pipeline.cp2.awaiting_new_topic}")
    print(f"  Last topic tokens: {pipeline.cp2.last_topic_tokens}")
    
    # Step 2: Try same topic
    print("\n--- Step 2: Same Topic ---")
    result3 = pipeline.process("More about React please", get_candidates("discuss React"))
    print(f"Result 3: {result3.decision.value}")
    print(f"Response: {result3.final_response[:100]}...")
    
    # Check CP-2 state after halt
    print(f"\nCP-2 State after potential halt:")
    print(f"  Awaiting new topic: {pipeline.cp2.awaiting_new_topic}")
    print(f"  Last topic tokens: {pipeline.cp2.last_topic_tokens}")
    
    # Step 3: Try different topic
    print("\n--- Step 3: Different Topic ---")
    result4 = pipeline.process("How do I cook pasta?", get_candidates("cook pasta"))
    print(f"Result 4: {result4.decision.value}")
    print(f"Response: {result4.final_response[:100]}...")


if __name__ == "__main__":
    debug_topic_detection()
    debug_full_pipeline()