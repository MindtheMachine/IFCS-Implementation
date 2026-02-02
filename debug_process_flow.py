#!/usr/bin/env python3
"""
Debug the exact process flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from corrected_governance_pipeline import CorrectedGovernancePipeline, PipelineDecision


def debug_process_flow():
    """Debug the exact process flow with prints"""
    print("=== DEBUGGING PROCESS FLOW ===")
    
    pipeline = CorrectedGovernancePipeline(cp2_theta=1.0)
    
    # Manually set up CP-2 in halted state
    pipeline.cp2.awaiting_new_topic = True
    pipeline.cp2.last_topic_tokens = {'react', 'framework'}
    pipeline.cp2.pending_decision = PipelineDecision.HALT
    
    print(f"CP-2 state before process:")
    print(f"  awaiting_new_topic: {pipeline.cp2.awaiting_new_topic}")
    print(f"  last_topic_tokens: {pipeline.cp2.last_topic_tokens}")
    print(f"  pending_decision: {pipeline.cp2.pending_decision}")
    
    # Test the topic gate check
    prompt = "Tell me more about React"
    print(f"\nTesting prompt: {prompt}")
    
    should_block, block_message, gate_decision = pipeline.cp2.should_block_prompt(prompt)
    print(f"should_block_prompt returned:")
    print(f"  should_block: {should_block}")
    print(f"  gate_decision: {gate_decision}")
    print(f"  message: {block_message[:50] if block_message else None}...")
    
    # Now test the full process
    candidates = ["Based on evidence, this is recommended."]
    print(f"\nCalling pipeline.process...")
    
    result = pipeline.process(prompt, candidates)
    print(f"Pipeline result:")
    print(f"  decision: {result.decision.value}")
    print(f"  response: {result.final_response[:80]}...")


if __name__ == "__main__":
    debug_process_flow()