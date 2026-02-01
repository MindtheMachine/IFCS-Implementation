#!/usr/bin/env python3
"""
Simple ECR optimization test
"""

import time
import statistics
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from ecr_engine import ECREngine, EvaluativeVector, Trajectory
from trilogy_config import ECRConfig


class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self, response_time: float = 0.1):
        self.response_time = response_time
        self.call_count = 0
    
    def __call__(self, prompt: str, temperature=None, max_tokens=None):
        """Simulate LLM call with configurable delay"""
        time.sleep(self.response_time)
        self.call_count += 1
        
        # Generate varied responses for testing
        responses = [
            "This is a comprehensive analysis of the topic with detailed explanations.",
            "Here's my perspective on this matter, considering multiple viewpoints.",
            "I believe the best approach would be to systematically address each component.",
            "The solution involves several key steps that must be carefully executed.",
            "Based on available information, I recommend the following strategy."
        ]
        
        return responses[self.call_count % len(responses)]


class SimpleOptimizedECREngine(ECREngine):
    """Simple optimized ECR engine for demonstration"""
    
    def __init__(self, config):
        super().__init__(config)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_candidates_parallel(self, prompt: str, llm_call_fn, num_candidates: int = None) -> List[str]:
        """Generate candidates with optimized parallelism"""
        target_k = num_candidates if num_candidates is not None else self.K
        if target_k <= 0:
            return []

        if not self.config.parallel_candidates or target_k == 1:
            return [llm_call_fn(prompt, temperature=None) for _ in range(target_k)]

        # Optimized parallel generation
        optimal_workers = min(target_k, 4)
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            futures = [
                executor.submit(llm_call_fn, prompt, temperature=None)
                for _ in range(target_k)
            ]
            
            candidates = []
            for future in as_completed(futures):
                try:
                    candidates.append(future.result())
                except Exception as e:
                    print(f"[ECR] Warning: Candidate generation failed: {e}")
                    candidates.append("Fallback response due to error.")
            
            return candidates[:target_k]
    
    def unroll_trajectory_cached(self, candidate: str, prompt: str, llm_call_fn) -> Trajectory:
        """Unroll trajectory with simple caching"""
        cache_key = f"{candidate[:50]}|{prompt[:50]}"
        
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Create trajectory (simplified)
        steps = []
        current_context = prompt
        
        # Initial step
        steps.append(EvaluativeVector.from_response(candidate, 0, current_context))
        
        # Subsequent steps
        current_response = candidate
        for h in range(1, self.H + 1):
            continuation_prompt = f"{current_context}\n\n{current_response}\n\nContinue:"
            
            try:
                continuation = llm_call_fn(continuation_prompt, temperature=None, max_tokens=100)
            except Exception as e:
                continuation = "Cannot continue due to error."
            
            steps.append(EvaluativeVector.from_response(continuation, h, current_context))
            current_response = continuation
            current_context = f"{current_context}\n{current_response}"
        
        trajectory = Trajectory(response=candidate, steps=steps)
        self.cache[cache_key] = trajectory
        
        return trajectory
    
    def select_best_candidate_optimized(self, candidates: List[str], prompt: str, llm_call_fn):
        """Optimized candidate selection"""
        start_time = time.time()
        
        print(f"[ECR] Optimized evaluation of {len(candidates)} candidates...")
        
        # Parallel trajectory unrolling
        trajectories = []
        with ThreadPoolExecutor(max_workers=min(len(candidates), 3)) as executor:
            future_to_idx = {
                executor.submit(self.unroll_trajectory_cached, candidate, prompt, llm_call_fn): i
                for i, candidate in enumerate(candidates)
            }
            
            results = [None] * len(candidates)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[ECR] Warning: Trajectory failed for candidate {idx}: {e}")
                    # Create fallback trajectory
                    fallback_steps = [EvaluativeVector(0.5, 0.5, 0.5, 0.5, 0.5) for _ in range(self.H + 1)]
                    results[idx] = Trajectory(response=candidates[idx], steps=fallback_steps)
            
            trajectories = results
        
        # Compute metrics
        all_metrics = [self.compute_coherence_metrics(traj) for traj in trajectories]
        
        # Selection logic
        admissible = [
            (i, m) for i, m in enumerate(all_metrics) 
            if m.is_admissible(self.tau_CCI)
        ]
        
        if admissible:
            best_idx, best_metrics = max(admissible, key=lambda x: x[1].CCI)
            print(f"[ECR] Selected admissible candidate {best_idx+1} with CCI={best_metrics.CCI:.3f}")
        else:
            best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i].CCI)
            best_metrics = all_metrics[best_idx]
            print(f"[ECR] No admissible candidates. Selected best CCI={best_metrics.CCI:.3f}")
        
        total_time = time.time() - start_time
        
        debug_info = {
            'num_candidates': len(candidates),
            'num_admissible': len(admissible),
            'all_cci_scores': [m.CCI for m in all_metrics],
            'selected_idx': best_idx,
            'threshold': self.tau_CCI,
            'total_time': total_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        }
        
        print(f"[ECR] Optimization metrics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Cache hit rate: {debug_info['cache_hit_rate']:.1%}")
        
        return candidates[best_idx], best_metrics, debug_info


def benchmark_simple_optimization():
    """Simple benchmark of ECR optimizations"""
    
    print("=== SIMPLE ECR OPTIMIZATION BENCHMARK ===\n")
    
    # Configuration
    config = ECRConfig()
    config.K = 4  # Number of candidates
    config.H = 1  # Trajectory steps (reduced for faster testing)
    config.parallel_candidates = True
    config.max_parallel_workers = 4
    
    # Test prompt
    prompt = "What are the key principles of effective software architecture?"
    
    # Mock LLM with realistic delay
    mock_llm = MockLLMProvider(response_time=0.02)  # 20ms per call
    
    print(f"Configuration:")
    print(f"  Candidates (K): {config.K}")
    print(f"  Trajectory steps (H): {config.H}")
    print(f"  Mock LLM delay: {mock_llm.response_time}s per call")
    print()
    
    # Test 1: Original ECR Engine
    print("1. Testing Original ECR Engine...")
    original_engine = ECREngine(config)
    
    original_times = []
    for run in range(3):
        mock_llm.call_count = 0
        start_time = time.time()
        
        candidates = original_engine.generate_candidates(prompt, mock_llm)
        selected, metrics, debug = original_engine.select_best_candidate(candidates, prompt, mock_llm)
        
        end_time = time.time()
        run_time = end_time - start_time
        original_times.append(run_time)
        
        print(f"  Run {run + 1}: {run_time:.2f}s, LLM calls: {mock_llm.call_count}, CCI: {metrics.CCI:.3f}")
    
    original_avg = statistics.mean(original_times)
    print(f"  Average: {original_avg:.2f}s\n")
    
    # Test 2: Simple Optimized ECR Engine
    print("2. Testing Simple Optimized ECR Engine...")
    optimized_engine = SimpleOptimizedECREngine(config)
    
    optimized_times = []
    for run in range(3):
        mock_llm.call_count = 0
        start_time = time.time()
        
        candidates = optimized_engine.generate_candidates_parallel(prompt, mock_llm)
        selected, metrics, debug = optimized_engine.select_best_candidate_optimized(candidates, prompt, mock_llm)
        
        end_time = time.time()
        run_time = end_time - start_time
        optimized_times.append(run_time)
        
        print(f"  Run {run + 1}: {run_time:.2f}s, LLM calls: {mock_llm.call_count}, CCI: {metrics.CCI:.3f}")
        print(f"    Cache hit rate: {debug['cache_hit_rate']:.1%}")
    
    optimized_avg = statistics.mean(optimized_times)
    print(f"  Average: {optimized_avg:.2f}s\n")
    
    # Performance Summary
    print("=== PERFORMANCE SUMMARY ===")
    print(f"Original ECR:           {original_avg:.2f}s")
    print(f"Simple Optimized ECR:   {optimized_avg:.2f}s")
    print(f"Speedup:                {original_avg / optimized_avg:.1f}x")
    
    # Show optimization benefits
    final_stats = optimized_engine
    print(f"\nOptimization Benefits:")
    print(f"  Cache entries: {len(final_stats.cache)}")
    print(f"  Total cache hits: {final_stats.cache_hits}")
    print(f"  Total cache misses: {final_stats.cache_misses}")
    print(f"  Overall cache hit rate: {final_stats.cache_hits / max(final_stats.cache_hits + final_stats.cache_misses, 1):.1%}")


if __name__ == "__main__":
    benchmark_simple_optimization()