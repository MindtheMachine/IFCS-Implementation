#!/usr/bin/env python3
"""
Simple gate performance benchmark without external dependencies
"""

import time
import statistics
from typing import List, Dict
from dataclasses import dataclass

from ecr_engine import ECREngine, EvaluativeVector, Trajectory
from ifcs_engine import IFCSEngine
from control_probe import ControlProbeType1, ControlProbeType2
from trilogy_config import ECRConfig, IFCSConfig, ControlProbeConfig


@dataclass
class GateMetrics:
    """Simple performance metrics for a gate"""
    name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_per_sec: float


class MockLLM:
    """Mock LLM for testing"""
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, prompt: str, temperature=None, max_tokens=None):
        time.sleep(0.001)  # 1ms simulated delay
        self.call_count += 1
        return f"Mock response {self.call_count} for: {prompt[:30]}..."


def benchmark_gate_performance():
    """Benchmark each gate's performance"""
    
    print("🚀 Simple Gate Performance Benchmark\n")
    
    # Test data
    test_cases = [
        {
            'prompt': 'What is Python?',
            'response': 'Python is a high-level programming language.'
        },
        {
            'prompt': 'What are the best practices for web development?',
            'response': 'Best practices include responsive design, performance optimization, accessibility, security, SEO, modern frameworks, version control, testing, and code quality standards.'
        },
        {
            'prompt': 'Should I invest in cryptocurrency right now?',
            'response': 'You should definitely invest in Bitcoin as it will certainly increase in value. The market is clearly bullish and you must act now to maximize profits.'
        }
    ]
    
    # Initialize components with minimal config
    ecr_config = ECRConfig()
    ecr_config.K = 3
    ecr_config.H = 1
    ecr_config.parallel_candidates = False  # Disable for consistent timing
    
    ifcs_config = IFCSConfig()
    cp_config = ControlProbeConfig()
    
    mock_llm = MockLLM()
    
    # Initialize gates
    ecr_engine = ECREngine(ecr_config)
    ifcs_engine = IFCSEngine(ifcs_config)
    cp_type1 = ControlProbeType1(cp_config)
    cp_type2 = ControlProbeType2(cp_config)
    
    results = {}
    
    # 1. Benchmark ECR Gate
    print("🔍 Benchmarking ECR Gate...")
    ecr_times = []
    
    for _ in range(10):  # Reduced runs for speed
        for test_case in test_cases:
            start_time = time.time()
            
            candidates = ecr_engine.generate_candidates(test_case['prompt'], mock_llm)
            selected, metrics, debug = ecr_engine.select_best_candidate(candidates, test_case['prompt'], mock_llm)
            
            end_time = time.time()
            ecr_times.append((end_time - start_time) * 1000)
    
    results['ECR'] = GateMetrics(
        name="ECR (Evaluative Coherence Regulation)",
        avg_time_ms=statistics.mean(ecr_times),
        min_time_ms=min(ecr_times),
        max_time_ms=max(ecr_times),
        throughput_ops_per_sec=1000 / statistics.mean(ecr_times)
    )
    
    # 2. Benchmark Control Probe Type-1
    print("🛡️ Benchmarking Control Probe Type-1...")
    cp1_times = []
    
    for _ in range(50):
        for test_case in test_cases:
            start_time = time.time()
            
            decision, sigma, debug_info = cp_type1.evaluate(test_case['response'], test_case['prompt'])
            is_admissible = decision.value == 'pass'
            
            end_time = time.time()
            cp1_times.append((end_time - start_time) * 1000)
    
    results['CP-Type1'] = GateMetrics(
        name="Control Probe Type-1 (Admissibility)",
        avg_time_ms=statistics.mean(cp1_times),
        min_time_ms=min(cp1_times),
        max_time_ms=max(cp1_times),
        throughput_ops_per_sec=1000 / statistics.mean(cp1_times)
    )
    
    # 3. Benchmark Control Probe Type-2
    print("🔍 Benchmarking Control Probe Type-2...")
    cp2_times = []
    
    for _ in range(50):
        for test_case in test_cases:
            start_time = time.time()
            
            # Add a turn to history first
            cp_type2.add_turn(test_case['prompt'], test_case['response'], 0.3)
            decision, debug_info = cp_type2.evaluate()
            
            end_time = time.time()
            cp2_times.append((end_time - start_time) * 1000)
    
    results['CP-Type2'] = GateMetrics(
        name="Control Probe Type-2 (Interaction)",
        avg_time_ms=statistics.mean(cp2_times),
        min_time_ms=min(cp2_times),
        max_time_ms=max(cp2_times),
        throughput_ops_per_sec=1000 / statistics.mean(cp2_times)
    )
    
    # 4. Benchmark IFCS Gate
    print("⚖️ Benchmarking IFCS Gate...")
    ifcs_times = []
    
    for _ in range(30):
        for test_case in test_cases:
            start_time = time.time()
            
            shaped_response, risk, debug = ifcs_engine.shape_commitment(
                test_case['response'],
                test_case['prompt'],
                context="",
                sigma=0.8
            )
            
            end_time = time.time()
            ifcs_times.append((end_time - start_time) * 1000)
    
    results['IFCS'] = GateMetrics(
        name="IFCS (Inference-Time Commitment Shaping)",
        avg_time_ms=statistics.mean(ifcs_times),
        min_time_ms=min(ifcs_times),
        max_time_ms=max(ifcs_times),
        throughput_ops_per_sec=1000 / statistics.mean(ifcs_times)
    )
    
    # 5. Benchmark κ(z*) Commitment-Actuality Gate
    print("🎯 Benchmarking κ(z*) Commitment-Actuality Gate...")
    kappa_times = []
    
    for _ in range(100):
        for test_case in test_cases:
            start_time = time.time()
            
            is_commitment_bearing = ifcs_engine.commitment_classifier.is_commitment_bearing(
                test_case['response'],
                test_case['prompt']
            )
            
            end_time = time.time()
            kappa_times.append((end_time - start_time) * 1000)
    
    results['Kappa'] = GateMetrics(
        name="κ(z*) Commitment-Actuality Gate",
        avg_time_ms=statistics.mean(kappa_times),
        min_time_ms=min(kappa_times),
        max_time_ms=max(kappa_times),
        throughput_ops_per_sec=1000 / statistics.mean(kappa_times)
    )
    
    # 6. Benchmark Semantic Analyzer
    print("🧠 Benchmarking Semantic Analyzer...")
    from semantic_analyzer import semantic_analyzer
    
    semantic_times = []
    
    for _ in range(50):
        for test_case in test_cases:
            start_time = time.time()
            
            universal_result = semantic_analyzer.analyze_universal_scope(test_case['response'])
            authority_result = semantic_analyzer.analyze_authority_cues(test_case['response'])
            evidential_result = semantic_analyzer.analyze_evidential_sufficiency(test_case['response'])
            temporal_result = semantic_analyzer.analyze_temporal_risk(test_case['response'], test_case['prompt'])
            
            end_time = time.time()
            semantic_times.append((end_time - start_time) * 1000)
    
    results['Semantic'] = GateMetrics(
        name="Semantic Analyzer (All Components)",
        avg_time_ms=statistics.mean(semantic_times),
        min_time_ms=min(semantic_times),
        max_time_ms=max(semantic_times),
        throughput_ops_per_sec=1000 / statistics.mean(semantic_times)
    )
    
    # Print Results
    print("\n" + "="*80)
    print("🎯 TRILOGY SYSTEM GATE PERFORMANCE RESULTS")
    print("="*80)
    
    # Sort by performance (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1].avg_time_ms)
    
    print(f"\n{'Rank':<5} {'Gate Name':<45} {'Avg Time':<12} {'Throughput':<15}")
    print("-" * 80)
    
    for rank, (key, metrics) in enumerate(sorted_results, 1):
        print(f"{rank:<5} {metrics.name:<45} {metrics.avg_time_ms:>8.3f}ms {metrics.throughput_ops_per_sec:>10.0f} ops/s")
    
    print("\n" + "="*80)
    print("📊 DETAILED PERFORMANCE BREAKDOWN")
    print("="*80)
    
    for key, metrics in sorted_results:
        print(f"\n🔹 {metrics.name}")
        print(f"   Average Time:     {metrics.avg_time_ms:.3f}ms")
        print(f"   Min/Max Time:     {metrics.min_time_ms:.3f}ms / {metrics.max_time_ms:.3f}ms")
        print(f"   Throughput:       {metrics.throughput_ops_per_sec:.0f} operations/second")
    
    # Performance Analysis
    print("\n" + "="*80)
    print("💡 PERFORMANCE ANALYSIS")
    print("="*80)
    
    fastest = min(results.values(), key=lambda x: x.avg_time_ms)
    slowest = max(results.values(), key=lambda x: x.avg_time_ms)
    
    print(f"\n🏆 Fastest Gate: {fastest.name}")
    print(f"   {fastest.avg_time_ms:.3f}ms average ({fastest.throughput_ops_per_sec:.0f} ops/sec)")
    
    print(f"\n🐌 Slowest Gate: {slowest.name}")
    print(f"   {slowest.avg_time_ms:.3f}ms average ({slowest.throughput_ops_per_sec:.0f} ops/sec)")
    
    print(f"\n⚡ Performance Ratio: {slowest.avg_time_ms / fastest.avg_time_ms:.1f}x difference")
    
    # Pipeline Analysis
    total_time = sum(m.avg_time_ms for m in results.values())
    print(f"\n🔄 Full Pipeline Estimate: {total_time:.3f}ms")
    print(f"   Pipeline Throughput: ~{1000/total_time:.0f} complete cycles/second")
    
    # Bottleneck Analysis
    print(f"\n🎯 Bottleneck Analysis:")
    for key, metrics in sorted(results.items(), key=lambda x: x[1].avg_time_ms, reverse=True):
        percentage = (metrics.avg_time_ms / total_time) * 100
        print(f"   {metrics.name}: {percentage:.1f}% of total pipeline time")
    
    print("\n" + "="*80)
    
    return results


if __name__ == "__main__":
    benchmark_gate_performance()