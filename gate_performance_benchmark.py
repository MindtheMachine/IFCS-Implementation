#!/usr/bin/env python3
"""
Comprehensive performance benchmark for each gate in the IFCS trilogy system
"""

import time
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
from unittest.mock import Mock

from ecr_engine import ECREngine
from control_probe import ControlProbeType1, ControlProbeType2
from ifcs_engine import IFCSEngine
from trilogy_config import TrilogyConfig, ECRConfig, ControlProbeConfig, IFCSConfig


@dataclass
class GatePerformanceMetrics:
    """Performance metrics for a single gate"""
    gate_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    throughput_ops_per_sec: float
    memory_usage_estimate_mb: float
    cpu_intensive: bool
    io_bound: bool


class MockLLMProvider:
    """Mock LLM provider with configurable response times"""
    
    def __init__(self, response_time: float = 0.01):
        self.response_time = response_time
        self.call_count = 0
    
    def __call__(self, prompt: str, temperature=None, max_tokens=None):
        """Simulate LLM call with delay"""
        time.sleep(self.response_time)
        self.call_count += 1
        
        responses = [
            "This is a comprehensive analysis with detailed explanations and recommendations.",
            "Here's my perspective considering multiple viewpoints and potential approaches.",
            "I believe the best approach involves systematic analysis of key components.",
            "The solution requires careful consideration of various factors and constraints.",
            "Based on available information, I recommend a structured methodology."
        ]
        
        return responses[self.call_count % len(responses)]


class GatePerformanceBenchmark:
    """Comprehensive performance benchmark for trilogy gates"""
    
    def __init__(self):
        self.config = TrilogyConfig()
        self.mock_llm = MockLLMProvider(response_time=0.005)  # 5ms per call
        
        # Initialize gates
        self.ecr_engine = ECREngine(self.config.ecr)
        self.control_probe_type1 = ControlProbeType1(self.config.control_probe)
        self.control_probe_type2 = ControlProbeType2(self.config.control_probe)
        self.ifcs_engine = IFCSEngine(self.config.ifcs)
        
        # Test cases with varying complexity
        self.test_cases = [
            {
                'name': 'Simple Query',
                'prompt': 'What is Python?',
                'response': 'Python is a high-level programming language.',
                'complexity': 'low'
            },
            {
                'name': 'Medium Query',
                'prompt': 'What are the best practices for web development?',
                'response': 'Best practices include responsive design, performance optimization, accessibility, security, SEO, modern frameworks, version control, testing, and code quality standards.',
                'complexity': 'medium'
            },
            {
                'name': 'Complex Query',
                'prompt': 'Should I invest in cryptocurrency right now given the current market conditions?',
                'response': 'Cryptocurrency investment involves significant risks and potential rewards. You should definitely invest in Bitcoin as it will certainly increase in value. The market is clearly bullish and you must act now to maximize profits.',
                'complexity': 'high'
            },
            {
                'name': 'High-Risk Query Type 1',
                'prompt': 'I have severe symptoms after recent activity. What could it be?',
                'response': 'Severe symptoms after activity could indicate several conditions. You definitely have a serious condition and should immediately start treatment. This is clearly a critical issue that requires urgent attention.',
                'complexity': 'high',
                'pattern_type': 'high_risk_1'
            },
            {
                'name': 'High-Risk Query Type 2', 
                'prompt': 'Can my employer terminate me for filing a compensation claim?',
                'response': 'Your employer absolutely cannot terminate you for filing compensation claims. This is definitely prohibited and you should immediately take action. The regulations clearly protect you in all cases.',
                'complexity': 'high',
                'pattern_type': 'high_risk_2'
            }
        ]
    
    def benchmark_ecr_gate(self, num_runs: int = 20) -> GatePerformanceMetrics:
        """Benchmark ECR (Evaluative Coherence Regulation) gate"""
        print("🔍 Benchmarking ECR Gate...")
        
        times = []
        
        for run in range(num_runs):
            for test_case in self.test_cases:
                start_time = time.time()
                
                # ECR process: candidate generation + selection
                candidates = self.ecr_engine.generate_candidates(
                    test_case['prompt'], 
                    self.mock_llm,
                    num_candidates=3  # Reduced for faster testing
                )
                
                selected, metrics, debug = self.ecr_engine.select_best_candidate(
                    candidates, 
                    test_case['prompt'], 
                    self.mock_llm
                )
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return GatePerformanceMetrics(
            gate_name="ECR (Evaluative Coherence Regulation)",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            memory_usage_estimate_mb=5.0,  # Trajectory matrices + candidates
            cpu_intensive=True,   # Matrix operations, coherence metrics
            io_bound=True        # LLM calls dominate
        )
    
    def benchmark_control_probe_gate(self, num_runs: int = 50) -> GatePerformanceMetrics:
        """Benchmark Control Probe (Type-1 and Type-2) gate"""
        print("🛡️ Benchmarking Control Probe Gate...")
        
        times = []
        
        for run in range(num_runs):
            for test_case in self.test_cases:
                start_time = time.time()
                
                # Control Probe Type-1: Admissibility check
                sigma_signal = self.control_probe_type1.compute_admissibility_signal(
                    test_case['response'],
                    test_case['prompt']
                )
                
                sigma = sigma_signal.compute_sigma()
                is_admissible = sigma >= self.config.control_probe.tau
                
                # Control Probe Type-2: Interaction monitoring (simplified)
                if is_admissible:
                    interaction_risk = self.control_probe_type2.compute_interaction_risk(
                        test_case['response'],
                        []  # Empty history for testing
                    )
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return GatePerformanceMetrics(
            gate_name="Control Probe (Type-1 & Type-2)",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            memory_usage_estimate_mb=1.0,  # Lightweight pattern matching
            cpu_intensive=False,  # Mostly pattern matching
            io_bound=False       # No external calls
        )
    
    def benchmark_ifcs_gate(self, num_runs: int = 50) -> GatePerformanceMetrics:
        """Benchmark IFCS (Inference-Time Commitment Shaping) gate"""
        print("⚖️ Benchmarking IFCS Gate...")
        
        times = []
        
        for run in range(num_runs):
            for test_case in self.test_cases:
                start_time = time.time()
                
                # IFCS process: commitment risk analysis + shaping
                shaped_response, risk, debug = self.ifcs_engine.shape_commitment(
                    test_case['response'],
                    test_case['prompt'],
                    context="",
                    sigma=0.8  # Assume passed Control Probe
                )
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return GatePerformanceMetrics(
            gate_name="IFCS (Inference-Time Commitment Shaping)",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            memory_usage_estimate_mb=2.0,  # Semantic analysis + transformations
            cpu_intensive=True,   # Semantic analysis, risk computation
            io_bound=False       # No external calls (post-generation)
        )
    
    def benchmark_commitment_actuality_gate(self, num_runs: int = 100) -> GatePerformanceMetrics:
        """Benchmark κ(z*) Commitment-Actuality Gate specifically"""
        print("🎯 Benchmarking κ(z*) Commitment-Actuality Gate...")
        
        times = []
        
        for run in range(num_runs):
            for test_case in self.test_cases:
                start_time = time.time()
                
                # Just the commitment-actuality classification
                is_commitment_bearing = self.ifcs_engine.commitment_classifier.is_commitment_bearing(
                    test_case['response'],
                    test_case['prompt']
                )
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return GatePerformanceMetrics(
            gate_name="κ(z*) Commitment-Actuality Gate",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            memory_usage_estimate_mb=0.5,  # Lightweight pattern analysis
            cpu_intensive=False,  # Simple pattern matching
            io_bound=False       # No external calls
        )
    
    def benchmark_signal_estimation(self, num_runs: int = 100) -> GatePerformanceMetrics:
        """Benchmark Signal Estimation performance"""
        print("🧠 Benchmarking Signal Estimation...")
        
        from signal_estimation import signal_estimator
        
        times = []
        
        for run in range(num_runs):
            for test_case in self.test_cases:
                start_time = time.time()
                
                # All signal estimation components
                scope_result = signal_estimator.estimate_scope_breadth(test_case['response'])
                authority_result = signal_estimator.estimate_authority_posture(test_case['response'])
                evidential_result = signal_estimator.estimate_evidential_risk(test_case['response'])
                temporal_result = signal_estimator.estimate_temporal_risk(test_case['response'], test_case['prompt'])
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        return GatePerformanceMetrics(
            gate_name="Signal Estimation (All Components)",
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            memory_usage_estimate_mb=1.5,  # Pattern dictionaries + analysis
            cpu_intensive=True,   # Text processing, pattern matching
            io_bound=False       # No external calls
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, GatePerformanceMetrics]:
        """Run comprehensive benchmark of all gates"""
        print("🚀 Starting Comprehensive Gate Performance Benchmark\n")
        
        results = {}
        
        # Benchmark each gate
        results['ecr'] = self.benchmark_ecr_gate()
        results['control_probe'] = self.benchmark_control_probe_gate()
        results['ifcs'] = self.benchmark_ifcs_gate()
        results['commitment_actuality'] = self.benchmark_commitment_actuality_gate()
        results['semantic_analyzer'] = self.benchmark_semantic_analyzer()
        
        return results
    
    def print_performance_report(self, results: Dict[str, GatePerformanceMetrics]):
        """Print comprehensive performance report"""
        print("\n" + "="*80)
        print("🎯 TRILOGY SYSTEM GATE PERFORMANCE REPORT")
        print("="*80)
        
        # Sort by average time for ranking
        sorted_gates = sorted(results.items(), key=lambda x: x[1].avg_time_ms)
        
        print(f"\n{'Gate Name':<40} {'Avg Time':<12} {'Throughput':<15} {'Type':<15}")
        print("-" * 80)
        
        for gate_key, metrics in sorted_gates:
            gate_type = "CPU+I/O" if metrics.cpu_intensive and metrics.io_bound else \
                       "CPU" if metrics.cpu_intensive else \
                       "I/O" if metrics.io_bound else "Lightweight"
            
            print(f"{metrics.gate_name:<40} {metrics.avg_time_ms:>8.2f}ms {metrics.throughput_ops_per_sec:>10.1f} ops/s {gate_type:<15}")
        
        print("\n" + "="*80)
        print("📊 DETAILED PERFORMANCE METRICS")
        print("="*80)
        
        for gate_key, metrics in results.items():
            print(f"\n🔹 {metrics.gate_name}")
            print(f"   Average Time:     {metrics.avg_time_ms:.2f}ms")
            print(f"   Min/Max Time:     {metrics.min_time_ms:.2f}ms / {metrics.max_time_ms:.2f}ms")
            print(f"   Standard Dev:     {metrics.std_dev_ms:.2f}ms")
            print(f"   Throughput:       {metrics.throughput_ops_per_sec:.1f} operations/second")
            print(f"   Memory Usage:     ~{metrics.memory_usage_estimate_mb:.1f}MB")
            print(f"   CPU Intensive:    {'Yes' if metrics.cpu_intensive else 'No'}")
            print(f"   I/O Bound:        {'Yes' if metrics.io_bound else 'No'}")
        
        # Performance insights
        print("\n" + "="*80)
        print("💡 PERFORMANCE INSIGHTS")
        print("="*80)
        
        fastest = min(results.values(), key=lambda x: x.avg_time_ms)
        slowest = max(results.values(), key=lambda x: x.avg_time_ms)
        
        print(f"\n🏆 Fastest Gate: {fastest.gate_name}")
        print(f"   {fastest.avg_time_ms:.2f}ms average, {fastest.throughput_ops_per_sec:.1f} ops/sec")
        
        print(f"\n🐌 Slowest Gate: {slowest.gate_name}")
        print(f"   {slowest.avg_time_ms:.2f}ms average, {slowest.throughput_ops_per_sec:.1f} ops/sec")
        
        print(f"\n⚡ Performance Ratio: {slowest.avg_time_ms / fastest.avg_time_ms:.1f}x difference")
        
        # Pipeline analysis
        total_pipeline_time = sum(m.avg_time_ms for m in results.values())
        print(f"\n🔄 Full Pipeline Estimate: {total_pipeline_time:.2f}ms")
        print(f"   Pipeline Throughput: ~{1000/total_pipeline_time:.1f} complete cycles/second")
        
        # Bottleneck analysis
        print(f"\n🎯 Bottleneck Analysis:")
        for gate_key, metrics in sorted(results.items(), key=lambda x: x[1].avg_time_ms, reverse=True):
            percentage = (metrics.avg_time_ms / total_pipeline_time) * 100
            print(f"   {metrics.gate_name}: {percentage:.1f}% of total time")
        
        print("\n" + "="*80)


def main():
    """Run the comprehensive gate performance benchmark"""
    benchmark = GatePerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.print_performance_report(results)


if __name__ == "__main__":
    main()