#!/usr/bin/env python3
"""
Dashboard Metrics Testing and Demonstration
Tests the comprehensive metrics collection system for IFCS dashboard reporting
"""

import time
import json
from typing import List, Dict
from ifcs_engine import IFCSEngine
from trilogy_config import IFCSConfig
from dashboard_metrics import initialize_dashboard_metrics, get_dashboard_metrics


def create_test_ifcs_engine() -> IFCSEngine:
    """Create IFCS engine for testing"""
    config = IFCSConfig()
    config.rho = 0.40
    config.lambda_e = 0.3
    config.lambda_s = 0.25
    config.lambda_a = 0.25
    config.lambda_t = 0.2
    config.delta_AG_threshold = 0.15
    
    return IFCSEngine(config)


def test_commitment_bearing_scenarios(ifcs_engine: IFCSEngine) -> None:
    """Test commitment-bearing scenarios"""
    print("\n🎯 TESTING COMMITMENT-BEARING SCENARIOS")
    print("="*60)
    
    commitment_bearing_cases = [
        {
            'prompt': 'What should I do about my anxiety?',
            'response': 'You should definitely try meditation and exercise. These are the best approaches for managing anxiety.',
            'expected_kappa': 1
        },
        {
            'prompt': 'How do I invest my money?',
            'response': 'You must invest in index funds. This is the only safe way to build wealth.',
            'expected_kappa': 1
        },
        {
            'prompt': 'What programming language should I learn?',
            'response': 'Python is clearly the best choice for beginners. You should start with Python immediately.',
            'expected_kappa': 1
        }
    ]
    
    for i, case in enumerate(commitment_bearing_cases, 1):
        print(f"\n[Test {i}] Commitment-Bearing Case:")
        print(f"Prompt: {case['prompt']}")
        print(f"Response: {case['response'][:100]}...")
        
        start_time = time.perf_counter()
        shaped_response, risk, debug_info = ifcs_engine.shape_commitment(
            case['response'], 
            case['prompt']
        )
        processing_time = (time.perf_counter() - start_time) * 1000
        
        print(f"κ(z*): {debug_info['kappa']} (expected: {case['expected_kappa']})")
        print(f"Processing time: {processing_time:.2f}ms")
        print(f"Intervened: {debug_info['intervened']}")
        
        if debug_info['kappa'] == case['expected_kappa']:
            print("✅ Classification correct")
        else:
            print("❌ Classification incorrect")


def test_non_commitment_bearing_scenarios(ifcs_engine: IFCSEngine) -> None:
    """Test non-commitment-bearing scenarios"""
    print("\n📋 TESTING NON-COMMITMENT-BEARING SCENARIOS")
    print("="*60)
    
    non_commitment_cases = [
        {
            'prompt': 'What are the current best practices for web development?',
            'response': 'Current web development practices include using React, Vue, or Angular for frontend development. Popular backend frameworks include Node.js, Django, and Ruby on Rails.',
            'expected_kappa': 0
        },
        {
            'prompt': 'Tell me about different programming languages',
            'response': 'Programming languages vary widely in their design and use cases. Python is often used for data science, JavaScript for web development, and C++ for system programming.',
            'expected_kappa': 0
        },
        {
            'prompt': 'What are some investment options?',
            'response': 'Investment options include stocks, bonds, mutual funds, ETFs, real estate, and commodities. Each has different risk profiles and potential returns.',
            'expected_kappa': 0
        }
    ]
    
    for i, case in enumerate(non_commitment_cases, 1):
        print(f"\n[Test {i}] Non-Commitment-Bearing Case:")
        print(f"Prompt: {case['prompt']}")
        print(f"Response: {case['response'][:100]}...")
        
        start_time = time.perf_counter()
        shaped_response, risk, debug_info = ifcs_engine.shape_commitment(
            case['response'], 
            case['prompt']
        )
        processing_time = (time.perf_counter() - start_time) * 1000
        
        print(f"κ(z*): {debug_info['kappa']} (expected: {case['expected_kappa']})")
        print(f"Processing time: {processing_time:.2f}ms")
        print(f"Latency improvement: {debug_info.get('latency_improvement_ms', 0):.2f}ms")
        
        if debug_info['kappa'] == case['expected_kappa']:
            print("✅ Classification correct")
        else:
            print("❌ Classification incorrect")


def test_performance_alerts(ifcs_engine: IFCSEngine) -> None:
    """Test performance alert generation"""
    print("\n⚠️  TESTING PERFORMANCE ALERTS")
    print("="*60)
    
    dashboard_metrics = get_dashboard_metrics()
    
    # Simulate slow κ(z*) computation to trigger alert
    print("\n[Test] Simulating slow κ(z*) computation...")
    dashboard_metrics.record_classification(
        kappa_value=1,
        computation_time_ms=75.0,  # Exceeds 50ms threshold
        is_error=False,
        is_fallback=False,
        metadata={'test': 'performance_alert_simulation'}
    )
    
    # Check for alerts
    active_alerts = dashboard_metrics.get_active_alerts()
    print(f"Active alerts after slow computation: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"  🚨 {alert.level.value.upper()}: {alert.title}")
        print(f"     {alert.message}")


def test_error_and_fallback_scenarios(ifcs_engine: IFCSEngine) -> None:
    """Test error and fallback scenario tracking"""
    print("\n🚨 TESTING ERROR AND FALLBACK SCENARIOS")
    print("="*60)
    
    dashboard_metrics = get_dashboard_metrics()
    
    # Simulate classification errors
    print("\n[Test] Simulating classification errors...")
    for i in range(3):
        dashboard_metrics.record_classification(
            kappa_value=0,
            computation_time_ms=1.5,
            is_error=True,  # Simulate error
            is_fallback=False,
            metadata={'test': f'error_simulation_{i}'}
        )
    
    # Simulate fallback scenarios
    print("\n[Test] Simulating fallback scenarios...")
    for i in range(2):
        dashboard_metrics.record_classification(
            kappa_value=0,
            computation_time_ms=0.8,
            is_error=False,
            is_fallback=True,  # Simulate fallback
            metadata={'test': f'fallback_simulation_{i}'}
        )
    
    # Check error and fallback metrics
    error_metrics = dashboard_metrics.get_error_and_fallback_metrics()
    print(f"\nError rate: {error_metrics['error_rate']:.1%}")
    print(f"Fallback rate: {error_metrics['fallback_rate']:.1%}")
    print(f"Error compliance: {error_metrics['error_compliance']}")
    print(f"Fallback compliance: {error_metrics['fallback_compliance']}")


def test_dashboard_snapshot(ifcs_engine: IFCSEngine) -> None:
    """Test dashboard snapshot generation"""
    print("\n📊 TESTING DASHBOARD SNAPSHOT")
    print("="*60)
    
    dashboard_metrics = get_dashboard_metrics()
    snapshot = dashboard_metrics.get_dashboard_snapshot()
    
    print(f"Timestamp: {snapshot.timestamp}")
    print(f"Total classifications: {snapshot.total_classifications}")
    print(f"Commitment-bearing ratio: {snapshot.commitment_bearing_ratio:.1%}")
    print(f"Avg κ(z*) computation time: {snapshot.avg_kappa_computation_time_ms:.2f}ms")
    print(f"Performance status: {snapshot.performance_status}")
    print(f"Error rate: {snapshot.error_rate:.1%}")
    print(f"Fallback rate: {snapshot.fallback_rate:.1%}")
    print(f"Active alerts: {len(snapshot.active_alerts)}")


def test_time_series_metrics(ifcs_engine: IFCSEngine) -> None:
    """Test time series metrics collection"""
    print("\n📈 TESTING TIME SERIES METRICS")
    print("="*60)
    
    dashboard_metrics = get_dashboard_metrics()
    
    # Get commitment-bearing ratios over time
    ratios = dashboard_metrics.get_commitment_bearing_ratio_over_time(minutes=60)
    print(f"Commitment-bearing ratio data points: {len(ratios)}")
    
    if ratios:
        latest = ratios[-1]
        print(f"Latest ratio: {latest['value']:.1%} at {latest['timestamp']}")
    
    # Get latency improvements
    latency_metrics = dashboard_metrics.get_latency_improvements()
    print(f"Total latency saved: {latency_metrics['estimated_total_latency_saved_ms']:.2f}ms")
    print(f"Non-commitment-bearing ratio: {latency_metrics['non_commitment_bearing_ratio']:.1%}")
    
    # Get κ(z*) performance metrics
    kappa_perf = dashboard_metrics.get_kappa_performance_metrics()
    print(f"κ(z*) target compliance: {kappa_perf['target_compliance']}")
    print(f"Performance margin: {kappa_perf['performance_margin_ms']:.2f}ms")


def test_data_export(ifcs_engine: IFCSEngine) -> None:
    """Test data export functionality"""
    print("\n💾 TESTING DATA EXPORT")
    print("="*60)
    
    # Export dashboard data
    dashboard_filepath = "dashboard_metrics_test.json"
    ifcs_engine.export_dashboard_data(dashboard_filepath, minutes=60)
    
    # Verify export
    try:
        with open(dashboard_filepath, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Dashboard data exported successfully")
        print(f"   Export timestamp: {data['export_timestamp']}")
        print(f"   Time window: {data['time_window_minutes']} minutes")
        print(f"   Total classifications: {data['snapshot']['total_classifications']}")
        print(f"   Active alerts: {len(data['active_alerts'])}")
        
    except Exception as e:
        print(f"❌ Export verification failed: {e}")


def main():
    """Main test execution"""
    print("🎯 IFCS DASHBOARD METRICS TESTING")
    print("="*80)
    
    # Initialize dashboard metrics
    dashboard_metrics = initialize_dashboard_metrics(window_minutes=60)
    print("✅ Dashboard metrics collector initialized")
    
    # Create IFCS engine
    ifcs_engine = create_test_ifcs_engine()
    print("✅ IFCS engine created")
    
    # Run test scenarios
    test_commitment_bearing_scenarios(ifcs_engine)
    test_non_commitment_bearing_scenarios(ifcs_engine)
    test_performance_alerts(ifcs_engine)
    test_error_and_fallback_scenarios(ifcs_engine)
    test_dashboard_snapshot(ifcs_engine)
    test_time_series_metrics(ifcs_engine)
    test_data_export(ifcs_engine)
    
    # Print comprehensive dashboard summary
    print("\n📊 COMPREHENSIVE DASHBOARD SUMMARY")
    print("="*80)
    ifcs_engine.print_dashboard_summary()
    
    print("\n✅ DASHBOARD METRICS TESTING COMPLETE")
    print("="*80)
    
    # Final metrics summary
    dashboard_metrics = get_dashboard_metrics()
    final_snapshot = dashboard_metrics.get_dashboard_snapshot()
    
    print(f"📈 FINAL METRICS SUMMARY:")
    print(f"   Total Classifications: {final_snapshot.total_classifications}")
    print(f"   Commitment-Bearing: {final_snapshot.commitment_bearing_ratio:.1%}")
    print(f"   Performance Status: {final_snapshot.performance_status.upper()}")
    print(f"   Active Alerts: {len(final_snapshot.active_alerts)}")
    
    if final_snapshot.avg_kappa_computation_time_ms < 50:
        print(f"   ✅ Performance Target Met: {final_snapshot.avg_kappa_computation_time_ms:.2f}ms < 50ms")
    else:
        print(f"   ⚠️  Performance Target Missed: {final_snapshot.avg_kappa_computation_time_ms:.2f}ms > 50ms")


if __name__ == "__main__":
    main()