"""
Dashboard Metrics Collection for IFCS Commitment-Actuality Gate
Implements comprehensive metrics collection for monitoring and alerting
"""

import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point with timestamp"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'metadata': self.metadata or {}
        }


@dataclass
class TimeSeriesMetric:
    """Time series metric with windowed aggregation"""
    name: str
    points: deque = None
    window_size: int = 1000  # Maximum points to keep
    
    def __post_init__(self):
        if self.points is None:
            self.points = deque()
    
    def add_point(self, value: float, metadata: Dict[str, Any] = None):
        """Add a new metric point"""
        point = MetricPoint(datetime.now(), value, metadata)
        self.points.append(point)
        
        # Maintain window size
        while len(self.points) > self.window_size:
            self.points.popleft()
    
    def get_recent_points(self, minutes: int = 60) -> List[MetricPoint]:
        """Get points from the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [p for p in self.points if p.timestamp >= cutoff]
    
    def get_aggregated_stats(self, minutes: int = 60) -> Dict[str, float]:
        """Get aggregated statistics for recent points"""
        recent_points = self.get_recent_points(minutes)
        if not recent_points:
            return {'count': 0, 'avg': 0, 'min': 0, 'max': 0, 'sum': 0}
        
        values = [p.value for p in recent_points]
        return {
            'count': len(values),
            'avg': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'sum': sum(values),
            'median': statistics.median(values) if len(values) > 1 else values[0]
        }


@dataclass
class Alert:
    """System alert for monitoring"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    metadata: Dict[str, Any] = None
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'metadata': self.metadata or {},
            'resolved': self.resolved,
            'resolved_timestamp': self.resolved_timestamp.isoformat() if self.resolved_timestamp else None
        }


@dataclass
class DashboardSnapshot:
    """Complete dashboard state snapshot"""
    timestamp: datetime
    commitment_bearing_ratio: float
    avg_kappa_computation_time_ms: float
    total_classifications: int
    recent_latency_improvement_ms: float
    active_alerts: List[Alert]
    performance_status: str
    error_rate: float
    fallback_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'commitment_bearing_ratio': self.commitment_bearing_ratio,
            'avg_kappa_computation_time_ms': self.avg_kappa_computation_time_ms,
            'total_classifications': self.total_classifications,
            'recent_latency_improvement_ms': self.recent_latency_improvement_ms,
            'active_alerts': [alert.to_dict() for alert in self.active_alerts],
            'performance_status': self.performance_status,
            'error_rate': self.error_rate,
            'fallback_rate': self.fallback_rate
        }


class DashboardMetricsCollector:
    """Comprehensive metrics collector for IFCS dashboard reporting"""
    
    def __init__(self, window_minutes: int = 60):
        """Initialize metrics collector
        
        Args:
            window_minutes: Time window for metric aggregation
        """
        self.window_minutes = window_minutes
        self.lock = threading.RLock()
        
        # Time series metrics
        self.metrics = {
            'commitment_bearing_ratio': TimeSeriesMetric('commitment_bearing_ratio'),
            'kappa_computation_time_ms': TimeSeriesMetric('kappa_computation_time_ms'),
            'latency_improvement_ms': TimeSeriesMetric('latency_improvement_ms'),
            'classification_errors': TimeSeriesMetric('classification_errors'),
            'fallback_scenarios': TimeSeriesMetric('fallback_scenarios'),
            'total_classifications': TimeSeriesMetric('total_classifications'),
            'ifcs_processing_time_ms': TimeSeriesMetric('ifcs_processing_time_ms'),
            'non_commitment_bearing_count': TimeSeriesMetric('non_commitment_bearing_count'),
            'commitment_bearing_count': TimeSeriesMetric('commitment_bearing_count')
        }
        
        # Alert management
        self.alerts = {}  # id -> Alert
        self.alert_thresholds = {
            'kappa_computation_time_ms': 50.0,  # Target < 50ms
            'classification_error_rate': 0.05,  # 5% error rate threshold
            'fallback_rate': 0.10,  # 10% fallback rate threshold
            'performance_degradation': 2.0  # 2x performance degradation
        }
        
        # Counters for rate calculations
        self.counters = defaultdict(int)
        self.last_reset = datetime.now()
        
        # Performance baseline tracking
        self.baseline_kappa_time_ms = 1.0  # Expected baseline performance
        
        print("[Dashboard] Metrics collector initialized")
    
    def record_classification(
        self, 
        kappa_value: int, 
        computation_time_ms: float,
        is_error: bool = False,
        is_fallback: bool = False,
        latency_improvement_ms: float = 0.0,
        total_processing_time_ms: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        """Record a -(z*) classification event
        
        Args:
            kappa_value: The -(z*) decision (0 or 1)
            computation_time_ms: Time taken for -(z*) computation
            is_error: Whether this was a classification error
            is_fallback: Whether this used fallback logic
            latency_improvement_ms: Latency improvement from avoided IFCS processing
            total_processing_time_ms: Total IFCS processing time
            metadata: Additional context metadata
        """
        with self.lock:
            # Update counters
            self.counters['total_classifications'] += 1
            if kappa_value == 1:
                self.counters['commitment_bearing'] += 1
            else:
                self.counters['non_commitment_bearing'] += 1
            
            if is_error:
                self.counters['classification_errors'] += 1
            
            if is_fallback:
                self.counters['fallback_scenarios'] += 1
            
            # Calculate current ratios
            total = self.counters['total_classifications']
            commitment_ratio = self.counters['commitment_bearing'] / total if total > 0 else 0
            
            # Record time series metrics
            self.metrics['commitment_bearing_ratio'].add_point(commitment_ratio, metadata)
            self.metrics['kappa_computation_time_ms'].add_point(computation_time_ms, metadata)
            self.metrics['total_classifications'].add_point(total, metadata)
            
            if kappa_value == 1:
                self.metrics['commitment_bearing_count'].add_point(1, metadata)
            else:
                self.metrics['non_commitment_bearing_count'].add_point(1, metadata)
                # Record latency improvement for non-commitment-bearing contexts
                if latency_improvement_ms > 0:
                    self.metrics['latency_improvement_ms'].add_point(latency_improvement_ms, metadata)
            
            if is_error:
                self.metrics['classification_errors'].add_point(1, metadata)
            
            if is_fallback:
                self.metrics['fallback_scenarios'].add_point(1, metadata)
            
            if total_processing_time_ms > 0:
                self.metrics['ifcs_processing_time_ms'].add_point(total_processing_time_ms, metadata)
            
            # Check for alerts
            self._check_performance_alerts(computation_time_ms, metadata)
            self._check_error_rate_alerts()
            self._check_fallback_rate_alerts()
    
    def _check_performance_alerts(self, computation_time_ms: float, metadata: Dict[str, Any] = None):
        """Check for performance-related alerts"""
        # -(z*) computation time alert
        if computation_time_ms > self.alert_thresholds['kappa_computation_time_ms']:
            alert_id = f"kappa_performance_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.WARNING,
                title="-(z*) Computation Performance Degradation",
                message=f"-(z*) computation took {computation_time_ms:.2f}ms, exceeding target of {self.alert_thresholds['kappa_computation_time_ms']}ms",
                timestamp=datetime.now(),
                metric_name='kappa_computation_time_ms',
                metric_value=computation_time_ms,
                threshold=self.alert_thresholds['kappa_computation_time_ms'],
                metadata=metadata
            )
            self.alerts[alert_id] = alert
            print(f"[Dashboard] --  ALERT: {alert.title}")
        
        # Performance degradation alert (compared to baseline)
        if computation_time_ms > self.baseline_kappa_time_ms * self.alert_thresholds['performance_degradation']:
            alert_id = f"performance_degradation_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.ERROR,
                title="Significant Performance Degradation",
                message=f"-(z*) computation time {computation_time_ms:.2f}ms is {computation_time_ms/self.baseline_kappa_time_ms:.1f}x baseline ({self.baseline_kappa_time_ms:.2f}ms)",
                timestamp=datetime.now(),
                metric_name='kappa_computation_time_ms',
                metric_value=computation_time_ms,
                threshold=self.baseline_kappa_time_ms * self.alert_thresholds['performance_degradation'],
                metadata=metadata
            )
            self.alerts[alert_id] = alert
            print(f"[Dashboard] - CRITICAL ALERT: {alert.title}")
    
    def _check_error_rate_alerts(self):
        """Check for classification error rate alerts"""
        total = self.counters['total_classifications']
        if total < 10:  # Need minimum sample size
            return
        
        error_rate = self.counters['classification_errors'] / total
        if error_rate > self.alert_thresholds['classification_error_rate']:
            alert_id = f"error_rate_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.ERROR,
                title="High Classification Error Rate",
                message=f"Classification error rate {error_rate:.1%} exceeds threshold of {self.alert_thresholds['classification_error_rate']:.1%}",
                timestamp=datetime.now(),
                metric_name='classification_error_rate',
                metric_value=error_rate,
                threshold=self.alert_thresholds['classification_error_rate'],
                metadata={'total_classifications': total, 'total_errors': self.counters['classification_errors']}
            )
            self.alerts[alert_id] = alert
            print(f"[Dashboard] - ERROR ALERT: {alert.title}")
    
    def _check_fallback_rate_alerts(self):
        """Check for fallback scenario rate alerts"""
        total = self.counters['total_classifications']
        if total < 10:  # Need minimum sample size
            return
        
        fallback_rate = self.counters['fallback_scenarios'] / total
        if fallback_rate > self.alert_thresholds['fallback_rate']:
            alert_id = f"fallback_rate_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                level=AlertLevel.WARNING,
                title="High Fallback Scenario Rate",
                message=f"Fallback scenario rate {fallback_rate:.1%} exceeds threshold of {self.alert_thresholds['fallback_rate']:.1%}",
                timestamp=datetime.now(),
                metric_name='fallback_rate',
                metric_value=fallback_rate,
                threshold=self.alert_thresholds['fallback_rate'],
                metadata={'total_classifications': total, 'total_fallbacks': self.counters['fallback_scenarios']}
            )
            self.alerts[alert_id] = alert
            print(f"[Dashboard] --  FALLBACK ALERT: {alert.title}")
    
    def get_commitment_bearing_ratio_over_time(self, minutes: int = None) -> List[Dict]:
        """Get commitment-bearing vs non-commitment-bearing ratios over time
        
        Args:
            minutes: Time window in minutes (default: instance window)
            
        Returns:
            List of time series data points
        """
        minutes = minutes or self.window_minutes
        points = self.metrics['commitment_bearing_ratio'].get_recent_points(minutes)
        return [point.to_dict() for point in points]
    
    def get_latency_improvements(self, minutes: int = None) -> Dict[str, Any]:
        """Get latency improvement metrics from avoided IFCS processing
        
        Args:
            minutes: Time window in minutes (default: instance window)
            
        Returns:
            Aggregated latency improvement statistics
        """
        minutes = minutes or self.window_minutes
        stats = self.metrics['latency_improvement_ms'].get_aggregated_stats(minutes)
        
        # Add additional context
        non_commitment_stats = self.metrics['non_commitment_bearing_count'].get_aggregated_stats(minutes)
        total_stats = self.metrics['total_classifications'].get_aggregated_stats(minutes)
        
        return {
            'latency_improvement_stats': stats,
            'non_commitment_bearing_count': non_commitment_stats.get('sum', 0),
            'total_classifications': total_stats.get('sum', 0),
            'non_commitment_bearing_ratio': non_commitment_stats.get('sum', 0) / max(total_stats.get('sum', 1), 1),
            'estimated_total_latency_saved_ms': stats.get('sum', 0)
        }
    
    def get_kappa_performance_metrics(self, minutes: int = None) -> Dict[str, Any]:
        """Get -(z*) computation performance metrics
        
        Args:
            minutes: Time window in minutes (default: instance window)
            
        Returns:
            Performance metrics and target compliance
        """
        minutes = minutes or self.window_minutes
        stats = self.metrics['kappa_computation_time_ms'].get_aggregated_stats(minutes)
        
        target_ms = self.alert_thresholds['kappa_computation_time_ms']
        avg_time = stats.get('avg', 0)
        
        return {
            'performance_stats': stats,
            'target_ms': target_ms,
            'target_compliance': avg_time < target_ms,
            'performance_margin_ms': target_ms - avg_time,
            'performance_ratio': avg_time / target_ms if target_ms > 0 else 0,
            'baseline_comparison': avg_time / self.baseline_kappa_time_ms if self.baseline_kappa_time_ms > 0 else 1
        }
    
    def get_error_and_fallback_metrics(self, minutes: int = None) -> Dict[str, Any]:
        """Get classification error and fallback scenario metrics
        
        Args:
            minutes: Time window in minutes (default: instance window)
            
        Returns:
            Error and fallback statistics
        """
        minutes = minutes or self.window_minutes
        
        error_stats = self.metrics['classification_errors'].get_aggregated_stats(minutes)
        fallback_stats = self.metrics['fallback_scenarios'].get_aggregated_stats(minutes)
        total_stats = self.metrics['total_classifications'].get_aggregated_stats(minutes)
        
        total_count = max(total_stats.get('sum', 1), 1)
        
        return {
            'error_count': error_stats.get('sum', 0),
            'fallback_count': fallback_stats.get('sum', 0),
            'total_classifications': total_count,
            'error_rate': error_stats.get('sum', 0) / total_count,
            'fallback_rate': fallback_stats.get('sum', 0) / total_count,
            'error_threshold': self.alert_thresholds['classification_error_rate'],
            'fallback_threshold': self.alert_thresholds['fallback_rate'],
            'error_compliance': (error_stats.get('sum', 0) / total_count) <= self.alert_thresholds['classification_error_rate'],
            'fallback_compliance': (fallback_stats.get('sum', 0) / total_count) <= self.alert_thresholds['fallback_rate']
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was found and resolved
        """
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_timestamp = datetime.now()
                print(f"[Dashboard] - Alert resolved: {alert_id}")
                return True
            return False
    
    def get_dashboard_snapshot(self) -> DashboardSnapshot:
        """Get complete dashboard state snapshot
        
        Returns:
            Current dashboard state with all key metrics
        """
        with self.lock:
            # Calculate current metrics
            total = self.counters['total_classifications']
            commitment_ratio = self.counters['commitment_bearing'] / total if total > 0 else 0
            
            kappa_perf = self.get_kappa_performance_metrics(minutes=10)  # Last 10 minutes
            latency_metrics = self.get_latency_improvements(minutes=10)
            error_metrics = self.get_error_and_fallback_metrics(minutes=10)
            
            # Determine performance status
            if kappa_perf['target_compliance']:
                if kappa_perf['performance_stats']['avg'] < self.baseline_kappa_time_ms * 1.5:
                    performance_status = "excellent"
                else:
                    performance_status = "good"
            else:
                performance_status = "degraded"
            
            active_alerts = self.get_active_alerts()
            
            return DashboardSnapshot(
                timestamp=datetime.now(),
                commitment_bearing_ratio=commitment_ratio,
                avg_kappa_computation_time_ms=kappa_perf['performance_stats']['avg'],
                total_classifications=total,
                recent_latency_improvement_ms=latency_metrics['latency_improvement_stats']['avg'],
                active_alerts=active_alerts,
                performance_status=performance_status,
                error_rate=error_metrics['error_rate'],
                fallback_rate=error_metrics['fallback_rate']
            )
    
    def export_dashboard_data(self, filepath: str, minutes: int = None) -> None:
        """Export comprehensive dashboard data to JSON
        
        Args:
            filepath: Output file path
            minutes: Time window for data export
        """
        minutes = minutes or self.window_minutes
        
        dashboard_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window_minutes': minutes,
            'snapshot': self.get_dashboard_snapshot().to_dict(),
            'commitment_bearing_ratios': self.get_commitment_bearing_ratio_over_time(minutes),
            'latency_improvements': self.get_latency_improvements(minutes),
            'kappa_performance': self.get_kappa_performance_metrics(minutes),
            'error_and_fallback_metrics': self.get_error_and_fallback_metrics(minutes),
            'all_alerts': [alert.to_dict() for alert in self.alerts.values()],
            'active_alerts': [alert.to_dict() for alert in self.get_active_alerts()],
            'counters': dict(self.counters),
            'alert_thresholds': self.alert_thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"[Dashboard] - Exported dashboard data to {filepath}")
    
    def print_dashboard_summary(self):
        """Print comprehensive dashboard summary"""
        snapshot = self.get_dashboard_snapshot()
        
        print("\n" + "="*80)
        print("- IFCS COMMITMENT-ACTUALITY GATE DASHBOARD SUMMARY")
        print("="*80)
        
        # Key metrics
        print(f"- CLASSIFICATION METRICS:")
        print(f"   Total Classifications: {snapshot.total_classifications}")
        print(f"   Commitment-Bearing Ratio: {snapshot.commitment_bearing_ratio:.1%}")
        print(f"   Non-Commitment-Bearing: {(1-snapshot.commitment_bearing_ratio):.1%}")
        
        # Performance metrics
        print(f"\n- PERFORMANCE METRICS:")
        print(f"   -(z*) Avg Computation Time: {snapshot.avg_kappa_computation_time_ms:.2f}ms")
        print(f"   Performance Status: {snapshot.performance_status.upper()}")
        print(f"   Recent Latency Improvement: {snapshot.recent_latency_improvement_ms:.2f}ms")
        
        # Quality metrics
        print(f"\n- QUALITY METRICS:")
        print(f"   Error Rate: {snapshot.error_rate:.1%}")
        print(f"   Fallback Rate: {snapshot.fallback_rate:.1%}")
        
        # Alerts
        print(f"\n- ALERTS:")
        if snapshot.active_alerts:
            for alert in snapshot.active_alerts:
                print(f"   [{alert.level.value.upper()}] {alert.title}")
                print(f"      {alert.message}")
        else:
            print("   - No active alerts")
        
        print("="*80)
    
    def reset_counters(self):
        """Reset all counters (useful for testing or periodic resets)"""
        with self.lock:
            self.counters.clear()
            self.last_reset = datetime.now()
            print("[Dashboard] - Counters reset")
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds
        
        Args:
            thresholds: Dictionary of threshold updates
        """
        with self.lock:
            self.alert_thresholds.update(thresholds)
            print(f"[Dashboard] --  Alert thresholds updated: {thresholds}")


# Global metrics collector instance
_dashboard_metrics = None

def get_dashboard_metrics() -> DashboardMetricsCollector:
    """Get global dashboard metrics collector instance"""
    global _dashboard_metrics
    if _dashboard_metrics is None:
        _dashboard_metrics = DashboardMetricsCollector()
    return _dashboard_metrics

def initialize_dashboard_metrics(window_minutes: int = 60) -> DashboardMetricsCollector:
    """Initialize global dashboard metrics collector
    
    Args:
        window_minutes: Time window for metric aggregation
        
    Returns:
        Initialized metrics collector
    """
    global _dashboard_metrics
    _dashboard_metrics = DashboardMetricsCollector(window_minutes)
    return _dashboard_metrics