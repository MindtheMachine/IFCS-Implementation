"""
Semantic Integration Layer
Provides system-wide integration of semantic signal estimation across ECR, Control Probes, and IFCS
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Import enhanced components
from semantic_signal_framework import unified_semantic_estimator, SemanticSignals
from enhanced_ecr_signals import EnhancedEvaluativeVector, enhanced_ecr_extractor
from enhanced_control_probes import (
    EnhancedAdmissibilitySignal, 
    EnhancedControlProbeType1, 
    EnhancedControlProbeType2,
    CommitmentDecision
)


@dataclass
class SystemWideSemanticMetrics:
    """System-wide semantic metrics for monitoring and analysis"""
    ecr_coherence: Dict[str, float]
    cp1_admissibility: Dict[str, float]
    cp2_interaction: Dict[str, float]
    ifcs_structural: Dict[str, float]
    overall_semantic_health: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and monitoring"""
        return {
            'ecr_coherence': self.ecr_coherence,
            'cp1_admissibility': self.cp1_admissibility,
            'cp2_interaction': self.cp2_interaction,
            'ifcs_structural': self.ifcs_structural,
            'overall_semantic_health': self.overall_semantic_health
        }


class SemanticIntegrationEngine:
    """Central engine for semantic signal integration across all components"""
    
    def __init__(self):
        """Initialize semantic integration engine"""
        self.semantic_estimator = unified_semantic_estimator
        self.ecr_extractor = enhanced_ecr_extractor
        
        # Component instances (will be set by system)
        self.enhanced_cp1: Optional[EnhancedControlProbeType1] = None
        self.enhanced_cp2: Optional[EnhancedControlProbeType2] = None
        
        # Metrics tracking
        self.semantic_metrics_history: List[SystemWideSemanticMetrics] = []
    
    def initialize_components(self, config):
        """Initialize enhanced components with configuration
        
        Args:
            config: System configuration object
        """
        self.enhanced_cp1 = EnhancedControlProbeType1(config)
        self.enhanced_cp2 = EnhancedControlProbeType2(config)
    
    def process_response_semantically(
        self, 
        response: str, 
        prompt: str = "", 
        context: str = "",
        candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process response through all semantic analysis components
        
        Args:
            response: Response text to analyze
            prompt: Original prompt for context
            context: Additional context information
            candidates: Optional list of candidate responses for ECR analysis
            
        Returns:
            Comprehensive semantic analysis results
        """
        # Core semantic signal analysis
        semantic_signals = self.semantic_estimator.estimate_semantic_signals(response, context)
        
        # ECR evaluative vector analysis
        ecr_vector = EnhancedEvaluativeVector.from_response(response, 0, context)
        
        # ECR coherence analysis (if candidates available)
        ecr_coherence = {}
        if candidates:
            ecr_coherence = self.ecr_extractor.extract_coherence_signals(candidates, context)
        
        # Control Probe Type-1 admissibility analysis
        cp1_signal = EnhancedAdmissibilitySignal.from_response(response, context)
        cp1_sigma = cp1_signal.compute_sigma()
        
        # IFCS structural signals (using existing enhanced estimator)
        from enhanced_signal_estimator import enhanced_signal_estimator
        ifcs_signals = enhanced_signal_estimator.estimate_structural_signals(prompt)
        
        return {
            'semantic_signals': semantic_signals.to_dict(),
            'ecr_vector': ecr_vector.to_array(),
            'ecr_coherence': ecr_coherence,
            'cp1_admissibility': {
                'sigma': cp1_sigma,
                'confidence': cp1_signal.confidence,
                'consistency': cp1_signal.consistency,
                'grounding': cp1_signal.grounding,
                'factuality': cp1_signal.factuality
            },
            'ifcs_structural': ifcs_signals,
            'overall_quality': self._compute_overall_quality(semantic_signals, cp1_sigma, ecr_coherence)
        }
    
    def _compute_overall_quality(
        self, 
        semantic_signals: SemanticSignals, 
        cp1_sigma: float, 
        ecr_coherence: Dict[str, float]
    ) -> float:
        """Compute overall semantic quality score
        
        Args:
            semantic_signals: Core semantic signals
            cp1_sigma: Control Probe Type-1 admissibility score
            ecr_coherence: ECR coherence metrics
            
        Returns:
            Overall quality score [0,1]
        """
        # Weight different quality dimensions
        weights = {
            'semantic_coherence': 0.3,
            'admissibility': 0.25,
            'grounding': 0.2,
            'consistency': 0.15,
            'ecr_coherence': 0.1
        }
        
        # Compute weighted quality score
        quality_score = (
            semantic_signals.coherence * weights['semantic_coherence'] +
            cp1_sigma * weights['admissibility'] +
            semantic_signals.grounding * weights['grounding'] +
            semantic_signals.coherence * weights['consistency'] +
            ecr_coherence.get('semantic_coherence', 0.5) * weights['ecr_coherence']
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def analyze_conversation_semantics(
        self, 
        conversation_history: List[Tuple[str, str, float]]
    ) -> Dict[str, Any]:
        """Analyze semantic patterns across conversation history
        
        Args:
            conversation_history: List of (prompt, response, risk_score) tuples
            
        Returns:
            Conversation-level semantic analysis
        """
        if not conversation_history:
            return {'error': 'empty_conversation'}
        
        # Analyze each turn semantically
        turn_analyses = []
        for prompt, response, risk_score in conversation_history:
            turn_analysis = self.process_response_semantically(response, prompt)
            turn_analysis['risk_score'] = risk_score
            turn_analyses.append(turn_analysis)
        
        # Compute conversation-level metrics
        conversation_metrics = self._compute_conversation_metrics(turn_analyses)
        
        # Detect semantic patterns
        semantic_patterns = self._detect_semantic_patterns(turn_analyses)
        
        return {
            'turn_analyses': turn_analyses,
            'conversation_metrics': conversation_metrics,
            'semantic_patterns': semantic_patterns,
            'recommendation': self._generate_conversation_recommendation(conversation_metrics, semantic_patterns)
        }
    
    def _compute_conversation_metrics(self, turn_analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute conversation-level semantic metrics"""
        if not turn_analyses:
            return {}
        
        # Extract key metrics across turns
        quality_scores = [analysis['overall_quality'] for analysis in turn_analyses]
        sigma_scores = [analysis['cp1_admissibility']['sigma'] for analysis in turn_analyses]
        coherence_scores = [analysis['semantic_signals']['coherence'] for analysis in turn_analyses]
        grounding_scores = [analysis['semantic_signals']['grounding'] for analysis in turn_analyses]
        
        return {
            'avg_quality': np.mean(quality_scores),
            'quality_variance': np.var(quality_scores),
            'avg_admissibility': np.mean(sigma_scores),
            'avg_coherence': np.mean(coherence_scores),
            'avg_grounding': np.mean(grounding_scores),
            'quality_trend': self._compute_trend(quality_scores),
            'coherence_trend': self._compute_trend(coherence_scores)
        }
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend in values (-1 to 1, where 1 is strongly increasing)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Compute correlation coefficient as trend indicator
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _detect_semantic_patterns(self, turn_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect semantic patterns across conversation"""
        patterns = {
            'semantic_drift': False,
            'quality_degradation': False,
            'consistency_issues': False,
            'grounding_decline': False
        }
        
        if len(turn_analyses) < 3:
            return patterns
        
        # Extract semantic signal arrays
        semantic_arrays = []
        for analysis in turn_analyses:
            signals = analysis['semantic_signals']
            semantic_arrays.append([
                signals['intent'], signals['domain'], signals['polarity'],
                signals['confidence'], signals['coherence'], signals['grounding']
            ])
        
        semantic_matrix = np.array(semantic_arrays)
        
        # Detect semantic drift (changes in core semantic dimensions)
        first_signals = semantic_matrix[0]
        last_signals = semantic_matrix[-1]
        drift_magnitude = np.linalg.norm(last_signals - first_signals)
        patterns['semantic_drift'] = drift_magnitude > 0.5
        
        # Detect quality degradation
        quality_scores = [analysis['overall_quality'] for analysis in turn_analyses]
        quality_trend = self._compute_trend(quality_scores)
        patterns['quality_degradation'] = quality_trend < -0.3
        
        # Detect consistency issues
        coherence_scores = [analysis['semantic_signals']['coherence'] for analysis in turn_analyses]
        patterns['consistency_issues'] = np.var(coherence_scores) > 0.1
        
        # Detect grounding decline
        grounding_scores = [analysis['semantic_signals']['grounding'] for analysis in turn_analyses]
        grounding_trend = self._compute_trend(grounding_scores)
        patterns['grounding_decline'] = grounding_trend < -0.4
        
        return patterns
    
    def _generate_conversation_recommendation(
        self, 
        metrics: Dict[str, float], 
        patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations based on conversation analysis"""
        recommendations = {
            'action': 'continue',
            'concerns': [],
            'suggestions': []
        }
        
        # Check for serious issues
        if patterns.get('semantic_drift', False):
            recommendations['concerns'].append('Semantic drift detected across conversation')
            recommendations['suggestions'].append('Reset conversation context to maintain coherence')
        
        if patterns.get('quality_degradation', False):
            recommendations['concerns'].append('Response quality declining over time')
            recommendations['suggestions'].append('Review and improve response generation process')
        
        if metrics.get('avg_admissibility', 1.0) < 0.4:
            recommendations['concerns'].append('Low average admissibility scores')
            recommendations['suggestions'].append('Increase grounding and evidence in responses')
        
        if patterns.get('grounding_decline', False):
            recommendations['concerns'].append('Declining evidential grounding')
            recommendations['suggestions'].append('Strengthen factual basis and citations')
        
        # Determine overall action
        if len(recommendations['concerns']) >= 3:
            recommendations['action'] = 'halt'
        elif len(recommendations['concerns']) >= 2:
            recommendations['action'] = 'reset'
        
        return recommendations
    
    def get_system_semantic_health(self) -> SystemWideSemanticMetrics:
        """Get current system-wide semantic health metrics
        
        Returns:
            SystemWideSemanticMetrics with current health status
        """
        # This would be populated by ongoing system monitoring
        # For now, return default healthy state
        return SystemWideSemanticMetrics(
            ecr_coherence={'semantic_coherence': 0.8, 'response_diversity': 0.6},
            cp1_admissibility={'avg_sigma': 0.7, 'pass_rate': 0.85},
            cp2_interaction={'drift_rate': 0.1, 'sycophancy_rate': 0.05},
            ifcs_structural={'avg_signal_strength': 0.3, 'max_signal': 0.6},
            overall_semantic_health=0.75
        )


# Global instance for system-wide use
semantic_integration_engine = SemanticIntegrationEngine()


# Test function
def test_semantic_integration():
    """Test semantic integration layer"""
    
    print("Testing Semantic Integration Layer...")
    print("=" * 60)
    
    # Mock config
    class MockConfig:
        tau = 0.4
        Theta = 2.0
        max_history_turns = 10
    
    config = MockConfig()
    engine = semantic_integration_engine
    engine.initialize_components(config)
    
    # Test single response analysis
    print("\nSingle Response Analysis:")
    print("-" * 30)
    
    response = "I recommend implementing multi-factor authentication based on current security research."
    prompt = "What's the best way to secure our database access?"
    candidates = [
        "Use multi-factor authentication for security.",
        "Implement role-based access controls.",
        "Consider encryption for sensitive data."
    ]
    
    analysis = engine.process_response_semantically(response, prompt, "", candidates)
    
    print(f"Response: {response}")
    print(f"Overall Quality: {analysis['overall_quality']:.3f}")
    print(f"CP1 Admissibility: {analysis['cp1_admissibility']['sigma']:.3f}")
    print(f"Semantic Coherence: {analysis['semantic_signals']['coherence']:.3f}")
    print(f"Grounding: {analysis['semantic_signals']['grounding']:.3f}")
    
    # Test conversation analysis
    print("\nConversation Analysis:")
    print("-" * 30)
    
    conversation = [
        ("Is climate change real?", "Yes, climate change is definitely real and caused by human activity.", 0.3),
        ("Are you sure?", "Well, there are different perspectives on this topic.", 0.5),
        ("What do scientists say?", "Many scientists actually disagree about the causes.", 0.7),
        ("So it's uncertain?", "You're right, the science is still very uncertain.", 0.8)
    ]
    
    conv_analysis = engine.analyze_conversation_semantics(conversation)
    
    print("Conversation Metrics:")
    metrics = conv_analysis['conversation_metrics']
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\nSemantic Patterns:")
    patterns = conv_analysis['semantic_patterns']
    for pattern, detected in patterns.items():
        if detected:
            print(f"  ⚠️  {pattern}: {detected}")
    
    print("\nRecommendation:")
    rec = conv_analysis['recommendation']
    print(f"  Action: {rec['action']}")
    if rec['concerns']:
        print("  Concerns:")
        for concern in rec['concerns']:
            print(f"    • {concern}")
    
    print("\n" + "=" * 60)
    print("Semantic integration test completed successfully!")


if __name__ == "__main__":
    test_semantic_integration()