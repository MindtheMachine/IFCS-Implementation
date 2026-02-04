"""
Enhanced ECR Evaluative Vectors using Semantic Signal Estimation
Replaces heuristic keyword counting with semantic similarity and fuzzy logic
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from semantic_signal_framework import unified_semantic_estimator, SemanticSignals


@dataclass
class EnhancedEvaluativeVector:
    """Enhanced evaluative vector using semantic signals"""
    confidence: float      # Epistemic certainty from semantic analysis
    retrieval: float       # Semantic grounding and context alignment  
    uncertainty: float     # Inverse of confidence with disagreement signals
    safety: float          # Domain-aware risk assessment
    consistency: float     # Coherence and logical flow
    
    # Additional semantic dimensions
    intent_clarity: float  # Intent clarity and directness
    domain_expertise: float # Domain specificity and expertise level
    polarity_strength: float # Stance and sentiment strength
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations"""
        return np.array([
            self.confidence, self.retrieval, self.uncertainty, self.safety, self.consistency,
            self.intent_clarity, self.domain_expertise, self.polarity_strength
        ])
    
    @classmethod
    def from_semantic_signals(cls, signals: SemanticSignals, context: str = "") -> 'EnhancedEvaluativeVector':
        """Create evaluative vector from semantic signals
        
        Args:
            signals: SemanticSignals object
            context: Optional context for additional analysis
            
        Returns:
            EnhancedEvaluativeVector with semantic-based values
        """
        # Core evaluative dimensions
        confidence = signals.confidence
        retrieval = signals.grounding  # Grounding represents retrieval quality
        uncertainty = max(0.0, 1.0 - confidence + signals.disagreement * 0.3)  # Uncertainty with disagreement
        
        # Safety assessment based on domain and authority
        safety = cls._compute_semantic_safety(signals)
        
        # Consistency from coherence signals
        consistency = signals.coherence
        
        # Additional semantic dimensions
        intent_clarity = signals.intent
        domain_expertise = signals.domain
        polarity_strength = signals.polarity
        
        return cls(
            confidence=confidence,
            retrieval=retrieval,
            uncertainty=min(1.0, uncertainty),  # Clamp to [0,1]
            safety=safety,
            consistency=consistency,
            intent_clarity=intent_clarity,
            domain_expertise=domain_expertise,
            polarity_strength=polarity_strength
        )
    
    @classmethod
    def from_response(cls, response: str, step: int, context: str = "") -> 'EnhancedEvaluativeVector':
        """Create evaluative vector from response text using semantic analysis
        
        Args:
            response: Response text to analyze
            step: Step number in trajectory (for future use)
            context: Context for grounding analysis
            
        Returns:
            EnhancedEvaluativeVector with semantic-based evaluation
        """
        # Get comprehensive semantic signals
        signals = unified_semantic_estimator.estimate_semantic_signals(response, context)
        
        # Create evaluative vector from signals
        return cls.from_semantic_signals(signals, context)
    
    @staticmethod
    def _compute_semantic_safety(signals: SemanticSignals) -> float:
        """Compute safety score based on semantic signals
        
        Args:
            signals: SemanticSignals object
            
        Returns:
            Safety score [0,1] where 1 is safest
        """
        # Base safety score
        base_safety = 0.8
        
        # Reduce safety for high authority without grounding (potential fabrication)
        if signals.authority > 0.7 and signals.grounding < 0.4:
            base_safety -= 0.3
        
        # Reduce safety for high confidence with disagreement (potential overconfidence)
        if signals.confidence > 0.8 and signals.disagreement > 0.5:
            base_safety -= 0.2
        
        # Reduce safety for domain-specific claims without expertise markers
        if signals.domain > 0.6 and signals.grounding < 0.5:
            base_safety -= 0.2
        
        # Boost safety for well-grounded, coherent responses
        if signals.grounding > 0.7 and signals.coherence > 0.7:
            base_safety += 0.1
        
        return max(0.0, min(1.0, base_safety))


class EnhancedECRSignalExtractor:
    """Enhanced ECR signal extraction using semantic similarity"""
    
    def __init__(self):
        """Initialize enhanced ECR signal extractor"""
        self.similarity_engine = unified_semantic_estimator.similarity_engine
    
    def extract_coherence_signals(self, candidates: List[str], context: str = "") -> Dict[str, float]:
        """Extract ECR coherence signals using semantic similarity
        
        Args:
            candidates: List of candidate responses
            context: Optional context for analysis
            
        Returns:
            Dictionary of coherence signal values
        """
        if not candidates:
            return {'semantic_coherence': 0.0, 'response_diversity': 0.0, 'consensus_strength': 0.0}
        
        # Get semantic signals for all candidates
        candidate_signals = []
        for candidate in candidates:
            signals = unified_semantic_estimator.estimate_semantic_signals(candidate, context)
            candidate_signals.append(signals)
        
        # Compute semantic coherence (average coherence across candidates)
        semantic_coherence = np.mean([signals.coherence for signals in candidate_signals])
        
        # Compute response diversity using semantic similarity
        response_diversity = self._compute_response_diversity(candidates)
        
        # Compute consensus strength (agreement in semantic signals)
        consensus_strength = self._compute_consensus_strength(candidate_signals)
        
        return {
            'semantic_coherence': semantic_coherence,
            'response_diversity': response_diversity,
            'consensus_strength': consensus_strength
        }
    
    def _compute_response_diversity(self, candidates: List[str]) -> float:
        """Compute response diversity using semantic similarity
        
        Args:
            candidates: List of candidate responses
            
        Returns:
            Diversity score [0,1] where 1 is most diverse
        """
        if len(candidates) < 2:
            return 0.0
        
        # Compute pairwise semantic similarities
        similarities = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                similarity = self.similarity_engine.compute_semantic_similarity(
                    candidates[i], candidates[j]
                )
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def _compute_consensus_strength(self, candidate_signals: List[SemanticSignals]) -> float:
        """Compute consensus strength across candidate semantic signals
        
        Args:
            candidate_signals: List of SemanticSignals for each candidate
            
        Returns:
            Consensus strength [0,1] where 1 is strongest consensus
        """
        if len(candidate_signals) < 2:
            return 1.0  # Single candidate has perfect consensus
        
        # Compute variance across key semantic dimensions
        signal_arrays = [signals.to_array() for signals in candidate_signals]
        signal_matrix = np.vstack(signal_arrays)
        
        # Compute coefficient of variation for each dimension
        means = np.mean(signal_matrix, axis=0)
        stds = np.std(signal_matrix, axis=0)
        
        # Avoid division by zero
        cvs = []
        for mean, std in zip(means, stds):
            if mean > 0.1:  # Only consider dimensions with meaningful values
                cv = std / mean
                cvs.append(cv)
        
        if not cvs:
            return 1.0
        
        # Consensus is inverse of average coefficient of variation
        avg_cv = np.mean(cvs)
        consensus = 1.0 / (1.0 + avg_cv)  # Transform to [0,1] range
        
        return max(0.0, min(1.0, consensus))
    
    def compute_trajectory_coherence(self, trajectory_vectors: List[EnhancedEvaluativeVector]) -> Dict[str, float]:
        """Compute trajectory-level coherence metrics using semantic signals
        
        Args:
            trajectory_vectors: List of EnhancedEvaluativeVector objects
            
        Returns:
            Dictionary of trajectory coherence metrics
        """
        if len(trajectory_vectors) < 2:
            return {
                'trajectory_smoothness': 1.0,
                'semantic_drift': 0.0,
                'consistency_evolution': 1.0
            }
        
        # Convert to matrix for analysis
        vector_matrix = np.vstack([vec.to_array() for vec in trajectory_vectors])
        
        # Compute trajectory smoothness (low variance in changes)
        trajectory_smoothness = self._compute_trajectory_smoothness(vector_matrix)
        
        # Compute semantic drift (changes in semantic dimensions)
        semantic_drift = self._compute_semantic_drift(vector_matrix)
        
        # Compute consistency evolution (how consistency changes over time)
        consistency_evolution = self._compute_consistency_evolution(trajectory_vectors)
        
        return {
            'trajectory_smoothness': trajectory_smoothness,
            'semantic_drift': semantic_drift,
            'consistency_evolution': consistency_evolution
        }
    
    def _compute_trajectory_smoothness(self, vector_matrix: np.ndarray) -> float:
        """Compute trajectory smoothness from vector matrix"""
        if vector_matrix.shape[0] < 2:
            return 1.0
        
        # Compute differences between consecutive vectors
        diffs = np.diff(vector_matrix, axis=0)
        
        # Compute average magnitude of changes
        change_magnitudes = np.linalg.norm(diffs, axis=1)
        avg_change = np.mean(change_magnitudes)
        
        # Smoothness is inverse of average change (normalized)
        smoothness = 1.0 / (1.0 + avg_change)
        
        return max(0.0, min(1.0, smoothness))
    
    def _compute_semantic_drift(self, vector_matrix: np.ndarray) -> float:
        """Compute semantic drift across trajectory"""
        if vector_matrix.shape[0] < 2:
            return 0.0
        
        # Focus on semantic dimensions (intent, domain, polarity)
        semantic_dims = vector_matrix[:, [5, 6, 7]]  # intent_clarity, domain_expertise, polarity_strength
        
        # Compute cumulative drift
        first_vector = semantic_dims[0]
        last_vector = semantic_dims[-1]
        
        drift_magnitude = np.linalg.norm(last_vector - first_vector)
        
        # Normalize by maximum possible drift
        max_drift = np.sqrt(3)  # Maximum L2 distance in 3D unit cube
        semantic_drift = drift_magnitude / max_drift
        
        return max(0.0, min(1.0, semantic_drift))
    
    def _compute_consistency_evolution(self, trajectory_vectors: List[EnhancedEvaluativeVector]) -> float:
        """Compute how consistency evolves over trajectory"""
        consistency_values = [vec.consistency for vec in trajectory_vectors]
        
        if len(consistency_values) < 2:
            return 1.0
        
        # Compute variance in consistency
        consistency_variance = np.var(consistency_values)
        
        # Evolution score is inverse of variance (stable consistency is good)
        evolution_score = 1.0 / (1.0 + consistency_variance * 4.0)  # Scale variance
        
        return max(0.0, min(1.0, evolution_score))


# Global instance for system-wide use
enhanced_ecr_extractor = EnhancedECRSignalExtractor()


# Test function
def test_enhanced_ecr_signals():
    """Test enhanced ECR signal extraction"""
    
    # Test candidates
    candidates = [
        "I recommend using the secure authentication protocol for database access.",
        "The best approach is to implement multi-factor authentication for security.",
        "You should definitely use encrypted connections when accessing sensitive data.",
        "Consider implementing role-based access controls for better security."
    ]
    
    context = "Database security implementation for enterprise application"
    
    print("Testing Enhanced ECR Signal Extraction...")
    print("=" * 60)
    
    # Test evaluative vectors
    print("\nEvaluative Vectors:")
    print("-" * 30)
    
    vectors = []
    for i, candidate in enumerate(candidates):
        vector = EnhancedEvaluativeVector.from_response(candidate, i, context)
        vectors.append(vector)
        
        print(f"Candidate {i+1}: {candidate[:50]}...")
        print(f"  Confidence: {vector.confidence:.3f}")
        print(f"  Retrieval: {vector.retrieval:.3f}")
        print(f"  Safety: {vector.safety:.3f}")
        print(f"  Intent Clarity: {vector.intent_clarity:.3f}")
        print(f"  Domain Expertise: {vector.domain_expertise:.3f}")
        print()
    
    # Test coherence signals
    print("Coherence Signals:")
    print("-" * 30)
    
    coherence_signals = enhanced_ecr_extractor.extract_coherence_signals(candidates, context)
    for signal_name, value in coherence_signals.items():
        print(f"  {signal_name}: {value:.3f}")
    
    # Test trajectory coherence
    print("\nTrajectory Coherence:")
    print("-" * 30)
    
    trajectory_coherence = enhanced_ecr_extractor.compute_trajectory_coherence(vectors)
    for metric_name, value in trajectory_coherence.items():
        print(f"  {metric_name}: {value:.3f}")
    
    print("\n" + "=" * 60)
    print("Enhanced ECR signal extraction test completed successfully!")


if __name__ == "__main__":
    test_enhanced_ecr_signals()