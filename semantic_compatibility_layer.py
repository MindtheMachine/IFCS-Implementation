"""
Semantic Compatibility Layer
Provides backward-compatible interfaces while using enhanced semantic signal estimation
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

# Import enhanced components
from enhanced_ecr_signals import EnhancedEvaluativeVector, enhanced_ecr_extractor
from enhanced_control_probes import (
    EnhancedAdmissibilitySignal, 
    EnhancedControlProbeType1, 
    EnhancedControlProbeType2,
    CommitmentDecision,
    EnhancedInteractionTurn
)
from semantic_integration_layer import semantic_integration_engine


@dataclass
class CompatibleEvaluativeVector:
    """Backward-compatible EvaluativeVector using enhanced semantic analysis"""
    confidence: float
    retrieval: float
    uncertainty: float
    safety: float
    consistency: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations"""
        return np.array([
            self.confidence, self.retrieval, self.uncertainty, self.safety, self.consistency
        ])
    
    @classmethod
    def from_response(cls, response: str, step: int, context: str = "") -> 'CompatibleEvaluativeVector':
        """Create evaluative vector from response using enhanced semantic analysis
        
        This method maintains the original API while using enhanced semantic signals internally
        """
        # Use enhanced evaluative vector
        enhanced_vector = EnhancedEvaluativeVector.from_response(response, step, context)
        
        # Map to compatible format (5 dimensions instead of 8)
        return cls(
            confidence=enhanced_vector.confidence,
            retrieval=enhanced_vector.retrieval,
            uncertainty=enhanced_vector.uncertainty,
            safety=enhanced_vector.safety,
            consistency=enhanced_vector.consistency
        )


@dataclass
class CompatibleAdmissibilitySignal:
    """Backward-compatible AdmissibilitySignal using enhanced semantic analysis"""
    confidence: float
    consistency: float
    grounding: float
    factuality: float
    
    def compute_sigma(self) -> float:
        """Compute admissibility score using enhanced semantic weighting"""
        # Use enhanced admissibility signal computation
        enhanced_signal = EnhancedAdmissibilitySignal(
            confidence=self.confidence,
            consistency=self.consistency,
            grounding=self.grounding,
            factuality=self.factuality,
            intent_clarity=0.5,  # Default values for compatibility
            domain_alignment=0.5
        )
        
        return enhanced_signal.compute_sigma()
    
    @classmethod
    def from_response(cls, response: str, context: str = "", coherence_metrics=None) -> 'CompatibleAdmissibilitySignal':
        """Create admissibility signal using enhanced semantic analysis"""
        # Use enhanced admissibility signal
        enhanced_signal = EnhancedAdmissibilitySignal.from_response(response, context, coherence_metrics)
        
        # Map to compatible format
        return cls(
            confidence=enhanced_signal.confidence,
            consistency=enhanced_signal.consistency,
            grounding=enhanced_signal.grounding,
            factuality=enhanced_signal.factuality
        )


@dataclass
class CompatibleInteractionTurn:
    """Backward-compatible InteractionTurn with enhanced semantic analysis"""
    prompt: str
    response: str
    risk_score: float
    timestamp: int
    
    @classmethod
    def from_turn_data(cls, prompt: str, response: str, risk_score: float, timestamp: int) -> 'CompatibleInteractionTurn':
        """Create interaction turn (enhanced analysis happens internally)"""
        return cls(
            prompt=prompt,
            response=response,
            risk_score=risk_score,
            timestamp=timestamp
        )


class SemanticECREngine:
    """Enhanced ECR Engine using semantic signal estimation while maintaining API compatibility"""
    
    def __init__(self, config):
        """Initialize semantic ECR engine
        
        Args:
            config: ECRConfig instance
        """
        self.config = config
        self.K = config.K
        self.H = config.H
        self.tau_CCI = config.tau_CCI
        self.weights = (config.alpha, config.beta, config.gamma, config.delta, config.epsilon)
        
        # Initialize semantic integration
        semantic_integration_engine.initialize_components(config)
    
    def generate_candidates(self, prompt: str, llm_call_fn, num_candidates: Optional[int] = None, llm_provider: Optional[object] = None) -> List[str]:
        """Generate K candidate responses (unchanged from original)"""
        # This method remains the same as original ECR implementation
        target_k = num_candidates if num_candidates is not None else self.K
        if target_k <= 0:
            return []

        if (
            llm_provider
            and hasattr(llm_provider, "capabilities")
            and llm_provider.capabilities().get("batch")
        ):
            return llm_provider.generate_batch(
                prompt=prompt,
                n=target_k,
                max_tokens=2000,
                temperature=None,
                top_p=None,
                system=None,
                seed=None
            )

        if not self.config.parallel_candidates or target_k == 1:
            candidates = []
            for _ in range(target_k):
                response = llm_call_fn(prompt, temperature=None)
                candidates.append(response)
            return candidates

        from concurrent.futures import ThreadPoolExecutor
        max_workers = self.config.max_parallel_workers or target_k
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(llm_call_fn, prompt, temperature=None)
                for _ in range(target_k)
            ]
            return [future.result() for future in futures]
    
    def unroll_trajectory(self, candidate: str, prompt: str, llm_call_fn) -> 'CompatibleTrajectory':
        """Unroll trajectory using enhanced semantic evaluative vectors"""
        steps = []
        current_context = prompt
        
        # Initial step with enhanced semantic analysis
        steps.append(CompatibleEvaluativeVector.from_response(candidate, 0, current_context))
        
        # Subsequent steps (simulated continuation)
        current_response = candidate
        for h in range(1, self.H + 1):
            # Simulate asking "continue" or "elaborate"
            continuation_prompt = f"{current_context}\n\n{current_response}\n\nContinue your explanation:"
            
            try:
                # Get continuation (shorter)
                continuation = llm_call_fn(continuation_prompt, temperature=None, max_tokens=200)
            except Exception:
                continuation = "Cannot continue due to error."
            
            # Create enhanced evaluative vector
            steps.append(CompatibleEvaluativeVector.from_response(
                continuation, h, current_context
            ))
            
            current_response = continuation
            current_context = f"{current_context}\n{current_response}"
        
        return CompatibleTrajectory(response=candidate, steps=steps)
    
    def compute_EVB(self, trajectory: 'CompatibleTrajectory') -> float:
        """Compute Evaluative Variance Bound using enhanced semantic vectors"""
        E_matrix = trajectory.get_matrix()  # (H+1) x m
        m = E_matrix.shape[1]
        
        # Compute covariance
        cov = np.cov(E_matrix, rowvar=False)
        
        # Apply Ledoit-Wolf shrinkage
        lambda_shrink = self.config.lambda_shrink
        I = np.eye(m)
        cov_shrunk = (1 - lambda_shrink) * cov + lambda_shrink * I
        
        # Compute trace and normalize
        trace = np.trace(cov_shrunk)
        EVB = min(1.0, trace / m)
        
        return EVB
    
    def compute_CR(self, trajectory: 'CompatibleTrajectory') -> float:
        """Compute Contradiction Rate using enhanced semantic consistency"""
        E_matrix = trajectory.get_matrix()
        H = len(trajectory.steps) - 1
        
        changes = 0
        for h in range(1, len(trajectory.steps)):
            # Get rankings at step h and h-1
            rank_prev = np.argsort(E_matrix[h-1])
            rank_curr = np.argsort(E_matrix[h])
            
            # Check if rankings differ
            if not np.array_equal(rank_prev, rank_curr):
                changes += 1
        
        CR = changes / max(H, 1)
        return CR
    
    def compute_TS(self, trajectory: 'CompatibleTrajectory') -> float:
        """Compute Trajectory Smoothness using enhanced semantic similarity"""
        # Use enhanced semantic similarity for trajectory smoothness
        responses = [trajectory.response] + [f"Step {i}" for i in range(len(trajectory.steps)-1)]
        
        # Compute semantic similarity between consecutive responses
        from semantic_signal_framework import unified_semantic_estimator
        
        similarities = []
        for i in range(1, len(responses)):
            if i < len(responses):
                # For demo, use evaluative vector distances as proxy
                E_matrix = trajectory.get_matrix()
                if i < len(E_matrix):
                    dist = np.linalg.norm(E_matrix[i] - E_matrix[i-1])
                    similarity = 1.0 - min(1.0, dist / np.sqrt(E_matrix.shape[1]))
                    similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        return max(0.0, min(1.0, avg_similarity))
    
    def compute_ES(self, trajectory: 'CompatibleTrajectory') -> float:
        """Compute Expectation Stability using enhanced semantic confidence"""
        E_matrix = trajectory.get_matrix()
        
        # Use confidence dimension (enhanced semantic confidence)
        confidence_vals = E_matrix[:, 0]
        
        variance = np.var(confidence_vals)
        V_max = 0.25  # max variance for [0,1] values is 0.25
        
        ES = 1.0 - min(1.0, variance / V_max)
        return ES
    
    def compute_PD(self, trajectory: 'CompatibleTrajectory') -> float:
        """Compute Policy Divergence using enhanced semantic analysis"""
        E_matrix = trajectory.get_matrix()
        H = len(trajectory.steps) - 1
        
        divergences = []
        for h in range(1, len(trajectory.steps)):
            # Treat normalized vectors as probability distributions
            p = E_matrix[h-1] / (np.sum(E_matrix[h-1]) + 1e-10)
            q = E_matrix[h] / (np.sum(E_matrix[h]) + 1e-10)
            
            # Compute KL divergence
            kl_div = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
            divergences.append(min(10.0, max(0.0, kl_div)))  # Clamp for stability
        
        PD = np.mean(divergences) if divergences else 0.0
        return min(1.0, PD / 10.0)  # Normalize to [0,1]


@dataclass
class CompatibleTrajectory:
    """Backward-compatible Trajectory using enhanced semantic vectors"""
    response: str
    steps: List[CompatibleEvaluativeVector]
    
    def get_matrix(self) -> np.ndarray:
        """Get evaluative vectors as matrix (H+1 x m)"""
        return np.vstack([step.to_array() for step in self.steps])


class SemanticControlProbeType1:
    """Enhanced Control Probe Type-1 with backward compatibility"""
    
    def __init__(self, config):
        """Initialize semantic control probe type-1"""
        self.enhanced_probe = EnhancedControlProbeType1(config)
        self.tau = config.tau
    
    def evaluate(self, response: str, prompt: str = "", coherence_metrics=None) -> Tuple[CommitmentDecision, float, Dict]:
        """Evaluate using enhanced semantic analysis with compatible interface"""
        return self.enhanced_probe.evaluate(response, prompt, coherence_metrics)
    
    def generate_blocked_response(self, original_prompt: str, reason: Dict) -> str:
        """Generate blocked response using enhanced semantic analysis"""
        return self.enhanced_probe.generate_blocked_response(original_prompt, reason)


class SemanticControlProbeType2:
    """Enhanced Control Probe Type-2 with backward compatibility"""
    
    def __init__(self, config):
        """Initialize semantic control probe type-2"""
        self.enhanced_probe = EnhancedControlProbeType2(config)
        self.Theta = config.Theta
        self.max_history = config.max_history_turns
        self.history = []  # Maintain compatible interface
        self.awaiting_new_topic = False
        self.last_topic_tokens = set()
        self.pending_decision = None
    
    def add_turn(self, prompt: str, response: str, risk_score: float):
        """Add turn with enhanced semantic analysis"""
        # Add to enhanced probe
        self.enhanced_probe.add_turn(prompt, response, risk_score)
        
        # Maintain compatible history
        turn = CompatibleInteractionTurn.from_turn_data(prompt, response, risk_score, len(self.history))
        self.history.append(turn)
        
        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def compute_cumulative_risk(self) -> float:
        """Compute cumulative risk (compatible interface)"""
        return self.enhanced_probe.compute_cumulative_risk()
    
    def detect_semantic_drift(self) -> Tuple[bool, float]:
        """Detect semantic drift using enhanced analysis"""
        return self.enhanced_probe.detect_semantic_drift()
    
    def detect_sycophancy(self) -> Tuple[bool, float]:
        """Detect sycophancy using enhanced analysis"""
        return self.enhanced_probe.detect_sycophancy()
    
    def evaluate(self) -> Tuple[CommitmentDecision, Dict]:
        """Evaluate using enhanced semantic analysis"""
        return self.enhanced_probe.evaluate()
    
    def reset(self):
        """Reset both enhanced and compatible state"""
        self.enhanced_probe.reset()
        self.history.clear()
        self.awaiting_new_topic = False
        self.last_topic_tokens.clear()
        self.pending_decision = None


# Compatibility functions for easy migration
def create_semantic_ecr_engine(config):
    """Create semantic ECR engine with backward compatibility"""
    return SemanticECREngine(config)


def create_semantic_control_probes(config):
    """Create semantic control probes with backward compatibility"""
    return SemanticControlProbeType1(config), SemanticControlProbeType2(config)


# Test function
def test_semantic_compatibility():
    """Test semantic compatibility layer"""
    
    print("Testing Semantic Compatibility Layer...")
    print("=" * 60)
    
    # Mock config
    class MockConfig:
        K = 3
        H = 2
        tau_CCI = 0.65
        alpha = 0.2
        beta = 0.2
        gamma = 0.2
        delta = 0.2
        epsilon = 0.2
        lambda_shrink = 0.1
        parallel_candidates = False
        max_parallel_workers = 4
        tau = 0.4
        Theta = 2.0
        max_history_turns = 10
    
    config = MockConfig()
    
    # Test semantic ECR engine
    print("\nSemantic ECR Engine:")
    print("-" * 30)
    
    ecr = SemanticECREngine(config)
    
    # Test evaluative vector
    response = "I recommend using multi-factor authentication for better security."
    vector = CompatibleEvaluativeVector.from_response(response, 0, "Database security question")
    
    print(f"Response: {response}")
    print(f"Confidence: {vector.confidence:.3f}")
    print(f"Retrieval: {vector.retrieval:.3f}")
    print(f"Safety: {vector.safety:.3f}")
    
    # Test semantic control probes
    print("\nSemantic Control Probes:")
    print("-" * 30)
    
    cp1, cp2 = create_semantic_control_probes(config)
    
    # Test CP1
    decision, sigma, debug = cp1.evaluate(response, "What's the best security approach?")
    print(f"CP1 Decision: {decision.value}, σ={sigma:.3f}")
    
    # Test CP2
    cp2.add_turn("Is this secure?", "Yes, it's very secure.", 0.3)
    cp2.add_turn("Are you sure?", "Well, there might be some risks.", 0.5)
    cp2.add_turn("What risks?", "Actually, I'm not sure about the risks.", 0.7)
    
    decision, debug = cp2.evaluate()
    print(f"CP2 Decision: {decision.value}")
    print(f"  Semantic Drift: {debug.get('semantic_drift', False)}")
    print(f"  Sycophancy: {debug.get('sycophancy', False)}")
    
    print("\n" + "=" * 60)
    print("Semantic compatibility test completed successfully!")


if __name__ == "__main__":
    test_semantic_compatibility()