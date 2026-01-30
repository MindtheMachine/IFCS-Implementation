"""
Evaluative Coherence Regulation (ECR) Implementation
Based on: Chatterjee, A. (2026b). Evaluative Coherence Regulation (ECR)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import re


@dataclass
class EvaluativeVector:
    """Evaluative vector at a single step"""
    confidence: float  # 0-1
    retrieval: float  # 0-1
    uncertainty: float  # 0-1
    safety: float  # 0-1
    consistency: float  # 0-1
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.confidence,
            self.retrieval,
            self.uncertainty,
            self.safety,
            self.consistency
        ])
    
    @classmethod
    def from_response(cls, response: str, step: int, context: str = "") -> 'EvaluativeVector':
        """Estimate evaluative vector from response text"""
        # These are heuristic approximations
        
        # Confidence: based on presence of certainty markers
        confidence_markers = ['definitely', 'certainly', 'clearly', 'obviously', 'must be']
        uncertainty_markers = ['might', 'could', 'possibly', 'perhaps', 'may']
        
        conf_count = sum(1 for marker in confidence_markers if marker in response.lower())
        uncert_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
        
        confidence = min(1.0, 0.5 + (conf_count * 0.1) - (uncert_count * 0.1))
        uncertainty = 1.0 - confidence
        
        # Retrieval: semantic similarity to context (simplified)
        if context:
            # Count overlapping significant words
            response_words = set(re.findall(r'\w+', response.lower()))
            context_words = set(re.findall(r'\w+', context.lower()))
            overlap = len(response_words & context_words)
            retrieval = min(1.0, overlap / max(len(context_words), 1) * 2)
        else:
            retrieval = 0.5  # neutral
        
        # Safety: absence of harmful patterns
        safety = 0.9  # default high unless specific issues detected
        
        # Consistency: placeholder (would need comparison with previous steps)
        consistency = 0.8
        
        return cls(
            confidence=confidence,
            retrieval=retrieval,
            uncertainty=uncertainty,
            safety=safety,
            consistency=consistency
        )


@dataclass
class Trajectory:
    """Response trajectory over H steps"""
    response: str
    steps: List[EvaluativeVector]
    
    def get_matrix(self) -> np.ndarray:
        """Get evaluative vectors as matrix (H+1 x m)"""
        return np.vstack([step.to_array() for step in self.steps])


@dataclass
class CoherenceMetrics:
    """Complete coherence metrics for a trajectory"""
    EVB: float  # Evaluative Variance Bound
    CR: float   # Contradiction Rate
    TS: float   # Trajectory Smoothness
    ES: float   # Expectation Stability
    PD: float   # Policy Divergence
    CCI: float  # Composite Coherence Index
    
    def is_admissible(self, threshold: float) -> bool:
        """Check if CCI meets threshold"""
        return self.CCI >= threshold


class ECREngine:
    """Evaluative Coherence Regulation Engine"""
    
    def __init__(self, config):
        """Initialize ECR with configuration
        
        Args:
            config: ECRConfig instance
        """
        self.config = config
        self.K = config.K
        self.H = config.H
        self.tau_CCI = config.tau_CCI
        self.weights = (config.alpha, config.beta, config.gamma, config.delta, config.epsilon)
        
    def generate_candidates(
        self,
        prompt: str,
        llm_call_fn,
        num_candidates: Optional[int] = None,
        llm_provider: Optional[object] = None
    ) -> List[str]:
        """Generate K candidate responses
        
        Args:
            prompt: Input prompt
            llm_call_fn: Function to call LLM (prompt, temperature) -> response
            num_candidates: Optional override for candidate count
            llm_provider: Optional provider instance (for native batch APIs)
            
        Returns:
            List of K candidate responses
        """
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
                # Leave temperature policy-driven by the provider.
                response = llm_call_fn(prompt, temperature=None)
                candidates.append(response)
            return candidates

        max_workers = self.config.max_parallel_workers or target_k
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(llm_call_fn, prompt, temperature=None)
                for _ in range(target_k)
            ]
            return [future.result() for future in futures]
    
    def unroll_trajectory(self, candidate: str, prompt: str, llm_call_fn) -> Trajectory:
        """Unroll trajectory over H steps
        
        Args:
            candidate: Candidate response
            prompt: Original prompt
            llm_call_fn: Function to call LLM
            
        Returns:
            Trajectory with evaluative vectors
        """
        steps = []
        current_context = prompt
        
        # Initial step
        steps.append(EvaluativeVector.from_response(candidate, 0, current_context))
        
        # Subsequent steps (simulated continuation)
        current_response = candidate
        for h in range(1, self.H + 1):
            # Simulate asking "continue" or "elaborate"
            continuation_prompt = f"{current_context}\n\n{current_response}\n\nContinue your explanation:"
            
            # Get continuation (shorter)
            continuation = llm_call_fn(continuation_prompt, temperature=None, max_tokens=200)
            
            # Create evaluative vector
            steps.append(EvaluativeVector.from_response(
                continuation, h, current_context
            ))
            
            current_response = continuation
            current_context = f"{current_context}\n{current_response}"
        
        return Trajectory(response=candidate, steps=steps)
    
    def compute_EVB(self, trajectory: Trajectory) -> float:
        """Compute Evaluative Variance Bound
        
        EVB = min(1, tr(Cov(E)) / m)
        with Ledoit-Wolf shrinkage for small H
        """
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
    
    def compute_CR(self, trajectory: Trajectory) -> float:
        """Compute Contradiction Rate
        
        CR = (1/H) * sum of rank changes
        """
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
    
    def compute_TS(self, trajectory: Trajectory) -> float:
        """Compute Trajectory Smoothness
        
        TS = 1 - (1/H) * sum of semantic distances
        """
        # Simplified: use Euclidean distance between consecutive vectors
        E_matrix = trajectory.get_matrix()
        H = len(trajectory.steps) - 1
        
        distances = []
        for h in range(1, len(trajectory.steps)):
            dist = np.linalg.norm(E_matrix[h] - E_matrix[h-1])
            # Normalize by sqrt(m) for scale invariance
            dist_norm = dist / np.sqrt(E_matrix.shape[1])
            distances.append(min(1.0, dist_norm))
        
        avg_dist = np.mean(distances) if distances else 0
        TS = 1.0 - avg_dist
        
        return TS
    
    def compute_ES(self, trajectory: Trajectory) -> float:
        """Compute Expectation Stability
        
        ES = 1 - Var(r - r_hat) / V_max
        
        Simplified: measure variance in confidence across steps
        """
        E_matrix = trajectory.get_matrix()
        
        # Use confidence dimension as proxy for expected utility
        confidence_vals = E_matrix[:, 0]
        
        variance = np.var(confidence_vals)
        V_max = 0.25  # max variance for [0,1] values is 0.25
        
        ES = 1.0 - min(1.0, variance / V_max)
        return ES
    
    def compute_PD(self, trajectory: Trajectory) -> float:
        """Compute Policy Divergence
        
        PD = (1/H) * sum of KL divergences
        
        Simplified: measure changes in evaluative distribution
        """
        E_matrix = trajectory.get_matrix()
        H = len(trajectory.steps) - 1
        
        divergences = []
        for h in range(1, len(trajectory.steps)):
            # Treat normalized vectors as probability distributions
            p = E_matrix[h-1] / (np.sum(E_matrix[h-1]) + 1e-10)
            q = E_matrix[h] / (np.sum(E_matrix[h]) + 1e-10)
            
            # KL divergence
            kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
            divergences.append(abs(kl))
        
        PD = np.mean(divergences) if divergences else 0
        
        # Normalize
        D_max = 2.0  # approximate max KL for this setup
        PD_norm = min(1.0, PD / D_max)
        
        return PD_norm
    
    def compute_coherence_metrics(self, trajectory: Trajectory) -> CoherenceMetrics:
        """Compute all coherence metrics and CCI"""
        EVB = self.compute_EVB(trajectory)
        CR = self.compute_CR(trajectory)
        TS = self.compute_TS(trajectory)
        ES = self.compute_ES(trajectory)
        PD = self.compute_PD(trajectory)
        
        # Compute CCI
        alpha, beta, gamma, delta, epsilon = self.weights
        CCI = (
            alpha * (1 - EVB) +
            beta * (1 - CR) +
            gamma * TS +
            delta * ES +
            epsilon * (1 - PD)
        )
        
        return CoherenceMetrics(
            EVB=EVB,
            CR=CR,
            TS=TS,
            ES=ES,
            PD=PD,
            CCI=CCI
        )
    
    def select_best_candidate(
        self, 
        candidates: List[str], 
        prompt: str,
        llm_call_fn
    ) -> Tuple[str, CoherenceMetrics, Dict]:
        """Select best candidate based on coherence
        
        Returns:
            (selected_response, metrics, debug_info)
        """
        trajectories = []
        all_metrics = []
        
        print(f"[ECR] Evaluating {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            print(f"[ECR] Processing candidate {i+1}/{len(candidates)}...")
            
            # Unroll trajectory
            trajectory = self.unroll_trajectory(candidate, prompt, llm_call_fn)
            trajectories.append(trajectory)
            
            # Compute metrics
            metrics = self.compute_coherence_metrics(trajectory)
            all_metrics.append(metrics)
            
            print(f"[ECR] Candidate {i+1} CCI: {metrics.CCI:.3f}")
        
        # Find admissible candidates
        admissible = [
            (i, m) for i, m in enumerate(all_metrics) 
            if m.is_admissible(self.tau_CCI)
        ]
        
        if admissible:
            # Select highest CCI among admissible
            best_idx, best_metrics = max(admissible, key=lambda x: x[1].CCI)
            print(f"[ECR] Selected admissible candidate {best_idx+1} with CCI={best_metrics.CCI:.3f}")
        else:
            # Fallback: select highest CCI even if not admissible
            best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i].CCI)
            best_metrics = all_metrics[best_idx]
            print(f"[ECR] No admissible candidates. Selected best CCI={best_metrics.CCI:.3f}")
        
        debug_info = {
            'num_candidates': len(candidates),
            'num_admissible': len(admissible),
            'all_cci_scores': [m.CCI for m in all_metrics],
            'selected_idx': best_idx,
            'threshold': self.tau_CCI
        }
        
        return candidates[best_idx], best_metrics, debug_info
