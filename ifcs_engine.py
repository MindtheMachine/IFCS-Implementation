"""
Inference-Time Commitment Shaping (IFCS) Implementation
Based on: Chatterjee, A. (2026c). Inference-Time Commitment Shaping

⚠️ DEPRECATION NOTICE:
This module contains the LEGACY IFCS implementation.
New code should use HybridIFCS from commitment_regulation_architecture.py

The hybrid implementation combines:
- Paper's R(z*) = λ₁·ê + λ₂·ŝ + λ₃·â + λ₄·t̂ computation
- Paper's six deterministic transformation rules (Γ)
- Implementation's semantic preservation guarantee with rollback

This legacy module is maintained for:
- Backward compatibility with existing tests
- Performance benchmarking comparisons
- Legacy system validation
- κ(z*) gate testing (CommitmentActualityClassifier)

For new development, use:
    from commitment_regulation_architecture import HybridIFCS
"""

import re
import time
import json
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from trilogy_config import IFCSConfig
from dashboard_metrics import get_dashboard_metrics


@dataclass
class CommitmentRisk:
    """Commitment risk components"""
    e_hat: float  # Evidential insufficiency
    s_hat: float  # Scope inflation
    a_hat: float  # Authority cues
    t_hat: float  # Temporal risk
    R: float      # Overall risk score
    
    def __str__(self) -> str:
        return (f"R={self.R:.3f} [-={self.e_hat:.2f}, -={self.s_hat:.2f}, "
                f"-={self.a_hat:.2f}, t-={self.t_hat:.2f}]")


@dataclass
class CommitmentActualityLog:
    """Structured log entry for commitment-actuality classification decisions"""
    timestamp: str
    kappa_value: int  # 0 or 1
    classification: str  # "commitment-bearing" or "non-commitment-bearing"
    computation_time_ms: float
    context_metadata: Dict
    classification_reasoning: Dict
    performance_metrics: Dict
    
    def to_json(self) -> str:
        """Convert to JSON string for logging"""
        return json.dumps(asdict(self), indent=2)


@dataclass 
class NonCommitmentBearingMetrics:
    """Metrics for non-commitment-bearing contexts with rationale"""
    timestamp: str
    prompt_length: int
    response_length: int
    classification_rationale: str
    descriptive_signals: Dict
    commitment_signals: Dict
    context_bias: float
    hedging_penalty: float
    final_score_difference: float


class CommitmentActualityClassifier:
    """Computes -(z*): commitment-actuality indicator using signal estimation"""
    
    def __init__(self):
        """Initialize the commitment-actuality classifier with signal-based approach"""
        # Signal estimation thresholds (industry-standard approach)
        self.assertion_strength_threshold = 0.3
        self.epistemic_certainty_threshold = 0.4
        self.scope_breadth_threshold = 0.3
        self.authority_posture_threshold = 0.3
        
        # Signal aggregation weights
        self.signal_weights = {
            'assertion_strength': 0.35,
            'epistemic_certainty': 0.25,
            'scope_breadth': 0.20,
            'authority_posture': 0.20
        }
        
        # Logging and metrics tracking
        self.classification_logs: List[CommitmentActualityLog] = []
        self.non_commitment_metrics: List[NonCommitmentBearingMetrics] = []
    
    def _aggregate_signals(self, signals: Dict[str, float]) -> float:
        """Aggregate signals using fuzzy logic over signals (not fuzzy text matching)
        
        Industry approach: R(z*) = fuzzy_aggregate(E, A, U)
        Returns: aggregated_signal - [0,1]
        """
        weighted_sum = sum(
            signals[signal] * self.signal_weights[signal] 
            for signal in signals if signal in self.signal_weights
        )
        
        # Apply fuzzy aggregation (industry-standard approach)
        # Non-linear aggregation to handle signal interactions
        aggregated = weighted_sum ** 0.8  # Slight compression for stability
        
        return min(1.0, max(0.0, aggregated))
    
    def _analyze_semantic_patterns(self, text: str) -> dict:
        """Analyze text for semantic commitment patterns using statistical signals"""
        from signal_estimation import signal_estimator, compute_text_stats

        stats = compute_text_stats(text)
        assertion = signal_estimator.estimate_assertion_strength(text)
        certainty = signal_estimator.estimate_epistemic_certainty(text)
        authority = signal_estimator.estimate_authority_posture(text)
        scope = signal_estimator.estimate_scope_breadth(text)

        commitment_score = min(
            1.0,
            0.35 * assertion +
            0.25 * certainty +
            0.25 * authority +
            0.15 * (1.0 - stats["question_ratio"])
        )

        descriptive_score = min(
            1.0,
            0.40 * stats["list_density"] +
            0.30 * stats["length_norm"] +
            0.20 * stats["type_token_ratio"] +
            0.10 * (1.0 - authority)
        )

        return {
            "commitment_score": commitment_score,
            "descriptive_score": descriptive_score,
            "commitment_features": {
                "assertion_strength": assertion,
                "epistemic_certainty": certainty,
                "authority_posture": authority,
                "scope_breadth": scope,
            },
            "descriptive_features": {
                "list_density": stats["list_density"],
                "length_norm": stats["length_norm"],
                "type_token_ratio": stats["type_token_ratio"],
            },
        }
    
    def _analyze_syntactic_patterns(self, text: str) -> dict:
        """Analyze syntactic patterns using signal estimation (no text-matching heuristics)"""
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator, compute_text_stats
        
        # Estimate syntactic commitment signals
        assertion_signal = signal_estimator.estimate_assertion_strength(text)
        authority_signal = signal_estimator.estimate_authority_posture(text)
        
        # Convert signals to syntactic scores
        commitment_syntax = (assertion_signal + authority_signal) * 2.0  # Scale to match original range
        descriptive_syntax = max(0, 2.0 - commitment_syntax)  # Inverse relationship
        
        return {
            'commitment_syntax': commitment_syntax,
            'descriptive_syntax': descriptive_syntax
        }
    
    def _log_classification_decision(
        self, 
        kappa: int, 
        response: str, 
        prompt: str, 
        computation_time_ms: float,
        semantic_analysis: dict,
        syntactic_analysis: dict,
        pragmatic_analysis: dict,
        final_scores: dict
    ) -> None:
        """Log -(z*) decision with comprehensive context metadata and reasoning"""
        
        classification = "commitment-bearing" if kappa == 1 else "non-commitment-bearing"
        
        # Context metadata
        context_metadata = {
            'prompt_length': len(prompt) if prompt else 0,
            'response_length': len(response) if response else 0,
            'prompt_hash': hash(prompt) if prompt else None,
            'response_hash': hash(response) if response else None,
            'word_count': len(response.split()) if response else 0,
            'sentence_count': len([s for s in re.split(r'[.!?]+', response) if s.strip()]) if response else 0
        }
        
        # Classification reasoning with signal-based approach
        classification_reasoning = {
            'method': 'signal_estimation',
            'signals': semantic_analysis.get('signals', {}),
            'aggregated_signal': final_scores.get('aggregated_signal', 0.0),
            'threshold': semantic_analysis.get('threshold', 0.4),
            'decision_logic': semantic_analysis.get('decision_logic', 'signal-based classification')
        }
        
        # Performance metrics
        performance_metrics = {
            'computation_time_ms': computation_time_ms,
            'signal_estimation_method': 'industry_standard',
            'signals_computed': len(semantic_analysis.get('signals', {})),
            'aggregation_method': 'fuzzy_logic_over_signals'
        }
        
        # Create structured log entry
        log_entry = CommitmentActualityLog(
            timestamp=datetime.now().isoformat(),
            kappa_value=kappa,
            classification=classification,
            computation_time_ms=computation_time_ms,
            context_metadata=context_metadata,
            classification_reasoning=classification_reasoning,
            performance_metrics=performance_metrics
        )
        
        # Store log entry
        self.classification_logs.append(log_entry)
        
        # Enhanced debug output with signal-based reasoning
        print(f"[IFCS--] Classification Decision: -(z*)={kappa} ({classification})")
        print(f"[IFCS--] Computation Time: {computation_time_ms:.2f}ms")
        print(f"[IFCS--] Context: {context_metadata['word_count']} words, {context_metadata['sentence_count']} sentences")
        print(f"[IFCS--] Signal-Based Reasoning: {classification_reasoning['decision_logic']}")
        
        # Signal breakdown for debugging
        signals = semantic_analysis.get('signals', {})
        if signals:
            print(f"[IFCS--] Signals: assertion_strength={signals.get('assertion_strength', 0):.3f}, "
                  f"epistemic_certainty={signals.get('epistemic_certainty', 0):.3f}, "
                  f"scope_breadth={signals.get('scope_breadth', 0):.3f}, "
                  f"authority_posture={signals.get('authority_posture', 0):.3f}")
    
    def _record_non_commitment_metrics(
        self, 
        response: str, 
        prompt: str, 
        signal_analysis: dict,
        final_scores: dict
    ) -> None:
        """Record detailed metrics for non-commitment-bearing contexts with signal-based rationale"""
        
        # Generate signal-based rationale
        signals = signal_analysis.get('signals', {})
        aggregated_signal = final_scores.get('aggregated_signal', 0.0)
        
        rationale_parts = []
        for signal_name, signal_value in signals.items():
            if signal_value < 0.3:  # Low signal threshold
                rationale_parts.append(f"low {signal_name} ({signal_value:.3f})")
        
        rationale = f"Signal-based classification: aggregated_signal ({aggregated_signal:.3f}) below threshold (0.4). " + \
                   ("Low signals: " + ", ".join(rationale_parts) if rationale_parts else "All signals below commitment threshold")
        
        # Create metrics entry
        metrics_entry = NonCommitmentBearingMetrics(
            timestamp=datetime.now().isoformat(),
            prompt_length=len(prompt) if prompt else 0,
            response_length=len(response) if response else 0,
            classification_rationale=rationale,
            descriptive_signals=signals,  # Now contains signal values instead of pattern counts
            commitment_signals={},  # Not used in signal-based approach
            context_bias=0.0,  # Not used in signal-based approach
            hedging_penalty=0.0,  # Not used in signal-based approach
            final_score_difference=aggregated_signal - 0.4  # Distance from threshold
        )
        
        # Store metrics
        self.non_commitment_metrics.append(metrics_entry)
        
        # Enhanced debug output for non-commitment-bearing contexts
        print(f"[IFCS--] Non-Commitment Rationale: {rationale}")
        print(f"[IFCS--] Aggregated Signal: {aggregated_signal:.3f} (threshold: 0.4)")
        if rationale_parts:
            print(f"[IFCS--] Low Signals: {', '.join(rationale_parts)}")
    
    def get_classification_logs(self, limit: Optional[int] = None) -> List[CommitmentActualityLog]:
        """Get recent classification logs for monitoring"""
        if limit:
            return self.classification_logs[-limit:]
        return self.classification_logs.copy()
    
    def get_non_commitment_metrics(self, limit: Optional[int] = None) -> List[NonCommitmentBearingMetrics]:
        """Get recent non-commitment-bearing metrics for analysis"""
        if limit:
            return self.non_commitment_metrics[-limit:]
        return self.non_commitment_metrics.copy()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for -(z*) computation"""
        if not self.classification_logs:
            return {'total_classifications': 0, 'avg_computation_time_ms': 0}
        
        times = [log.computation_time_ms for log in self.classification_logs]
        commitment_count = sum(1 for log in self.classification_logs if log.kappa_value == 1)
        
        return {
            'total_classifications': len(self.classification_logs),
            'commitment_bearing_count': commitment_count,
            'non_commitment_bearing_count': len(self.classification_logs) - commitment_count,
            'commitment_bearing_ratio': commitment_count / len(self.classification_logs),
            'avg_computation_time_ms': sum(times) / len(times),
            'min_computation_time_ms': min(times),
            'max_computation_time_ms': max(times),
            'total_computation_time_ms': sum(times)
        }
    
    def _analyze_pragmatic_context(self, response: str, prompt: str) -> dict:
        """Analyze pragmatic context from prompt to inform classification"""
        if not prompt:
            return {'context_bias': 0}

        from signal_estimation import compute_text_stats

        stats = compute_text_stats(prompt)
        context_bias = 0

        # Statistical proxy: shorter, question-heavy prompts lean advice-seeking
        if stats["question_ratio"] >= 0.5 and stats["length_norm"] < 0.35:
            context_bias += 1

        # Statistical proxy: longer, list-like prompts lean descriptive
        if stats["length_norm"] > 0.5 or stats["list_density"] > 0.15:
            context_bias -= 1

        return {"context_bias": context_bias}
    
    def is_commitment_bearing(self, response: str, prompt: str) -> bool:
        """
        Determine if response constitutes a commitment-bearing act using true signal estimation.
        Industry approach: Estimates latent epistemic signals using statistical methods.
        No regex patterns, no hardcoded word lists, no text-matching heuristics.
        
        Args:
            response: The response text to classify
            prompt: The original prompt for context
            
        Returns:
            True (-=1) if response compresses uncertainty into actionable stance
            False (-=0) if response is descriptive/enumerative/non-binding
        """
        # Start performance timing
        start_time = time.perf_counter()
        
        # Get dashboard metrics collector
        dashboard_metrics = get_dashboard_metrics()
        
        if not response or not response.strip() or len(response.strip()) < 15:
            # Log minimal response case
            computation_time_ms = (time.perf_counter() - start_time) * 1000
            self._log_classification_decision(
                kappa=0,
                response=response,
                prompt=prompt,
                computation_time_ms=computation_time_ms,
                semantic_analysis={'reason': 'minimal_response'},
                syntactic_analysis={'reason': 'minimal_response'},
                pragmatic_analysis={'reason': 'minimal_response'},
                final_scores={'aggregated_signal': 0.0, 'signals': {}}
            )
            
            # Record dashboard metrics
            dashboard_metrics.record_classification(
                kappa_value=0,
                computation_time_ms=computation_time_ms,
                is_error=False,
                is_fallback=True,  # Minimal response is a fallback scenario
                metadata={'reason': 'minimal_response', 'response_length': len(response.strip())}
            )
            
            return False
        
        # True signal estimation (replaces all text-matching heuristics)
        from signal_estimation import signal_estimator
        
        signals = {
            'assertion_strength': signal_estimator.estimate_assertion_strength(response),
            'epistemic_certainty': signal_estimator.estimate_epistemic_certainty(response),
            'scope_breadth': signal_estimator.estimate_scope_breadth(response),
            'authority_posture': signal_estimator.estimate_authority_posture(response)
        }
        
        # Aggregate signals using fuzzy logic over signals (not fuzzy text)
        aggregated_signal = self._aggregate_signals(signals)
        
        # Decision threshold (industry-standard bounded risk score)
        commitment_threshold = 0.35  # Slightly lower threshold for better sensitivity
        kappa = 1 if aggregated_signal >= commitment_threshold else 0
        
        # Calculate computation time
        computation_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Prepare analysis for logging (signal-based instead of pattern-based)
        signal_analysis = {
            'signals': signals,
            'aggregated_signal': aggregated_signal,
            'threshold': commitment_threshold,
            'decision_logic': f"aggregated_signal ({aggregated_signal:.3f}) {'-' if kappa == 1 else '<'} threshold ({commitment_threshold:.3f})"
        }
        
        # Comprehensive logging
        self._log_classification_decision(
            kappa=kappa,
            response=response,
            prompt=prompt,
            computation_time_ms=computation_time_ms,
            semantic_analysis=signal_analysis,
            syntactic_analysis={'method': 'signal_estimation'},
            pragmatic_analysis={'method': 'signal_estimation'},
            final_scores={'aggregated_signal': aggregated_signal, 'signals': signals}
        )
        
        # Record metrics for non-commitment-bearing contexts
        if kappa == 0:
            self._record_non_commitment_metrics(response, prompt, signal_analysis, 
                                              {'aggregated_signal': aggregated_signal, 'signals': signals})
        
        # Record dashboard metrics
        dashboard_metrics.record_classification(
            kappa_value=kappa,
            computation_time_ms=computation_time_ms,
            is_error=False,
            is_fallback=False,
            metadata={
                'aggregated_signal': aggregated_signal,
                'signals': signals,
                'response_length': len(response),
                'prompt_length': len(prompt) if prompt else 0
            }
        )
        
        return kappa == 1
        dashboard_metrics.record_classification(
            kappa_value=kappa,
            computation_time_ms=computation_time_ms,
            is_error=False,  # No error detection logic yet - could be enhanced
            is_fallback=False,  # Normal classification
            metadata={
                'response_length': len(response),
                'prompt_length': len(prompt) if prompt else 0,
                'commitment_score': commitment_score,
                'descriptive_score': descriptive_score,
                'adjusted_difference': adjusted_difference
            }
        )
        
        return kappa == 1


class IFCSEngine:
    """Inference-Time Commitment Shaping Engine"""
    
    def __init__(self, config):
        """Initialize IFCS with configuration
        
        Args:
            config: IFCSConfig instance
        """
        self.config = config
        self.rho = config.rho
        self.weights = (config.lambda_e, config.lambda_s, config.lambda_a, config.lambda_t)
        self.commitment_classifier = CommitmentActualityClassifier()

    def compute_evidential_insufficiency(
        self,
        response: str,
        prompt: str = "",
        context: str = ""
    ) -> float:
        """Compute -(z*): Evidential insufficiency using semantic analysis

        Args:
            response: Response text
            prompt: Original prompt
            context: Available context/grounding

        Returns:
            - score [0, 1]
        """
        if not response or not response.strip():
            return 0.5  # Neutral
        
        # Use signal estimation for evidential analysis
        from signal_estimation import signal_estimator
        evidential_risk = signal_estimator.estimate_evidential_risk(response, context)
        
        # Combine with structural insufficiency signals
        structural_signals = self._structural_signals(prompt, response)
        structural_e = max(structural_signals.values()) if structural_signals else 0.0
        
        # Take the maximum of evidential and structural risk
        return min(1.0, max(evidential_risk, structural_e))

    def _detect_jurisdiction_dependency(self, prompt: str, response: str) -> float:
        """Detect answers that depend on external jurisdiction using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator
        
        # Estimate jurisdictional dependency signals
        authority_signal = signal_estimator.estimate_authority_posture(response)
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        
        # High authority + high certainty in permission context suggests jurisdictional dependency
        combined_signal = (authority_signal + certainty_signal) / 2.0
        
        # Scale based on prompt context (question-heavy, short prompts increase risk)
        stats = compute_text_stats(prompt)
        context_factor = 0.6 * stats["question_ratio"] + 0.4 * (1.0 - stats["length_norm"])

        if context_factor > 0.5 and combined_signal > 0.6:
            return min(0.8, combined_signal * (1.0 + context_factor * 0.4))

        return combined_signal * 0.3  # Lower baseline risk

    def _detect_policy_dependency(self, prompt: str, response: str) -> float:
        """Detect reliance on external policies using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator, compute_text_stats
        
        # Estimate policy dependency signals
        authority_signal = signal_estimator.estimate_authority_posture(response)
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        
        # Prompt complexity as a proxy for policy/context questions
        stats = compute_text_stats(prompt)
        policy_context = min(1.0, stats["length_norm"] * 0.6 + stats["list_density"] * 0.4)
        
        # High authority + certainty in policy context suggests dependency
        combined_signal = (authority_signal + certainty_signal) / 2.0
        
        if policy_context > 0.4 and combined_signal > 0.5:
            return min(0.6, combined_signal * (1.0 + policy_context * 0.2))
        
        return combined_signal * 0.2  # Lower baseline risk

    def _detect_permission_framing(self, prompt: str, response: str) -> float:
        """Detect binary permission framing using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator, compute_text_stats
        
        # Estimate binary framing signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Permission-seeking proxy: short, question-heavy prompts
        stats = compute_text_stats(prompt)
        permission_seeking = stats["question_ratio"] > 0.5 and stats["length_norm"] < 0.4
        
        # High certainty + authority in permission context suggests binary framing
        combined_signal = (certainty_signal + authority_signal) / 2.0
        
        if permission_seeking and combined_signal > 0.5:
            return min(0.7, combined_signal * 1.3)
        
        return combined_signal * 0.1  # Very low baseline risk

    def _detect_missing_personal_data(self, prompt: str, response: str) -> float:
        """Detect personal-need questions answered with strong diagnosis using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator, compute_text_stats
        
        # Estimate diagnostic certainty signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Personal-context proxy: higher specificity but short length
        stats = compute_text_stats(prompt)
        personal_context = stats["length_norm"] < 0.4 and stats["type_token_ratio"] < 0.7
        
        # High certainty + authority in personal context suggests missing personal data
        combined_signal = (certainty_signal + authority_signal) / 2.0
        
        if personal_context and combined_signal > 0.6:
            return min(0.9, combined_signal * 1.4)
        
        return combined_signal * 0.2  # Lower baseline risk

    def _detect_consequence_asymmetry(self, prompt: str, response: str) -> float:
        """Detect high-consequence framing with definitive guidance using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator, compute_text_stats
        
        # Estimate consequence-related certainty signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Consequence proxy: emphasis cues in prompt
        stats = compute_text_stats(prompt)
        consequence_context = min(1.0, stats["exclam_ratio"] * 0.7 + stats["question_ratio"] * 0.3)
        
        # High certainty + authority in consequence context suggests asymmetry
        combined_signal = (certainty_signal + authority_signal) / 2.0
        
        if consequence_context > 0.3 and combined_signal > 0.5:
            return min(0.8, combined_signal * (1.0 + consequence_context * 0.6))
        
        return combined_signal * 0.1  # Very low baseline risk

    def prompt_structural_signals(self, prompt: str) -> Dict[str, float]:
        """Enhanced structural risk estimation using fuzzy logic and semantic analysis"""
        if not prompt:
            return {}
        
        try:
            # Use enhanced signal estimation with fuzzy logic and semantic analysis
            from enhanced_signal_estimator import enhanced_signal_estimator
            
            # Get enhanced signals using fuzzy logic and semantic analysis
            signals = enhanced_signal_estimator.estimate_structural_signals(prompt)
            
            # Validate signals are in proper range
            validated_signals = {}
            for signal_type, value in signals.items():
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0 and not (value != value):  # Check for NaN
                    validated_signals[signal_type] = value
                else:
                    print(f"[IFCS] Warning: Invalid signal value for {signal_type}: {value}, using 0.0")
                    validated_signals[signal_type] = 0.0
            
            return validated_signals
            
        except Exception as e:
            # Graceful fallback to original heuristic approach
            print(f"[IFCS] Enhanced signal estimation failed: {e}, falling back to heuristic approach")
            return self._fallback_heuristic_signals(prompt)
    
    def _fallback_heuristic_signals(self, prompt: str) -> Dict[str, float]:
        """Fallback signal estimation using statistical semantics"""
        if not prompt:
            return {}

        from intent_classifier import intent_classifier
        return intent_classifier.analyze_prompt(prompt)

    def _structural_signals(self, prompt: str, response: str) -> Dict[str, float]:
        """Structural insufficiency signals (domain-agnostic)."""
        if not prompt:
            return {}
        
        # Use semantic intent signals + fuzzy weighting (enhanced estimator)
        try:
            from enhanced_signal_estimator import enhanced_signal_estimator
            signals = enhanced_signal_estimator.estimate_structural_signals(prompt)
            # Only keep the structural signals used by IFCS
            return {
                "jurisdictional": signals.get("jurisdictional", 0.0),
                "policy": signals.get("policy", 0.0),
                "binary": signals.get("binary", 0.0),
                "personal_data": signals.get("personal_data", 0.0),
                "consequence": signals.get("consequence", 0.0),
            }
        except Exception:
            # Fallback to legacy detectors
            return {
                "jurisdictional": self._detect_jurisdiction_dependency(prompt, response),
                "policy": self._detect_policy_dependency(prompt, response),
                "binary": self._detect_permission_framing(prompt, response),
                "personal_data": self._detect_missing_personal_data(prompt, response),
                "consequence": self._detect_consequence_asymmetry(prompt, response),
            }

    def _adaptive_rho(self, structural_signals: Dict[str, float], default_rho: float) -> Tuple[float, str]:
        """Adaptive threshold based on structural risk (domain-agnostic)."""
        if not structural_signals:
            return default_rho, "Default threshold (?=0.40) - no structural signals"

        max_signal = max(structural_signals.values())
        max_type = max(structural_signals.items(), key=lambda item: item[1])[0]

        if max_signal >= 0.7:
            return 0.30, f"Strict threshold (?=0.30) due to high {max_type} dependency"
        if max_signal >= 0.5:
            return 0.35, f"Moderate threshold (?=0.35) due to {max_type} concern"
        return default_rho, "Default threshold (?=0.40) - low structural risk"

    def compute_scope_inflation(self, response: str) -> float:
        """Compute -(z*): Scope inflation using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            - score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use signal estimation for scope analysis
        from signal_estimation import signal_estimator
        scope_signal = signal_estimator.estimate_scope_breadth(response)
        
        return scope_signal
    
    def compute_authority_cues(self, response: str) -> float:
        """Compute -(z*): Authority cues using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            - score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use signal estimation for authority analysis
        from signal_estimation import signal_estimator
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        return authority_signal
    
    def compute_temporal_risk(self, response: str, prompt: str) -> float:
        """Compute t-(z*): Temporal risk using semantic analysis
        
        Args:
            response: Response text
            prompt: Original prompt
            
        Returns:
            t- score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use signal estimation for temporal analysis
        from signal_estimation import signal_estimator
        temporal_signal = signal_estimator.estimate_temporal_risk(response, prompt)
        
        return temporal_signal
    
    def compute_early_authority_gradient(self, response: str) -> float:
        """Compute -AG: Early authority gradient using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            -AG score
        """
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) < 3:
            return 0.0
        
        # Compute authority scores for each sentence using semantic analysis
        def sentence_authority(sent: str) -> float:
            # Use signal estimation for authority detection
            from signal_estimation import signal_estimator
            authority_signal = signal_estimator.estimate_authority_posture(sent)
            return authority_signal
        
        # Initial authority (first 2 sentences)
        A_initial = sum(sentence_authority(s) for s in valid_sentences[:2]) / 2.0
        
        # Average authority
        A_average = sum(sentence_authority(s) for s in valid_sentences) / len(valid_sentences)
        
        delta_AG = A_initial - A_average
        return delta_AG

    def _prompt_risk_components(self, prompt: str) -> Dict[str, float]:
        """Estimate prompt-driven risk priors using signal estimation."""
        if not prompt:
            return {'e_hat': 0.0, 's_hat': 0.0, 'a_hat': 0.0, 't_hat': 0.0}
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator, compute_text_stats

        stats = compute_text_stats(prompt)
        e_hat = 0.0
        s_hat = 0.0
        a_hat = 0.0
        t_hat = 0.0

        t_hat = signal_estimator.estimate_temporal_risk("", prompt)

        # Ambiguity proxy: short, question-heavy prompts
        ambiguity = min(1.0, (1.0 - stats["length_norm"]) * (0.6 + stats["question_ratio"] * 0.4))
        if ambiguity > 0:
            e_hat = max(e_hat, min(0.6, ambiguity * 0.8))
            s_hat = max(s_hat, min(0.4, ambiguity * 0.6))

        # Scope proxy
        scope_proxy = signal_estimator.estimate_scope_breadth(prompt)
        s_hat = max(s_hat, scope_proxy * 0.6)

        # Authority-seeking proxy
        authority_proxy = signal_estimator.estimate_authority_posture(prompt)
        a_hat = max(a_hat, authority_proxy * 0.6)

        return {"e_hat": e_hat, "s_hat": s_hat, "a_hat": a_hat, "t_hat": t_hat}
    
    def compute_commitment_risk(
        self, 
        response: str, 
        prompt: str = "",
        context: str = ""
    ) -> CommitmentRisk:
        """Compute R(z*): Overall commitment risk
        
        Args:
            response: Response text
            prompt: Original prompt
            context: Available context
            
        Returns:
            CommitmentRisk with all components
        """
        # C6 COMPLIANCE: No domain detection or classification
        # Risk sensitivity emerges from -/-/-/t- score patterns only
        default_config = self.config  # Use base IFCS config directly
        
        # Use default weights (C6: domain-agnostic core mechanism)
        lambda_e, lambda_s, lambda_a, lambda_t = (
            default_config.lambda_e,
            default_config.lambda_s,
            default_config.lambda_a,
            default_config.lambda_t
        )
        
        # C6 COMPLIANCE: No domain detection or logging
        # Risk sensitivity emerges from -/-/-/t- score patterns only
        # No explicit domain classification is performed
        
        # Compute components
        e_hat = self.compute_evidential_insufficiency(response, prompt, context)
        s_hat = self.compute_scope_inflation(response)
        a_hat = self.compute_authority_cues(response)
        t_hat = self.compute_temporal_risk(response, prompt)
        
        # Blend in prompt-driven priors (promotes expected intervention for risky prompts)
        prompt_risk = self._prompt_risk_components(prompt)
        e_hat = max(e_hat, prompt_risk['e_hat'])
        s_hat = max(s_hat, prompt_risk['s_hat'])
        a_hat = max(a_hat, prompt_risk['a_hat'])
        t_hat = max(t_hat, prompt_risk['t_hat'])
        
        # Compute R(z*) using default config (C6: domain-agnostic core mechanism)
        R = (lambda_e * e_hat +
             lambda_s * s_hat +
             lambda_a * a_hat +
             lambda_t * t_hat)

        # Adaptive threshold based on structural signals (domain-agnostic).
        structural_signals = self._structural_signals(prompt, response)
        adaptive_rho, rho_reason = self._adaptive_rho(structural_signals, self.config.rho)
        rho_used = adaptive_rho

        # C6 COMPLIANCE: Risk sensitivity emerges from -/-/-/t- score patterns
        # No explicit domain classification or configuration override
        # All domain-specific logic completely removed for architectural compliance

        # Persist the effective rho for the caller (C6: no domain used)
        self._last_rho_used = rho_used
        self._last_structural_signals = structural_signals
        self._last_rho_reason = rho_reason
        
        return CommitmentRisk(
            e_hat=e_hat,
            s_hat=s_hat,
            a_hat=a_hat,
            t_hat=t_hat,
            R=R
        )
    
    def should_intervene(self, risk: CommitmentRisk, sigma: float, rho: float) -> bool:
        """Determine if IFCS should intervene
        
        Fires iff: -(z*) - - - R(z*) > -
        
        Args:
            risk: Computed commitment risk
            sigma: Admissibility signal from Control Probe Type-1
            rho: Commitment risk threshold to use
            
        Returns:
            True if should intervene
        """
        # Three-part firing condition: -(z*) - - - R(z*) > - - -(z*) = 1
        # Note: -(z*) = 1 is checked in shape_commitment before calling this method
        tau = 0.40  # Default Control Probe threshold
        return sigma >= tau and risk.R > rho
    
    def apply_transformation_rules(self, response: str, risk: CommitmentRisk) -> str:
        """Apply - operator: commitment-shaping transformations
        
        Args:
            response: Original response
            risk: Computed commitment risk
        Returns:
            Shaped response
        """
        shaped = response
        # Rule 1: Weaken Universal Claims
        if risk.s_hat >= 0.4:
            shaped = self._rule1_weaken_universals(shaped)
        
        # Rule 2: Surface Implicit Assumptions
        if risk.e_hat >= 0.4:
            shaped = self._rule2_surface_assumptions(shaped)
        
        # Rule 3: Attenuate Authority Cues
        if risk.a_hat >= 0.4:
            shaped = self._rule3_attenuate_authority(shaped)
        
        # Rule 4: Flatten Early Authority Gradient
        delta_AG = self.compute_early_authority_gradient(shaped)
        if delta_AG > self.config.delta_AG_threshold:
            shaped = self._rule4_flatten_gradient(shaped)
        
        # Rule 5: Add Conditional Framing
        if risk.e_hat > 0.6 or risk.s_hat > 0.6:
            shaped = self._rule5_add_conditionals(shaped)
        
        # Rule 6: Surface Disambiguation (detect ambiguous queries)
        # This requires prompt analysis, handled separately
        
        return shaped
    
    def _rule1_weaken_universals(self, text: str) -> str:
        """Rule 1: Weaken universal claims using signal-guided transformation"""
        # Use signal estimation to identify universal claims instead of regex patterns
        from signal_estimation import signal_estimator
        
        scope_signal = signal_estimator.estimate_scope_breadth(text)
        
        # Only apply transformations if scope signal is high
        if scope_signal < 0.4:
            return text

        prefix = "In many cases, "
        stripped = text.lstrip()
        if stripped.lower().startswith(prefix.lower()):
            return text
        leading_ws = text[: len(text) - len(stripped)]
        return f"{leading_ws}{prefix}{stripped}"
    
    def _rule2_surface_assumptions(self, text: str) -> str:
        """Rule 2: Surface implicit assumptions using signal-guided analysis"""
        # Use signal estimation to determine if assumptions need surfacing
        from signal_estimation import signal_estimator
        
        evidential_risk = signal_estimator.estimate_evidential_risk(text)
        
        # Only add qualifiers if evidential risk is high
        if evidential_risk < 0.4:
            return text

        suffix = " This may depend on specific context."
        marker = "may depend on specific context"
        if marker in text.lower():
            return text
        return f"{text.rstrip()}{suffix}"
    
    def _rule3_attenuate_authority(self, text: str) -> str:
        """Rule 3: Attenuate authority cues using signal-guided transformation"""
        # Use signal estimation to identify authority cues
        from signal_estimation import signal_estimator
        
        authority_signal = signal_estimator.estimate_authority_posture(text)
        
        # Only apply transformations if authority signal is high
        if authority_signal < 0.4:
            return text

        prefix = "One possible approach is: "
        stripped = text.lstrip()
        if stripped.lower().startswith(prefix.lower()):
            return text
        leading_ws = text[: len(text) - len(stripped)]
        return f"{leading_ws}{prefix}{stripped}"
    
    def _rule4_flatten_gradient(self, text: str) -> str:
        """Rule 4: Flatten early authority gradient"""
        prefix = "Based on available information, "
        stripped = text.lstrip()
        if stripped.lower().startswith(prefix.lower()):
            return text
        leading_ws = text[: len(text) - len(stripped)]
        return f"{leading_ws}{prefix}{stripped}"
    
    def _rule5_add_conditionals(self, text: str) -> str:
        """Rule 5: Add conditional framing using linguistic analysis
        
        Uses a hybrid approach:
        1. Detect existing hedges/qualifiers (avoid double-hedging)
        2. Parse sentence structure to identify transformation opportunities
        3. Apply transformations based on linguistic patterns, not just string matching
        """
        # Use signal estimation to determine if conditionals are needed
        from signal_estimation import signal_estimator
        
        evidential_risk = signal_estimator.estimate_evidential_risk(text)
        
        # Only add conditionals if evidential risk is high
        if evidential_risk < 0.6:
            return text
        
        # Check for existing hedges or temporal qualifiers - if already hedged, don't modify
        if self._has_existing_qualifiers(text):
            return text  # Already hedged, don't modify
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'([.!?]+)', text) if s.strip()]
        if not sentences:
            return text
        
        # Transform the first sentence using linguistic analysis
        first_sentence = sentences[0]
        transformed_first = self._transform_sentence_linguistically(first_sentence)
        
        # Replace first sentence with transformed version
        if transformed_first != first_sentence:
            sentences[0] = transformed_first
        
        # Reconstruct text
        result = ''.join(sentences)
        
        # Add a meaningful caveat at the end if not already present
        # Skip caveats for explanatory constructions (this is, that is, it is)
        if (not any(marker in text.lower() for marker in ['however', 'though', 'but', 'exceptions', 'may vary', 'can vary']) and
            not text.lower().strip().startswith(('this is', 'that is', 'it is', 'there is', 'here is'))):
            result = self._add_meaningful_caveat(result, text.lower())
        
        return result
    
    def _has_existing_qualifiers(self, text: str) -> bool:
        """Check if text already contains hedges, temporal qualifiers, or epistemic markers"""
        existing_qualifiers = [
            # Modal hedges
            'typically', 'generally', 'often', 'usually', 'commonly', 'frequently',
            'in most cases', 'in many cases', 'in some cases', 'may', 'might', 'could',
            'possibly', 'likely', 'probably', 'sometimes', 'occasionally',
            # Temporal qualifiers that already indicate uncertainty/limitation
            'as of', 'according to', 'based on', 'up to', 'until', 'as far as',
            'to my knowledge', 'in my understanding', 'from what i know',
            # Epistemic markers
            'it appears', 'it seems', 'appears to be', 'seems to be',
            'i believe', 'i think', 'in my opinion', 'in my view',
            'it is believed', 'it is thought', 'it is considered',
            # Uncertainty markers
            'approximately', 'roughly', 'about', 'around', 'nearly',
            'tend to', 'tends to', 'inclined to'
        ]
        
        text_lower = text.lower()
        return any(qualifier in text_lower for qualifier in existing_qualifiers)
    
    def _transform_sentence_linguistically(self, sentence: str) -> str:
        """Transform sentence using linguistic analysis rather than simple pattern matching
        
        This method analyzes the sentence structure to identify:
        - Copula constructions (X is Y)
        - Modal verbs (should, must, will)
        - Superlatives (best, worst, most)
        - Imperatives (use, try, avoid)
        - Universal quantifiers (always, never, all)
        
        And applies appropriate transformations based on the linguistic context.
        """
        sentence = sentence.strip()
        if not sentence:
            return sentence
        
        # Tokenize into words while preserving case
        words = sentence.split()
        if not words:
            return sentence
        
        sentence_lower = sentence.lower()
        
        # Pattern 1: Copula with superlative (X is the best/worst Y)
        if self._has_copula_superlative(sentence_lower):
            return self._transform_copula_superlative(sentence, words)
        
        # Pattern 2: Modal obligation (should, must)
        if self._has_modal_obligation(words):
            return self._transform_modal_obligation(sentence, words)
        
        # Pattern 3: Future prediction (will)
        if self._has_future_modal(words):
            return self._transform_future_modal(sentence, words)
        
        # Pattern 4: Universal quantifiers (always, never, all, every)
        if self._has_universal_quantifier(words):
            return self._transform_universal_quantifier(sentence, words)
        
        # Pattern 5: Imperative (starts with verb)
        if self._is_imperative(sentence_lower, words):
            return self._transform_imperative(sentence, words)
        
        # Pattern 6: Copula without superlative (X is Y)
        if self._has_simple_copula(sentence_lower):
            return self._transform_simple_copula(sentence)
        
        return sentence
    
    def _has_copula_superlative(self, sentence_lower: str) -> bool:
        """Check if sentence has copula + superlative (is the best/worst/most)
        
        Excludes sentences starting with 'this is', 'that is', 'it is' as these
        are typically explanatory rather than assertive.
        """
        # Exclude explanatory constructions
        if sentence_lower.startswith(('this is', 'that is', 'it is', 'there is', 'here is')):
            return False
        
        copula_superlatives = [
            ' is the best ', ' is the worst ', ' is the most ',
            ' are the best ', ' are the worst ', ' are the most ',
            ' was the best ', ' was the worst ', ' was the most ',
            ' were the best ', ' were the worst ', ' were the most '
        ]
        return any(pattern in sentence_lower for pattern in copula_superlatives)
    
    def _transform_copula_superlative(self, sentence: str, words: list) -> str:
        """Transform 'X is the best Y' -> 'X is often considered the best Y'"""
        replacements = [
            (' is the best ', ' is often considered the best '),
            (' is the worst ', ' is often considered the worst '),
            (' is the most ', ' is often the most '),
            (' are the best ', ' are often considered the best '),
            (' are the worst ', ' are often considered the worst '),
            (' are the most ', ' are often the most '),
            (' was the best ', ' was often considered the best '),
            (' was the worst ', ' was often considered the worst '),
            (' was the most ', ' was often the most '),
            (' were the best ', ' were often considered the best '),
            (' were the worst ', ' were often considered the worst '),
            (' were the most ', ' were often the most ')
        ]
        
        result = sentence
        for old, new in replacements:
            if old in sentence.lower():
                # Preserve case by finding the actual occurrence
                idx = sentence.lower().find(old)
                if idx != -1:
                    result = sentence[:idx] + new + sentence[idx + len(old):]
                    break
        return result
    
    def _has_modal_obligation(self, words: list) -> bool:
        """Check if sentence contains modal obligation (should, must, ought to)"""
        modal_obligations = ['should', 'must', 'ought']
        return any(word.lower() in modal_obligations for word in words)
    
    def _transform_modal_obligation(self, sentence: str, words: list) -> str:
        """Transform modal obligations to suggestions"""
        transformations = {
            'should': 'may want to',
            'Should': 'May want to',
            'must': 'typically need to',
            'Must': 'Typically need to',
            'ought to': 'may want to',
            'Ought to': 'May want to'
        }
        
        result = sentence
        for old, new in transformations.items():
            if old in sentence:
                result = result.replace(old, new, 1)
                break
        return result
    
    def _has_future_modal(self, words: list) -> bool:
        """Check if sentence contains future modal 'will'"""
        return 'will' in [w.lower() for w in words]
    
    def _transform_future_modal(self, sentence: str, words: list) -> str:
        """Transform 'will' to 'will often' or 'will typically'"""
        # Find position of 'will'
        for i, word in enumerate(words):
            if word.lower() == 'will':
                # Insert 'often' after 'will'
                words_copy = words.copy()
                words_copy.insert(i + 1, 'often')
                return ' '.join(words_copy)
        return sentence
    
    def _has_universal_quantifier(self, words: list) -> bool:
        """Check if sentence starts with universal quantifier"""
        if not words:
            return False
        first_word = words[0].lower()
        return first_word in ['always', 'never', 'all', 'every', 'none']
    
    def _transform_universal_quantifier(self, sentence: str, words: list) -> str:
        """Transform universal quantifiers to qualified versions"""
        transformations = {
            'Always': 'Usually',
            'always': 'usually',
            'Never': 'Rarely',
            'never': 'rarely',
            'All': 'Most',
            'all': 'most',
            'Every': 'Most',
            'every': 'most',
            'None': 'Few',
            'none': 'few'
        }
        
        first_word = words[0]
        if first_word in transformations:
            words[0] = transformations[first_word]
            return ' '.join(words)
        return sentence
    
    def _is_imperative(self, sentence_lower: str, words: list) -> bool:
        """Check if sentence is imperative (command)"""
        if not words:
            return False
        
        imperative_verbs = [
            'use', 'try', 'avoid', 'choose', 'select', 'pick',
            'install', 'configure', 'set', 'enable', 'disable',
            'run', 'execute', 'start', 'stop', 'create', 'delete',
            'add', 'remove', 'update', 'upgrade', 'download'
        ]
        
        first_word = words[0].lower()
        return first_word in imperative_verbs
    
    def _transform_imperative(self, sentence: str, words: list) -> str:
        """Transform imperative to suggestion"""
        # Convert to gerund form with "Consider"
        verb = words[0]
        rest = ' '.join(words[1:])
        
        verb_to_gerund = {
            'use': 'using', 'try': 'trying', 'avoid': 'avoiding',
            'choose': 'choosing', 'select': 'selecting', 'pick': 'picking',
            'install': 'installing', 'configure': 'configuring',
            'set': 'setting', 'enable': 'enabling', 'disable': 'disabling',
            'run': 'running', 'execute': 'executing', 'start': 'starting',
            'stop': 'stopping', 'create': 'creating', 'delete': 'deleting',
            'add': 'adding', 'remove': 'removing', 'update': 'updating',
            'upgrade': 'upgrading', 'download': 'downloading'
        }
        
        verb_lower = verb.lower()
        if verb_lower in verb_to_gerund:
            gerund = verb_to_gerund[verb_lower]
            # Preserve capitalization
            if verb[0].isupper():
                return f"Consider {gerund} {rest}"
            else:
                return f"consider {gerund} {rest}"
        
        return sentence
    
    def _has_simple_copula(self, sentence_lower: str) -> bool:
        """Check if sentence has simple copula (is/are) without superlative"""
        copulas = [' is ', ' are ', ' was ', ' were ']
        # Exclude sentences starting with 'this is', 'that is', 'it is'
        if sentence_lower.startswith(('this is', 'that is', 'it is', 'there is', 'there are')):
            return False
        return any(copula in sentence_lower for copula in copulas)
    
    def _transform_simple_copula(self, sentence: str) -> str:
        """Transform 'X is Y' to 'X is often Y'"""
        copula_patterns = [
            (' is ', ' is often '),
            (' are ', ' are often '),
            (' was ', ' was often '),
            (' were ', ' were often ')
        ]
        
        result = sentence
        for old, new in copula_patterns:
            if old in sentence.lower():
                idx = sentence.lower().find(old)
                if idx != -1:
                    result = sentence[:idx] + new + sentence[idx + len(old):]
                    break
        return result
    
    def _restructure_sentence_with_conditional(self, sentence: str) -> str:
        """Legacy method - now delegates to linguistic analysis"""
        return self._transform_sentence_linguistically(sentence)
    
    def _add_meaningful_caveat(self, text: str, text_lower: str) -> str:
        """Add a meaningful caveat based on content type"""
        # Determine content type and add appropriate caveat
        
        # For technical/procedural content
        if any(word in text_lower for word in ['system', 'process', 'method', 'approach', 'implementation']):
            caveat = " Implementation details may vary based on specific requirements."
        
        # For recommendations
        elif any(word in text_lower for word in ['recommend', 'suggest', 'should', 'consider', 'may want']):
            caveat = " Your specific situation may require a different approach."
        
        # For factual statements
        elif any(word in text_lower for word in [' is ', ' are ', ' will ', ' can ']):
            caveat = " Exceptions may apply in specific contexts."
        
        # Default
        else:
            caveat = " Individual circumstances may vary."
        
        # Add caveat with proper punctuation
        text = text.rstrip()
        if not text.endswith('.'):
            text += '.'
        
        return f"{text}{caveat}"
    
    def shape_commitment(
        self,
        response: str,
        prompt: str,
        context: str = "",
        sigma: float = 1.0
    ) -> Tuple[str, CommitmentRisk, Dict]:
        """Main entry point: evaluate and shape commitment
        
        Args:
            response: Original response
            prompt: Original prompt
            context: Available context
            sigma: Admissibility signal from CP Type-1
            
        Returns:
            (shaped_response, risk, debug_info)
        """
        # Start overall timing for performance metrics
        overall_start_time = time.perf_counter()
        
        # Get dashboard metrics collector
        dashboard_metrics = get_dashboard_metrics()
        
        # Step 1: Commitment-Actuality Gate (-(z*) check) with timing
        kappa_start_time = time.perf_counter()
        is_commitment_bearing = self.commitment_classifier.is_commitment_bearing(response, prompt)
        kappa_computation_time_ms = (time.perf_counter() - kappa_start_time) * 1000
        kappa = 1 if is_commitment_bearing else 0
        
        print(f"[IFCS] Commitment-actuality classification: -(z*)={kappa} ({'commitment-bearing' if kappa else 'non-commitment-bearing'})")
        print(f"[IFCS] -(z*) computation time: {kappa_computation_time_ms:.2f}ms")
        
        # Compute commitment risk with timing
        risk_start_time = time.perf_counter()
        risk = self.compute_commitment_risk(response, prompt, context)
        risk_computation_time_ms = (time.perf_counter() - risk_start_time) * 1000
        
        # C6 COMPLIANCE: No domain detection performed
        rho_used = getattr(self, "_last_rho_used", self.config.rho)
        structural_signals = getattr(self, "_last_structural_signals", {})
        rho_reason = getattr(self, "_last_rho_reason", "")
        
        print(f"[IFCS] Commitment risk: {risk}")
        print(f"[IFCS] Risk computation time: {risk_computation_time_ms:.2f}ms")
        
        threshold_tier = (
            "strict" if rho_used <= 0.30 else
            "moderate" if rho_used <= 0.35 else
            "default"
        )
        
        # Enhanced debug info with comprehensive logging data
        debug_info = {
            'risk': risk,
            'rho': rho_used,
            'rho_reason': rho_reason,
            'structural_signals': structural_signals,
            'threshold_tier': threshold_tier,
            'adaptive_active': True,
            'kappa': kappa,
            'commitment_bearing': is_commitment_bearing,
            'intervened': False,  # Will be updated below
            'sigma': sigma,
            # Performance metrics for -(z*) computation
            'kappa_computation_time_ms': kappa_computation_time_ms,
            'risk_computation_time_ms': risk_computation_time_ms,
            # Classification logs and metrics
            'classification_logs_count': len(self.commitment_classifier.classification_logs),
            'non_commitment_metrics_count': len(self.commitment_classifier.non_commitment_metrics),
            'classifier_performance_summary': self.commitment_classifier.get_performance_summary()
        }
        
        # Step 2: Three-part firing condition: -(z*) - - - R(z*) > - - -(z*) = 1
        if kappa == 0:
            # Non-commitment-bearing context: observation without intervention
            overall_time_ms = (time.perf_counter() - overall_start_time) * 1000
            
            # Calculate latency improvement (estimated time saved by avoiding IFCS processing)
            estimated_ifcs_processing_time_ms = 50.0  # Typical IFCS processing time
            latency_improvement_ms = max(0, estimated_ifcs_processing_time_ms - overall_time_ms)
            
            print(f"[IFCS] NON-INTERVENTION: -(z*)=0 (non-commitment-bearing context)")
            print(f"[IFCS] Total processing time: {overall_time_ms:.2f}ms (avoided IFCS transformation overhead)")
            print(f"[IFCS] Estimated latency improvement: {latency_improvement_ms:.2f}ms")
            
            debug_info['intervened'] = False
            debug_info['intervention_reason'] = 'non-commitment-bearing'
            debug_info['total_processing_time_ms'] = overall_time_ms
            debug_info['transformation_time_ms'] = 0  # No transformation applied
            debug_info['latency_improvement_ms'] = latency_improvement_ms
            
            # Record dashboard metrics for non-commitment-bearing context
            dashboard_metrics.record_classification(
                kappa_value=0,
                computation_time_ms=kappa_computation_time_ms,
                is_error=False,
                is_fallback=False,
                latency_improvement_ms=latency_improvement_ms,
                total_processing_time_ms=overall_time_ms,
                metadata={
                    'intervention_reason': 'non-commitment-bearing',
                    'response_length': len(response),
                    'prompt_length': len(prompt) if prompt else 0
                }
            )
            
            return response, risk, debug_info
        
        # Determine if intervention needed (commitment-bearing contexts only)
        should_intervene = self.should_intervene(risk, sigma, rho_used)
        debug_info['intervened'] = should_intervene
        
        if should_intervene:
            print(f"[IFCS] INTERVENING: -={sigma:.3f} - - - R={risk.R:.3f} > -={debug_info['rho']:.3f} - -=1")
            
            # Apply transformations with timing
            transformation_start_time = time.perf_counter()
            shaped_response = self.apply_transformation_rules(response, risk)
            transformation_time_ms = (time.perf_counter() - transformation_start_time) * 1000
            
            # Recompute risk after shaping
            risk_after = self.compute_commitment_risk(shaped_response, prompt, context)
            reduction = ((risk.R - risk_after.R) / risk.R * 100) if risk.R > 0 else 0
            
            overall_time_ms = (time.perf_counter() - overall_start_time) * 1000
            
            print(f"[IFCS] Risk after shaping: {risk_after}")
            print(f"[IFCS] Commitment reduction: {reduction:.1f}%")
            print(f"[IFCS] Transformation time: {transformation_time_ms:.2f}ms")
            print(f"[IFCS] Total processing time: {overall_time_ms:.2f}ms")
            
            debug_info['risk_after'] = risk_after
            debug_info['reduction_percent'] = reduction
            debug_info['intervention_reason'] = 'commitment_risk_exceeded'
            debug_info['transformation_time_ms'] = transformation_time_ms
            debug_info['total_processing_time_ms'] = overall_time_ms
            
            # Record dashboard metrics for commitment-bearing context with intervention
            dashboard_metrics.record_classification(
                kappa_value=1,
                computation_time_ms=kappa_computation_time_ms,
                is_error=False,
                is_fallback=False,
                latency_improvement_ms=0,  # No latency improvement for intervened contexts
                total_processing_time_ms=overall_time_ms,
                metadata={
                    'intervention_reason': 'commitment_risk_exceeded',
                    'transformation_time_ms': transformation_time_ms,
                    'risk_reduction_percent': reduction,
                    'response_length': len(response),
                    'shaped_response_length': len(shaped_response)
                }
            )
            
            return shaped_response, risk, debug_info
        else:
            overall_time_ms = (time.perf_counter() - overall_start_time) * 1000
            
            print(f"[IFCS] PASS: R={risk.R:.3f} - -={debug_info['rho']:.3f} (commitment-bearing but low risk)")
            print(f"[IFCS] Total processing time: {overall_time_ms:.2f}ms")
            
            debug_info['intervention_reason'] = 'commitment_risk_acceptable'
            debug_info['total_processing_time_ms'] = overall_time_ms
            debug_info['transformation_time_ms'] = 0  # No transformation applied
            
            # Record dashboard metrics for commitment-bearing context without intervention
            dashboard_metrics.record_classification(
                kappa_value=1,
                computation_time_ms=kappa_computation_time_ms,
                is_error=False,
                is_fallback=False,
                latency_improvement_ms=0,  # No latency improvement
                total_processing_time_ms=overall_time_ms,
                metadata={
                    'intervention_reason': 'commitment_risk_acceptable',
                    'risk_level': risk.R,
                    'response_length': len(response)
                }
            )
            
            return response, risk, debug_info
    def get_config(self) -> IFCSConfig:
        """Get IFCS configuration
        
        Returns:
            IFCS configuration object
        """
        return self.config
    
    def get_classification_logs(self, limit: Optional[int] = None) -> List[CommitmentActualityLog]:
        """Get recent -(z*) classification logs for monitoring and analysis
        
        Args:
            limit: Maximum number of recent logs to return
            
        Returns:
            List of CommitmentActualityLog entries
        """
        return self.commitment_classifier.get_classification_logs(limit)
    
    def get_non_commitment_metrics(self, limit: Optional[int] = None) -> List[NonCommitmentBearingMetrics]:
        """Get recent non-commitment-bearing context metrics
        
        Args:
            limit: Maximum number of recent metrics to return
            
        Returns:
            List of NonCommitmentBearingMetrics entries
        """
        return self.commitment_classifier.get_non_commitment_metrics(limit)
    
    def get_kappa_performance_summary(self) -> Dict:
        """Get performance summary for -(z*) computation
        
        Returns:
            Dictionary with performance metrics including:
            - total_classifications: Total number of classifications performed
            - commitment_bearing_ratio: Ratio of commitment-bearing to total classifications
            - avg_computation_time_ms: Average computation time in milliseconds
            - latency_metrics: Min, max, and total computation times
        """
        return self.commitment_classifier.get_performance_summary()
    
    def get_dashboard_metrics(self) -> Dict:
        """Get comprehensive dashboard metrics
        
        Returns:
            Dictionary with all dashboard-relevant metrics
        """
        dashboard_metrics = get_dashboard_metrics()
        return {
            'snapshot': dashboard_metrics.get_dashboard_snapshot().to_dict(),
            'commitment_bearing_ratios': dashboard_metrics.get_commitment_bearing_ratio_over_time(),
            'latency_improvements': dashboard_metrics.get_latency_improvements(),
            'kappa_performance': dashboard_metrics.get_kappa_performance_metrics(),
            'error_and_fallback_metrics': dashboard_metrics.get_error_and_fallback_metrics(),
            'active_alerts': [alert.to_dict() for alert in dashboard_metrics.get_active_alerts()]
        }
    
    def export_dashboard_data(self, filepath: str, minutes: int = 60) -> None:
        """Export comprehensive dashboard data to JSON file
        
        Args:
            filepath: Path to output JSON file
            minutes: Time window for data export
        """
        dashboard_metrics = get_dashboard_metrics()
        dashboard_metrics.export_dashboard_data(filepath, minutes)
    
    def print_dashboard_summary(self) -> None:
        """Print comprehensive dashboard summary"""
        dashboard_metrics = get_dashboard_metrics()
        dashboard_metrics.print_dashboard_summary()
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a dashboard alert
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was found and resolved
        """
        dashboard_metrics = get_dashboard_metrics()
        return dashboard_metrics.resolve_alert(alert_id)
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Update dashboard alert thresholds
        
        Args:
            thresholds: Dictionary of threshold updates
        """
        dashboard_metrics = get_dashboard_metrics()
        dashboard_metrics.update_alert_thresholds(thresholds)
    
    def export_classification_logs_json(self, filepath: str, limit: Optional[int] = None) -> None:
        """Export classification logs to JSON file for external analysis
        
        Args:
            filepath: Path to output JSON file
            limit: Maximum number of recent logs to export
        """
        logs = self.get_classification_logs(limit)
        log_data = [asdict(log) for log in logs]
        
        with open(filepath, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_logs': len(log_data),
                'classification_logs': log_data
            }, f, indent=2)
        
        print(f"[IFCS] Exported {len(log_data)} classification logs to {filepath}")
    
    def export_non_commitment_metrics_json(self, filepath: str, limit: Optional[int] = None) -> None:
        """Export non-commitment-bearing metrics to JSON file for analysis
        
        Args:
            filepath: Path to output JSON file
            limit: Maximum number of recent metrics to export
        """
        metrics = self.get_non_commitment_metrics(limit)
        metrics_data = [asdict(metric) for metric in metrics]
        
        with open(filepath, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_metrics': len(metrics_data),
                'non_commitment_metrics': metrics_data
            }, f, indent=2)
        
        print(f"[IFCS] Exported {len(metrics_data)} non-commitment metrics to {filepath}")
    
    def print_performance_report(self) -> None:
        """Print comprehensive performance report for -(z*) computation"""
        summary = self.get_kappa_performance_summary()
        
        if summary['total_classifications'] == 0:
            print("[IFCS] No -(z*) classifications performed yet.")
            return
        
        print("\n" + "="*60)
        print("-(z*) COMMITMENT-ACTUALITY GATE PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Classifications: {summary['total_classifications']}")
        print(f"Commitment-Bearing: {summary['commitment_bearing_count']} ({summary['commitment_bearing_ratio']:.1%})")
        print(f"Non-Commitment-Bearing: {summary['non_commitment_bearing_count']} ({1-summary['commitment_bearing_ratio']:.1%})")
        print(f"Average Computation Time: {summary['avg_computation_time_ms']:.2f}ms")
        print(f"Min/Max Computation Time: {summary['min_computation_time_ms']:.2f}ms / {summary['max_computation_time_ms']:.2f}ms")
        print(f"Total Computation Time: {summary['total_computation_time_ms']:.2f}ms")
        
        # Performance analysis
        if summary['avg_computation_time_ms'] < 50:
            print(f"- Performance Target Met: {summary['avg_computation_time_ms']:.2f}ms < 50ms target")
        else:
            print(f"--  Performance Target Missed: {summary['avg_computation_time_ms']:.2f}ms > 50ms target")
        
        print("="*60)
    
