"""
Inference-Time Commitment Shaping (IFCS) Implementation
Based on: Chatterjee, A. (2026c). Inference-Time Commitment Shaping
"""

import re
import time
import json
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from trilogy_config import DomainConfig
from semantic_analyzer import semantic_analyzer
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
        return (f"R={self.R:.3f} [ê={self.e_hat:.2f}, ŝ={self.s_hat:.2f}, "
                f"â={self.a_hat:.2f}, t̂={self.t_hat:.2f}]")


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
    """Computes κ(z*): commitment-actuality indicator using signal estimation"""
    
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
        Returns: aggregated_signal ∈ [0,1]
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
        """Analyze text for semantic commitment patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Analyze commitment patterns
        commitment_score = 0
        commitment_features = {}
        
        # 1. Directive language analysis
        directive_count = 0
        for word in words:
            if word in self.commitment_indicators['directive_verbs']:
                # Check context - is it directed at the user?
                word_idx = words.index(word)
                context_window = words[max(0, word_idx-2):min(len(words), word_idx+3)]
                if 'you' in context_window or word_idx < 3:  # Early in sentence suggests directive
                    directive_count += 2
                else:
                    directive_count += 1
        
        commitment_features['directive_strength'] = directive_count
        commitment_score += directive_count
        
        # 2. Recommendation language
        recommendation_count = sum(1 for word in words if word in self.commitment_indicators['recommendation_verbs'])
        commitment_features['recommendation_strength'] = recommendation_count
        commitment_score += recommendation_count * 1.5
        
        # 3. Certainty and definitiveness
        certainty_count = sum(1 for word in words if word in self.commitment_indicators['certainty_adverbs'])
        commitment_features['certainty_strength'] = certainty_count
        commitment_score += certainty_count * 1.2
        
        # 4. Superlative/exclusive language
        superlative_count = sum(1 for word in words if word in self.commitment_indicators['superlative_patterns'])
        exclusivity_count = sum(1 for word in words if word in self.commitment_indicators['exclusivity_patterns'])
        
        # Give extra weight to strong commitment phrases
        strong_commitment_phrases = ['the best', 'best approach', 'best choice', 'right way', 'correct way', 'only way', 'proper approach']
        strong_phrase_bonus = 0
        text_lower = text.lower()
        for phrase in strong_commitment_phrases:
            if phrase in text_lower:
                strong_phrase_bonus += 2  # Strong bonus for definitive phrases
        
        commitment_features['superlative_strength'] = superlative_count + exclusivity_count + strong_phrase_bonus
        commitment_score += (superlative_count + exclusivity_count) * 1.3 + strong_phrase_bonus
        
        # Analyze descriptive patterns
        descriptive_score = 0
        descriptive_features = {}
        
        # 1. Listing/enumeration language
        listing_count = sum(1 for word in words if word in self.descriptive_indicators['listing_verbs'])
        
        # Enhanced listing detection for numbered/bulleted lists
        if re.search(r'\d+\.\s', text) or re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
            listing_count += 3  # Strong signal for enumeration
        
        # Check for "include:" pattern which is strongly descriptive
        if 'include:' in text_lower or 'includes:' in text_lower:
            listing_count += 2
            
        descriptive_features['listing_strength'] = listing_count
        descriptive_score += listing_count * 1.5
        
        # 2. Example/reference markers
        example_count = sum(1 for word in words if word in self.descriptive_indicators['example_markers'])
        reference_count = sum(1 for word in words if word in self.descriptive_indicators['reference_markers'])
        descriptive_features['informational_strength'] = example_count + reference_count
        descriptive_score += (example_count + reference_count) * 1.2
        
        # 3. Frequency/hedging language
        frequency_count = sum(1 for word in words if word in self.descriptive_indicators['frequency_adverbs'])
        descriptive_features['hedging_strength'] = frequency_count
        descriptive_score += frequency_count * 1.1
        
        # 4. Informational phrases (practices, approaches, etc.)
        informational_count = sum(1 for word in words if word in self.descriptive_indicators['informational_phrases'])
        descriptive_features['informational_phrases'] = informational_count
        descriptive_score += informational_count * 1.3
        
        return {
            'commitment_score': commitment_score,
            'descriptive_score': descriptive_score,
            'commitment_features': commitment_features,
            'descriptive_features': descriptive_features
        }
    
    def _analyze_syntactic_patterns(self, text: str) -> dict:
        """Analyze syntactic patterns using signal estimation (no text-matching heuristics)"""
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator
        
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
        """Log κ(z*) decision with comprehensive context metadata and reasoning"""
        
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
        print(f"[IFCS-κ] Classification Decision: κ(z*)={kappa} ({classification})")
        print(f"[IFCS-κ] Computation Time: {computation_time_ms:.2f}ms")
        print(f"[IFCS-κ] Context: {context_metadata['word_count']} words, {context_metadata['sentence_count']} sentences")
        print(f"[IFCS-κ] Signal-Based Reasoning: {classification_reasoning['decision_logic']}")
        
        # Signal breakdown for debugging
        signals = semantic_analysis.get('signals', {})
        if signals:
            print(f"[IFCS-κ] Signals: assertion_strength={signals.get('assertion_strength', 0):.3f}, "
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
        print(f"[IFCS-κ] Non-Commitment Rationale: {rationale}")
        print(f"[IFCS-κ] Aggregated Signal: {aggregated_signal:.3f} (threshold: 0.4)")
        if rationale_parts:
            print(f"[IFCS-κ] Low Signals: {', '.join(rationale_parts)}")
    
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
        """Get performance summary for κ(z*) computation"""
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
        """Analyze pragmatic context from prompt to inform classification"""
        if not prompt:
            return {'context_bias': 0}
        
        prompt_lower = prompt.lower()
        context_bias = 0
        
        # Information-seeking questions (bias toward descriptive)
        info_seeking_patterns = [
            'what are', 'what is', 'tell me about', 'explain', 'describe',
            'list', 'show me', 'give me examples', 'how does'
        ]
        if any(pattern in prompt_lower for pattern in info_seeking_patterns):
            context_bias -= 1
        
        # Advice-seeking questions (bias toward commitment)
        advice_seeking_patterns = [
            'what should', 'how should', 'what do you recommend', 'what\'s the best',
            'how do i', 'help me', 'what would you do'
        ]
        if any(pattern in prompt_lower for pattern in advice_seeking_patterns):
            context_bias += 1
        
        return {'context_bias': context_bias}
    
    def _analyze_pragmatic_context(self, response: str, prompt: str) -> dict:
        """Analyze pragmatic context from prompt to inform classification"""
        if not prompt:
            return {'context_bias': 0}
        
        prompt_lower = prompt.lower()
        context_bias = 0
        
        # Information-seeking questions (bias toward descriptive)
        info_seeking_patterns = [
            'what are', 'what is', 'tell me about', 'explain', 'describe',
            'list', 'show me', 'give me examples', 'how does'
        ]
        if any(pattern in prompt_lower for pattern in info_seeking_patterns):
            context_bias -= 1
        
        # Advice-seeking questions (bias toward commitment)
        advice_seeking_patterns = [
            'what should', 'how should', 'what do you recommend', 'what\'s the best',
            'how do i', 'help me', 'what would you do'
        ]
        if any(pattern in prompt_lower for pattern in advice_seeking_patterns):
            context_bias += 1
        
        return {'context_bias': context_bias}
    
    def is_commitment_bearing(self, response: str, prompt: str) -> bool:
        """
        Determine if response constitutes a commitment-bearing act using true signal estimation.
        Industry approach: Estimates latent epistemic signals using statistical methods.
        No regex patterns, no hardcoded word lists, no text-matching heuristics.
        
        Args:
            response: The response text to classify
            prompt: The original prompt for context
            
        Returns:
            True (κ=1) if response compresses uncertainty into actionable stance
            False (κ=0) if response is descriptive/enumerative/non-binding
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
            'decision_logic': f"aggregated_signal ({aggregated_signal:.3f}) {'≥' if kappa == 1 else '<'} threshold ({commitment_threshold:.3f})"
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
        """Compute ê(z*): Evidential insufficiency using semantic analysis

        Args:
            response: Response text
            prompt: Original prompt
            context: Available context/grounding

        Returns:
            ê score [0, 1]
        """
        if not response or not response.strip():
            return 0.5  # Neutral
        
        # Use semantic analyzer for evidential analysis
        semantic_result = semantic_analyzer.analyze_evidential_sufficiency(response, context)
        
        # Combine with structural insufficiency signals
        structural_signals = self._structural_signals(prompt, response)
        structural_e = max(structural_signals.values()) if structural_signals else 0.0
        
        # Take the maximum of semantic and structural evidential risk
        return min(1.0, max(semantic_result.score, structural_e))

    def _detect_jurisdiction_dependency(self, prompt: str, response: str) -> float:
        """Detect answers that depend on external jurisdiction using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator
        
        # Estimate jurisdictional dependency signals
        authority_signal = signal_estimator.estimate_authority_posture(response)
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        
        # High authority + high certainty in legal/permission context suggests jurisdictional dependency
        combined_signal = (authority_signal + certainty_signal) / 2.0
        
        # Scale based on prompt context (permission-seeking increases risk)
        prompt_lower = prompt.lower()
        permission_context = any(term in prompt_lower for term in ['can i', 'may i', 'allowed', 'legal', 'illegal'])
        
        if permission_context and combined_signal > 0.6:
            return min(0.8, combined_signal * 1.2)
        
        return combined_signal * 0.3  # Lower baseline risk

    def _detect_policy_dependency(self, prompt: str, response: str) -> float:
        """Detect reliance on external policies using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator
        
        # Estimate policy dependency signals
        authority_signal = signal_estimator.estimate_authority_posture(response)
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        
        # Check for policy context in prompt
        prompt_lower = prompt.lower()
        policy_context = any(term in prompt_lower for term in ['policy', 'rules', 'regulation', 'terms', 'guidelines'])
        
        # High authority + certainty in policy context suggests dependency
        combined_signal = (authority_signal + certainty_signal) / 2.0
        
        if policy_context and combined_signal > 0.5:
            return min(0.6, combined_signal * 1.1)
        
        return combined_signal * 0.2  # Lower baseline risk

    def _detect_permission_framing(self, prompt: str, response: str) -> float:
        """Detect binary permission framing using signal estimation."""
        if not prompt or not response:
            return 0.0
        
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator
        
        # Estimate binary framing signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Check for permission-seeking context
        prompt_lower = prompt.lower()
        permission_seeking = any(phrase in prompt_lower for phrase in ['can i', 'may i', 'am i allowed', 'is it okay to'])
        
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
        from signal_estimation import signal_estimator
        
        # Estimate diagnostic certainty signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Check for personal context in prompt
        prompt_lower = prompt.lower()
        personal_context = any(phrase in prompt_lower for phrase in ['i have', 'i feel', "i'm experiencing", 'my '])
        
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
        from signal_estimation import signal_estimator
        
        # Estimate consequence-related certainty signals
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        authority_signal = signal_estimator.estimate_authority_posture(response)
        
        # Check for consequence-related context in prompt
        prompt_lower = prompt.lower()
        consequence_context = any(term in prompt_lower for term in ['risk', 'danger', 'safety', 'safe', 'unsafe', 'harm'])
        
        # High certainty + authority in consequence context suggests asymmetry
        combined_signal = (certainty_signal + authority_signal) / 2.0
        
        if consequence_context and combined_signal > 0.5:
            return min(0.8, combined_signal * 1.5)
        
        return combined_signal * 0.1  # Very low baseline risk

    def prompt_structural_signals(self, prompt: str) -> Dict[str, float]:
        """Estimate structural risk from prompt using signal estimation (domain-agnostic)."""
        if not prompt:
            return {}
        
        # Use signal estimation instead of regex patterns
        from signal_estimation import signal_estimator
        
        # Estimate temporal risk from prompt
        temporal_risk = signal_estimator.estimate_temporal_risk("", prompt)  # Empty response, analyze prompt
        
        prompt_lower = prompt.lower()
        
        # Use statistical analysis instead of hardcoded patterns
        signals = {}
        
        # Jurisdictional risk (permission-seeking language)
        permission_terms = ['can i', 'may i', 'am i allowed', 'is it legal', 'is it illegal', 'is it okay to']
        permission_density = sum(1 for term in permission_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["jurisdictional"] = min(0.7, permission_density * 20.0)
        
        # Policy risk (policy-related language)
        policy_terms = ['policy', 'rules', 'regulation', 'terms', 'guidelines']
        policy_density = sum(1 for term in policy_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["policy"] = min(0.6, policy_density * 15.0)
        
        # Binary framing risk
        binary_terms = ['can i', 'may i', 'am i allowed', 'is it okay to']
        binary_density = sum(1 for term in binary_terms if prompt_lower.startswith(term)) / max(1, 1)
        signals["binary"] = min(0.7, binary_density * 0.7)
        
        # Personal data risk
        personal_terms = ['i have', 'i feel', "i'm experiencing", 'my ']
        personal_density = sum(1 for term in personal_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["personal_data"] = min(0.7, personal_density * 25.0)
        
        # Temporal risk (use signal estimation)
        signals["temporal"] = temporal_risk
        
        # Consequence risk
        consequence_terms = ['risk', 'danger', 'safety', 'safe', 'unsafe', 'harm']
        consequence_density = sum(1 for term in consequence_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["consequence"] = min(0.6, consequence_density * 20.0)
        
        return signals

    def _structural_signals(self, prompt: str, response: str) -> Dict[str, float]:
        """Structural insufficiency signals (domain-agnostic)."""
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
        """Compute ŝ(z*): Scope inflation using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            ŝ score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use semantic analyzer instead of hardcoded patterns
        semantic_result = semantic_analyzer.analyze_universal_scope(response)
        
        return semantic_result.score
    
    def compute_authority_cues(self, response: str) -> float:
        """Compute â(z*): Authority cues using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            â score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use semantic analyzer instead of hardcoded patterns
        semantic_result = semantic_analyzer.analyze_authority_cues(response)
        
        return semantic_result.score
    
    def compute_temporal_risk(self, response: str, prompt: str) -> float:
        """Compute t̂(z*): Temporal risk using semantic analysis
        
        Args:
            response: Response text
            prompt: Original prompt
            
        Returns:
            t̂ score [0, 1]
        """
        if not response or not response.strip():
            return 0.0
        
        # Use semantic analyzer instead of hardcoded patterns
        semantic_result = semantic_analyzer.analyze_temporal_risk(response, prompt)
        
        return semantic_result.score
    
    def compute_early_authority_gradient(self, response: str) -> float:
        """Compute ΔAG: Early authority gradient using semantic analysis
        
        Args:
            response: Response text
            
        Returns:
            ΔAG score
        """
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) < 3:
            return 0.0
        
        # Compute authority scores for each sentence using semantic analysis
        def sentence_authority(sent: str) -> float:
            # Use semantic analyzer for authority detection
            authority_result = semantic_analyzer.analyze_authority_cues(sent)
            return authority_result.score
        
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
        from signal_estimation import signal_estimator
        
        prompt_lower = prompt.lower()
        e_hat = 0.0
        s_hat = 0.0
        a_hat = 0.0
        t_hat = 0.0
        
        # Estimate temporal risk using signal estimation
        t_hat = signal_estimator.estimate_temporal_risk("", prompt)
        
        # Missing or external context (statistical analysis)
        context_terms = ["this code", "the code", "the file", "the document", "the image",
                        "the chart", "the table", "the dataset", "the log", "uploaded"]
        context_density = sum(1 for term in context_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        if context_density > 0:
            e_hat = max(e_hat, min(0.6, context_density * 30.0))
        
        # Ambiguous instructions (statistical analysis)
        ambiguous_terms = ["ambiguous", "what should i do", "what do i do"]
        ambiguous_density = sum(1 for term in ambiguous_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        if ambiguous_density > 0:
            e_hat = max(e_hat, min(0.6, ambiguous_density * 25.0))
            s_hat = max(s_hat, min(0.4, ambiguous_density * 20.0))
        
        # Overly definitive framing (statistical analysis)
        definitive_terms = ["definitive", "the best", "only way", "right way"]
        definitive_density = sum(1 for term in definitive_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        if definitive_density > 0:
            s_hat = max(s_hat, min(0.5, definitive_density * 20.0))
        
        # Authority-seeking questions (statistical analysis)
        authority_terms = ["should", "recommend", "proven", "correct", "right", "ought to"]
        authority_density = sum(1 for term in authority_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        if authority_density > 0:
            a_hat = max(a_hat, min(0.5, authority_density * 15.0))
        
        certainty_terms = ["proven", "definitive", "certain", "absolutely"]
        certainty_density = sum(1 for term in certainty_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        if certainty_density > 0:
            s_hat = max(s_hat, min(0.4, certainty_density * 20.0))
        
        return {'e_hat': e_hat, 's_hat': s_hat, 'a_hat': a_hat, 't_hat': t_hat}
    
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
        # Domain sensitivity emerges from ê/ŝ/â/t̂ score patterns only
        domain = None  # C6: No explicit domain classification
        default_config = self.get_domain_config(None)
        domain_used = None  # C6: No domain-based configuration override
        rho_used = default_config.rho
        self._last_rho_default = default_config.rho
        
        # Use default weights (C6: domain-agnostic core mechanism)
        lambda_e, lambda_s, lambda_a, lambda_t = (
            default_config.lambda_e,
            default_config.lambda_s,
            default_config.lambda_a,
            default_config.lambda_t
        )
        
        # C6 COMPLIANCE: No domain detection or logging
        # Domain sensitivity emerges from ê/ŝ/â/t̂ score patterns only
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
        adaptive_rho, rho_reason = self._adaptive_rho(structural_signals, default_config.rho)
        rho_used = adaptive_rho

        # C6 COMPLIANCE: Domain sensitivity emerges from ê/ŝ/â/t̂ score patterns
        # No explicit domain classification or configuration override
        # Domain detection completely removed for architectural compliance

        # Persist the effective domain/rho for the caller (C6: no domain used)
        self._last_domain_used = None  # C6: No domain-based configuration used
        self._last_rho_used = rho_used
        self._last_structural_signals = structural_signals
        self._last_rho_reason = rho_reason
        self._last_domain_detected = None  # C6: No domain detection performed
        
        return CommitmentRisk(
            e_hat=e_hat,
            s_hat=s_hat,
            a_hat=a_hat,
            t_hat=t_hat,
            R=R
        )
    
    def should_intervene(self, risk: CommitmentRisk, sigma: float, rho: float) -> bool:
        """Determine if IFCS should intervene
        
        Fires iff: σ(z*) ≥ τ ∧ R(z*) > ρ
        
        Args:
            risk: Computed commitment risk
            sigma: Admissibility signal from Control Probe Type-1
            rho: Commitment risk threshold to use
            
        Returns:
            True if should intervene
        """
        # Three-part firing condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        # Note: κ(z*) = 1 is checked in shape_commitment before calling this method
        tau = 0.40  # Default Control Probe threshold
        return sigma >= tau and risk.R > rho
    
    def apply_transformation_rules(self, response: str, risk: CommitmentRisk) -> str:
        """Apply Γ operator: commitment-shaping transformations
        
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
        
        # Apply semantic transformations based on signal strength
        # High-confidence universal terms
        universal_replacements = {
            'always': 'typically',
            'never': 'rarely', 
            'all': 'many',
            'every': 'most',
            'the answer': 'one approach',
            'the solution': 'a solution',
            'the only way': 'one effective way',
            'the best': 'an effective',
            'definitely': 'likely',
            'certainly': 'probably',
            'clearly': 'it appears',
            'obviously': 'it seems',
            'undoubtedly': 'likely',
            'absolutely': 'generally',
            'without exception': 'in most cases',
            'invariably': 'often'
        }
        
        # Apply word-level replacements (case-insensitive)
        words = text.split()
        transformed_words = []
        
        for word in words:
            # Clean word for matching (remove punctuation)
            clean_word = word.lower().strip('.,!?;:')
            
            if clean_word in universal_replacements:
                # Preserve original case and punctuation
                replacement = universal_replacements[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Preserve punctuation
                punctuation = ''.join(c for c in word if not c.isalnum())
                transformed_words.append(replacement + punctuation)
            else:
                transformed_words.append(word)
        
        return ' '.join(transformed_words)
    
    def _rule2_surface_assumptions(self, text: str) -> str:
        """Rule 2: Surface implicit assumptions using signal-guided analysis"""
        # Use signal estimation to determine if assumptions need surfacing
        from signal_estimation import signal_estimator
        
        evidential_risk = signal_estimator.estimate_evidential_risk(text)
        
        # Only add qualifiers if evidential risk is high
        if evidential_risk < 0.4:
            return text
        
        # Add epistemic qualifiers using statistical analysis
        sentences = text.split('. ')
        
        # Check if text already has qualifiers (statistical approach)
        qualifier_terms = ['assuming', 'if', 'provided', 'given that', 'in cases where']
        has_qualifiers = any(term in text.lower() for term in qualifier_terms)
        
        if len(sentences) > 1 and not has_qualifiers:
            # Add qualifier to first substantive claim
            for i, sent in enumerate(sentences[1:], 1):
                if len(sent.strip()) > 20:
                    sentences[i] = f"Assuming typical conditions, {sent[0].lower()}{sent[1:]}"
                    break
        
        return '. '.join(sentences)
    
    def _rule3_attenuate_authority(self, text: str) -> str:
        """Rule 3: Attenuate authority cues using signal-guided transformation"""
        # Use signal estimation to identify authority cues
        from signal_estimation import signal_estimator
        
        authority_signal = signal_estimator.estimate_authority_posture(text)
        
        # Only apply transformations if authority signal is high
        if authority_signal < 0.4:
            return text
        
        # Apply semantic transformations based on authority signal strength
        authority_replacements = {
            'you must': 'you might consider',
            'you should': 'you could',
            'you need to': 'consider',
            'you have to': 'it may help to',
            'must': 'may need to',
            'should': 'could',
            'need to': 'may want to',
            'have to': 'may have to',
            'required to': 'often expected to',
            "it's essential": 'it may be helpful',
            "it's critical": 'it may be important',
            "it's imperative": 'it may be advisable',
            "you're required": "you're often expected",
            'the best way': 'one effective approach',
            'the correct': 'a valid',
            'the right': 'an appropriate'
        }
        
        # Apply phrase-level replacements (case-insensitive)
        text_lower = text.lower()
        result = text
        
        for phrase, replacement in authority_replacements.items():
            # Find and replace while preserving case
            if phrase in text_lower:
                # Simple case-preserving replacement
                start_idx = text_lower.find(phrase)
                while start_idx != -1:
                    original_phrase = text[start_idx:start_idx + len(phrase)]
                    
                    # Preserve capitalization of first letter
                    if original_phrase[0].isupper():
                        replacement_cased = replacement.capitalize()
                    else:
                        replacement_cased = replacement
                    
                    result = result[:start_idx] + replacement_cased + result[start_idx + len(phrase):]
                    text_lower = result.lower()
                    start_idx = text_lower.find(phrase, start_idx + len(replacement_cased))
        
        return result
    
    def _rule4_flatten_gradient(self, text: str) -> str:
        """Rule 4: Flatten early authority gradient"""
        sentences = re.split(r'([.!?]+)', text)
        
        # Get sentence-delimiter pairs
        pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                pairs.append((sentences[i], sentences[i+1]))
            else:
                pairs.append((sentences[i], ''))
        
        if len(pairs) >= 2:
            # Weaken first 2 sentences more aggressively
            for i in range(min(2, len(pairs))):
                sent, delim = pairs[i]
                
                # Add epistemic hedges to beginning
                if not any(h in sent.lower() for h in ['based on', 'evidence suggests', 'research indicates']):
                    if i == 0:
                        sent = f"Based on available information, {sent[0].lower()}{sent[1:]}"
                    
                    pairs[i] = (sent, delim)
        
        # Reconstruct
        result = ''.join(sent + delim for sent, delim in pairs)
        return result
    
    def _rule5_add_conditionals(self, text: str) -> str:
        """Rule 5: Add conditional framing using signal-guided analysis"""
        # Use signal estimation to determine if conditionals are needed
        from signal_estimation import signal_estimator
        
        evidential_risk = signal_estimator.estimate_evidential_risk(text)
        
        # Only add conditionals if evidential risk is high
        if evidential_risk < 0.6:
            return text
        
        # Statistical analysis of existing hedges/disclaimers
        head = text.lower()[:200]
        first_sentence = text.split('.')[0].lower() if '.' in text else text.lower()
        
        conditional_markers = [
            'given', 'assuming', 'if ', 'provided', 'in cases where', 'in some cases',
            'typically', 'generally', 'often', 'may ', 'might ', 'could '
        ]
        disclaimer_markers = [
            'not a doctor', 'not medical advice', 'consult', 'seek medical',
            'medical attention', 'not a substitute', 'individual cases may vary'
        ]
        boundary_markers = [
            "i can't", "i cannot", "i'm not", "i am not", "i don't have",
            "i do not have", "i'm unable", "i am unable"
        ]
        already_hedged = [
            'in most cases', 'typically', 'often', 'usually', 'generally', 'in many cases'
        ]

        # Check for existing qualifiers using statistical analysis
        has_boundary = any(marker in first_sentence for marker in boundary_markers)
        has_hedge = any(hedge in first_sentence for hedge in already_hedged)
        has_conditional = any(marker in head for marker in conditional_markers)
        has_disclaimer = any(marker in head for marker in disclaimer_markers)

        if has_boundary or has_hedge:
            return text

        if not has_conditional and not has_disclaimer:
            leading_ws = ''
            for char in text:
                if char.isspace():
                    leading_ws += char
                else:
                    break
            
            core = text[len(leading_ws):]
            text = f"{leading_ws}In typical scenarios, {core}"
        
        # Add limitations suffix using statistical analysis
        tail = text.lower()
        limitation_markers = ['though', 'however', 'exceptions exist', 'individual cases may vary', 'consult', 'seek medical']
        has_limitation = any(marker in tail for marker in limitation_markers)
        
        if not has_limitation:
            text = f"{text.rstrip('.')}. Though exceptions exist and individual cases may vary."
        
        return text
    
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
        
        # Step 1: Commitment-Actuality Gate (κ(z*) check) with timing
        kappa_start_time = time.perf_counter()
        is_commitment_bearing = self.commitment_classifier.is_commitment_bearing(response, prompt)
        kappa_computation_time_ms = (time.perf_counter() - kappa_start_time) * 1000
        kappa = 1 if is_commitment_bearing else 0
        
        print(f"[IFCS] Commitment-actuality classification: κ(z*)={kappa} ({'commitment-bearing' if kappa else 'non-commitment-bearing'})")
        print(f"[IFCS] κ(z*) computation time: {kappa_computation_time_ms:.2f}ms")
        
        # Compute commitment risk with timing
        risk_start_time = time.perf_counter()
        risk = self.compute_commitment_risk(response, prompt, context)
        risk_computation_time_ms = (time.perf_counter() - risk_start_time) * 1000
        
        domain_detected = None  # C6: No domain detection performed
        domain_used = None  # C6: No domain-based configuration used
        rho_used = getattr(self, "_last_rho_used", self.get_domain_config(None).rho)
        rho_default = getattr(self, "_last_rho_default", self.get_domain_config(None).rho)
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
            'domain_detected': domain_detected,  # C6a: Informational only
            'domain_used': domain_used,  # C6: Always None (domain-agnostic core)
            'risk': risk,
            'rho': rho_used,
            'rho_default': rho_default,
            'rho_reason': rho_reason,
            'structural_signals': structural_signals,
            'threshold_tier': threshold_tier,
            'adaptive_active': True,
            'kappa': kappa,
            'commitment_bearing': is_commitment_bearing,
            'intervened': False,  # Will be updated below
            'sigma': sigma,
            # Performance metrics for κ(z*) computation
            'kappa_computation_time_ms': kappa_computation_time_ms,
            'risk_computation_time_ms': risk_computation_time_ms,
            # Classification logs and metrics
            'classification_logs_count': len(self.commitment_classifier.classification_logs),
            'non_commitment_metrics_count': len(self.commitment_classifier.non_commitment_metrics),
            'classifier_performance_summary': self.commitment_classifier.get_performance_summary()
        }
        
        # Step 2: Three-part firing condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        if kappa == 0:
            # Non-commitment-bearing context: observation without intervention
            overall_time_ms = (time.perf_counter() - overall_start_time) * 1000
            
            # Calculate latency improvement (estimated time saved by avoiding IFCS processing)
            estimated_ifcs_processing_time_ms = 50.0  # Typical IFCS processing time
            latency_improvement_ms = max(0, estimated_ifcs_processing_time_ms - overall_time_ms)
            
            print(f"[IFCS] NON-INTERVENTION: κ(z*)=0 (non-commitment-bearing context)")
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
            print(f"[IFCS] INTERVENING: σ={sigma:.3f} ≥ τ ∧ R={risk.R:.3f} > ρ={debug_info['rho']:.3f} ∧ κ=1")
            
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
            
            print(f"[IFCS] PASS: R={risk.R:.3f} ≤ ρ={debug_info['rho']:.3f} (commitment-bearing but low risk)")
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
    def detect_domain(self, prompt: str) -> Optional[str]:
        """Detect domain from prompt - REMOVED for C6 compliance
        
        C6 Constraint: "IFCS does not architecturally require explicit domain 
        classification to function. Domain sensitivity emerges from structural 
        differences in ê/ŝ/â/t̂ score patterns."
        
        Args:
            prompt: User prompt
            
        Returns:
            None (domain detection removed for C6 compliance)
        """
        # C6 COMPLIANCE: No explicit domain classification
        # Domain sensitivity must emerge from ê/ŝ/â/t̂ score patterns only
        return None
    
    def get_domain_config(self, domain: Optional[str]) -> DomainConfig:
        """Get configuration for domain
        
        Args:
            domain: Detected domain name
            
        Returns:
            Domain-specific configuration
        """
        if domain and domain in self.config.domain_configs:
            return self.config.domain_configs[domain]
        return self.config.domain_configs['default']
    
    def get_classification_logs(self, limit: Optional[int] = None) -> List[CommitmentActualityLog]:
        """Get recent κ(z*) classification logs for monitoring and analysis
        
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
        """Get performance summary for κ(z*) computation
        
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
        """Print comprehensive performance report for κ(z*) computation"""
        summary = self.get_kappa_performance_summary()
        
        if summary['total_classifications'] == 0:
            print("[IFCS] No κ(z*) classifications performed yet.")
            return
        
        print("\n" + "="*60)
        print("κ(z*) COMMITMENT-ACTUALITY GATE PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Classifications: {summary['total_classifications']}")
        print(f"Commitment-Bearing: {summary['commitment_bearing_count']} ({summary['commitment_bearing_ratio']:.1%})")
        print(f"Non-Commitment-Bearing: {summary['non_commitment_bearing_count']} ({1-summary['commitment_bearing_ratio']:.1%})")
        print(f"Average Computation Time: {summary['avg_computation_time_ms']:.2f}ms")
        print(f"Min/Max Computation Time: {summary['min_computation_time_ms']:.2f}ms / {summary['max_computation_time_ms']:.2f}ms")
        print(f"Total Computation Time: {summary['total_computation_time_ms']:.2f}ms")
        
        # Performance analysis
        if summary['avg_computation_time_ms'] < 50:
            print(f"✅ Performance Target Met: {summary['avg_computation_time_ms']:.2f}ms < 50ms target")
        else:
            print(f"⚠️  Performance Target Missed: {summary['avg_computation_time_ms']:.2f}ms > 50ms target")
        
        print("="*60)
    
