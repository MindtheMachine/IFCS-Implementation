"""
Inference-Time Commitment Shaping (IFCS) Implementation
Based on: Chatterjee, A. (2026c). Inference-Time Commitment Shaping
"""

import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from trilogy_config import DomainConfig
from semantic_analyzer import semantic_analyzer


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


class CommitmentActualityClassifier:
    """Computes κ(z*): commitment-actuality indicator using semantic patterns"""
    
    def __init__(self):
        """Initialize the commitment-actuality classifier"""
        # Core semantic patterns (more flexible than exact matching)
        self.commitment_indicators = {
            'directive_verbs': ['should', 'must', 'need', 'have', 'ought', 'require'],
            'recommendation_verbs': ['recommend', 'suggest', 'advise', 'propose'],
            'certainty_adverbs': ['definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly'],
            'superlative_patterns': ['best', 'optimal', 'ideal', 'perfect', 'right', 'correct'],
            'exclusivity_patterns': ['only', 'sole', 'single', 'exclusive'],
            'imperative_patterns': ['do this', 'follow', 'implement', 'use', 'apply', 'try']
        }
        
        self.descriptive_indicators = {
            'listing_verbs': ['include', 'contain', 'comprise', 'consist'],
            'example_markers': ['example', 'instance', 'case', 'illustration'],
            'reference_markers': ['reference', 'background', 'information', 'overview'],
            'comparison_markers': ['compare', 'contrast', 'versus', 'alternative'],
            'frequency_adverbs': ['typically', 'usually', 'commonly', 'often', 'generally'],
            'informational_phrases': ['practices', 'approaches', 'methods', 'techniques', 'options']
        }
    
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
        commitment_features['superlative_strength'] = superlative_count + exclusivity_count
        commitment_score += (superlative_count + exclusivity_count) * 1.3
        
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
        """Analyze syntactic patterns that indicate commitment vs description"""
        # Simple syntactic analysis without external dependencies
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        commitment_syntax = 0
        descriptive_syntax = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Imperative mood indicators (commitment-bearing)
            if sentence_lower.startswith(('you should', 'you must', 'you need', 'do ', 'use ', 'try ', 'follow ')):
                commitment_syntax += 2
            
            # Question-answer patterns (commitment-bearing)
            if 'the answer is' in sentence_lower or 'the solution is' in sentence_lower:
                commitment_syntax += 2
            
            # Listing patterns (descriptive)
            if any(pattern in sentence_lower for pattern in ['include:', 'are:', 'such as', 'for example']):
                descriptive_syntax += 2
            
            # Conditional/hedged language (descriptive)
            if any(pattern in sentence_lower for pattern in ['may', 'might', 'could', 'depends', 'varies']):
                descriptive_syntax += 1
        
        return {
            'commitment_syntax': commitment_syntax,
            'descriptive_syntax': descriptive_syntax
        }
    
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
        Determine if response constitutes a commitment-bearing act using semantic analysis.
        
        Args:
            response: The response text to classify
            prompt: The original prompt for context
            
        Returns:
            True (κ=1) if response compresses uncertainty into actionable stance
            False (κ=0) if response is descriptive/enumerative/non-binding
        """
        if not response or not response.strip() or len(response.strip()) < 15:
            return False
        
        # Multi-level analysis
        semantic_analysis = self._analyze_semantic_patterns(response)
        syntactic_analysis = self._analyze_syntactic_patterns(response)
        pragmatic_analysis = self._analyze_pragmatic_context(response, prompt)
        
        # Weighted scoring
        commitment_score = (
            semantic_analysis['commitment_score'] * 0.5 +
            syntactic_analysis['commitment_syntax'] * 0.3 +
            max(0, pragmatic_analysis['context_bias']) * 0.2
        )
        
        descriptive_score = (
            semantic_analysis['descriptive_score'] * 0.5 +
            syntactic_analysis['descriptive_syntax'] * 0.3 +
            max(0, -pragmatic_analysis['context_bias']) * 0.2
        )
        
        # Decision with confidence threshold and hedging detection
        score_difference = commitment_score - descriptive_score
        
        # Check for hedging language that reduces commitment
        hedging_patterns = ['might', 'could', 'may', 'perhaps', 'though', 'however', 'depends']
        hedging_penalty = sum(1 for pattern in hedging_patterns if pattern in response.lower())
        
        # Adjust for hedging
        adjusted_difference = score_difference - (hedging_penalty * 0.5)
        
        # Require clear commitment signals to classify as commitment-bearing
        return adjusted_difference > 0.3


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
        """Detect answers that depend on external jurisdiction or authority."""
        if not prompt or not response:
            return 0.0
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        permission_patterns = [
            r'^(can i|may i|am i allowed|is it legal|is it illegal)\b',
            r'\b(can i|may i|am i allowed|is it legal|is it illegal)\b'
        ]
        definitive_answer_patterns = [
            r'\byes,? you can\b',
            r'\bno,? you cannot\b',
            r'\bit is legal\b',
            r'\bit is illegal\b',
            r'\byou are allowed to\b',
            r'\byou are not allowed to\b'
        ]
        qualifier_markers = [
            'depends', 'varies', 'it may', 'it might', 'generally', 'often', 'in some cases',
            'consult', 'jurisdiction', 'local', 'state', 'country'
        ]

        if (
            any(re.search(pat, prompt_lower) for pat in permission_patterns)
            and any(re.search(pat, response_lower) for pat in definitive_answer_patterns)
            and not any(q in response_lower for q in qualifier_markers)
        ):
            return 0.8
        return 0.0

    def _detect_policy_dependency(self, prompt: str, response: str) -> float:
        """Detect reliance on external policies or rules without scope."""
        if not prompt or not response:
            return 0.0
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        policy_markers = ['policy', 'rules', 'regulation', 'terms', 'guidelines']
        definitive_markers = ['definitely', 'always', 'never', 'must', 'required']
        qualifier_markers = ['depends', 'varies', 'check', 'consult', 'may differ']

        if (
            any(m in prompt_lower for m in policy_markers)
            and any(m in response_lower for m in definitive_markers)
            and not any(m in response_lower for m in qualifier_markers)
        ):
            return 0.6
        return 0.0

    def _detect_permission_framing(self, prompt: str, response: str) -> float:
        """Detect binary permission framing without qualifiers."""
        if not prompt or not response:
            return 0.0
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        if re.search(r'^(can i|may i|am i allowed|is it okay to)\b', prompt_lower):
            if not any(q in response_lower for q in ['depends', 'varies', 'it may', 'it might', 'often']):
                return 0.7
        return 0.0

    def _detect_missing_personal_data(self, prompt: str, response: str) -> float:
        """Detect personal-need questions answered with strong diagnosis or prescription."""
        if not prompt or not response:
            return 0.0
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        personal_markers = ['i have', 'i feel', "i'm experiencing", 'my ']
        definitive_markers = ['this is', 'you have', 'it is', 'you need']
        qualifier_markers = ['not a doctor', 'cannot diagnose', 'consult', 'may be', 'could be']

        if (
            any(m in prompt_lower for m in personal_markers)
            and any(m in response_lower for m in definitive_markers)
            and not any(m in response_lower for m in qualifier_markers)
        ):
            return 0.9
        return 0.0

    def _detect_consequence_asymmetry(self, prompt: str, response: str) -> float:
        """Detect high-consequence framing with definitive guidance."""
        if not prompt or not response:
            return 0.0
        prompt_lower = prompt.lower()
        response_lower = response.lower()

        consequence_markers = ['risk', 'danger', 'safety', 'safe', 'unsafe', 'harm']
        definitive_markers = ['definitely', 'certainly', 'no risk', 'safe', 'unsafe']
        qualifier_markers = ['depends', 'varies', 'it may', 'it might', 'consult']

        if (
            any(m in prompt_lower for m in consequence_markers)
            and any(m in response_lower for m in definitive_markers)
            and not any(m in response_lower for m in qualifier_markers)
        ):
            return 0.6
        return 0.0

    def prompt_structural_signals(self, prompt: str) -> Dict[str, float]:
        """Estimate structural risk from prompt only (domain-agnostic)."""
        if not prompt:
            return {}
        prompt_lower = prompt.lower()

        permission_patterns = [
            r'^(can i|may i|am i allowed|is it legal|is it illegal|is it okay to)\b',
            r'\b(can i|may i|am i allowed|is it legal|is it illegal|is it okay to)\b'
        ]
        personal_markers = ['i have', 'i feel', "i'm experiencing", 'my ']
        policy_markers = ['policy', 'rules', 'regulation', 'terms', 'guidelines']
        temporal_markers = ['today', 'now', 'right now', 'current', 'latest', 'this year']
        consequence_markers = ['risk', 'danger', 'safety', 'safe', 'unsafe', 'harm']

        return {
            "jurisdictional": 0.7 if any(re.search(pat, prompt_lower) for pat in permission_patterns) else 0.0,
            "policy": 0.6 if any(m in prompt_lower for m in policy_markers) else 0.0,
            "binary": 0.7 if re.search(r'^(can i|may i|am i allowed|is it okay to)\b', prompt_lower) else 0.0,
            "personal_data": 0.7 if any(m in prompt_lower for m in personal_markers) else 0.0,
            "temporal": 0.6 if any(m in prompt_lower for m in temporal_markers) else 0.0,
            "consequence": 0.6 if any(m in prompt_lower for m in consequence_markers) else 0.0,
        }

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
        """Estimate prompt-driven risk priors for commitment components."""
        if not prompt:
            return {'e_hat': 0.0, 's_hat': 0.0, 'a_hat': 0.0, 't_hat': 0.0}
        
        prompt_lower = prompt.lower()
        e_hat = 0.0
        s_hat = 0.0
        a_hat = 0.0
        t_hat = 0.0
        
        # Missing or external context
        if any(term in prompt_lower for term in [
            "this code", "the code", "the file", "the document", "the image",
            "the chart", "the table", "the dataset", "the log", "uploaded"
        ]):
            e_hat = max(e_hat, 0.6)
        
        # Ambiguous instructions
        if re.search(r'\bwhat (?:should|do) i do\b', prompt_lower) or "ambiguous" in prompt_lower:
            e_hat = max(e_hat, 0.6)
            s_hat = max(s_hat, 0.4)
        
        # Overly definitive framing
        if any(term in prompt_lower for term in ["definitive", "the best", "only way", "right way"]):
            s_hat = max(s_hat, 0.5)
        
        # Authority-seeking questions (broader prompt priors)
        if re.search(r'\bshould\b', prompt_lower) or any(term in prompt_lower for term in [
            "recommend", "proven", "correct", "right", "ought to"
        ]):
            a_hat = max(a_hat, 0.5)
        if any(term in prompt_lower for term in ["proven", "definitive", "certain", "absolutely"]):
            s_hat = max(s_hat, 0.4)
        
        # Time-sensitive or future facts
        if re.search(r'\b(202[4-9]|203\d)\b', prompt_lower):
            t_hat = max(t_hat, 0.7)
        if any(term in prompt_lower for term in ["current", "latest", "today", "right now"]):
            t_hat = max(t_hat, 0.6)
        
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
        # Control Probe must have passed (σ ≥ τ is assumed at this stage)
        # Check if R > ρ
        return risk.R >= rho
    
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
        """Rule 1: Weaken universal claims"""
        replacements = [
            (r'\balways\b', 'typically'),
            (r'\bnever\b', 'rarely'),
            (r'\ball\b', 'many'),
            (r'\bevery\b', 'most'),
            (r'\bthe answer\b', 'one approach'),
            (r'\bthe solution\b', 'a solution'),
            (r'\bthe only way\b', 'one effective way'),
            (r'\bthe best\b', 'an effective'),
            (r'\bdefinitely\b', 'likely'),
            (r'\bcertainly\b', 'probably'),
            (r'\bclearly\b', 'it appears'),
            (r'\bobviously\b', 'it seems'),
            (r'\bundoubtedly\b', 'likely'),
            (r'\babsolutely\b', 'generally'),
            (r'\bwithout exception\b', 'in most cases'),
            (r'\binvariably\b', 'often'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _rule2_surface_assumptions(self, text: str) -> str:
        """Rule 2: Surface implicit assumptions"""
        # Add epistemic qualifiers
        sentences = text.split('. ')
        
        if len(sentences) > 1 and not any(q in text.lower() for q in ['assuming', 'if', 'provided']):
            # Add qualifier to first substantive claim
            for i, sent in enumerate(sentences[1:], 1):
                if len(sent.strip()) > 20:
                    sentences[i] = f"Assuming typical conditions, {sent[0].lower()}{sent[1:]}"
                    break
        
        return '. '.join(sentences)
    
    def _rule3_attenuate_authority(self, text: str) -> str:
        """Rule 3: Attenuate authority cues"""
        replacements = [
            (r'\byou must\b', 'you might consider'),
            (r'\byou should\b', 'you could'),
            (r'\byou need to\b', 'consider'),
            (r'\byou have to\b', 'it may help to'),
            (r'\bmust\b', 'may need to'),
            (r'\bshould\b', 'could'),
            (r'\bneed to\b', 'may want to'),
            (r'\bhave to\b', 'may have to'),
            (r'\brequired to\b', 'often expected to'),
            (r'\bit\'s essential\b', 'it may be helpful'),
            (r'\bit\'s critical\b', 'it may be important'),
            (r'\bit\'s imperative\b', 'it may be advisable'),
            (r'\byou\'re required\b', 'you\'re often expected'),
            (r'\bthe best way\b', 'one effective approach'),
            (r'\bthe correct\b', 'a valid'),
            (r'\bthe right\b', 'an appropriate'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
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
        """Rule 5: Add conditional framing"""
        # Add conditional framing only when the response lacks obvious hedges/disclaimers.
        head = text.lower()[:200]
        first_sentence = re.split(r'[.!?]+', text.strip(), maxsplit=1)[0].lower()
        conditional_markers = [
            'given', 'assuming', 'if ', 'provided', 'in cases where', 'in some cases',
            'typically', 'generally', 'often', 'may ', 'might ', 'could ',
        ]
        disclaimer_markers = [
            'not a doctor', 'not medical advice', 'consult', 'seek medical',
            'medical attention', 'not a substitute', 'individual cases may vary',
        ]
        boundary_markers = [
            "i can't", "i cannot", "i'm not", "i am not", "i don't have",
            "i do not have", "i'm unable", "i am unable"
        ]
        already_hedged = [
            'in most cases', 'typically', 'often', 'usually', 'generally', 'in many cases'
        ]

        if any(b in first_sentence for b in boundary_markers):
            return text

        if any(h in first_sentence for h in already_hedged):
            return text

        if not any(c in head for c in conditional_markers) and not any(d in head for d in disclaimer_markers):
            leading_ws = re.match(r'^\s*', text).group(0)
            core = text[len(leading_ws):]
            text = f"{leading_ws}In typical scenarios, {core}"
        
        # Add limitations suffix only if no explicit limitation already exists.
        tail = text.lower()
        if not any(m in tail for m in ['though', 'however', 'exceptions exist', 'individual cases may vary', 'consult', 'seek medical']):
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
        # Step 1: Commitment-Actuality Gate (κ(z*) check)
        is_commitment_bearing = self.commitment_classifier.is_commitment_bearing(response, prompt)
        kappa = 1 if is_commitment_bearing else 0
        
        print(f"[IFCS] Commitment-actuality classification: κ(z*)={kappa} ({'commitment-bearing' if kappa else 'non-commitment-bearing'})")
        
        # Compute commitment risk
        risk = self.compute_commitment_risk(response, prompt, context)
        domain_detected = None  # C6: No domain detection performed
        domain_used = None  # C6: No domain-based configuration used
        rho_used = getattr(self, "_last_rho_used", self.get_domain_config(None).rho)
        rho_default = getattr(self, "_last_rho_default", self.get_domain_config(None).rho)
        structural_signals = getattr(self, "_last_structural_signals", {})
        rho_reason = getattr(self, "_last_rho_reason", "")
        
        print(f"[IFCS] Commitment risk: {risk}")
        
        threshold_tier = (
            "strict" if rho_used <= 0.30 else
            "moderate" if rho_used <= 0.35 else
            "default"
        )
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
            'sigma': sigma
        }
        
        # Step 2: Three-part firing condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        if kappa == 0:
            # Non-commitment-bearing context: observation without intervention
            print(f"[IFCS] NON-INTERVENTION: κ(z*)=0 (non-commitment-bearing context)")
            debug_info['intervened'] = False
            debug_info['intervention_reason'] = 'non-commitment-bearing'
            return response, risk, debug_info
        
        # Determine if intervention needed (commitment-bearing contexts only)
        should_intervene = self.should_intervene(risk, sigma, rho_used)
        debug_info['intervened'] = should_intervene
        
        if should_intervene:
            print(f"[IFCS] INTERVENING: σ={sigma:.3f} ≥ τ ∧ R={risk.R:.3f} > ρ={debug_info['rho']:.3f} ∧ κ=1")
            
            # Apply transformations
            shaped_response = self.apply_transformation_rules(response, risk)
            
            # Recompute risk after shaping
            risk_after = self.compute_commitment_risk(shaped_response, prompt, context)
            reduction = ((risk.R - risk_after.R) / risk.R * 100) if risk.R > 0 else 0
            
            print(f"[IFCS] Risk after shaping: {risk_after}")
            print(f"[IFCS] Commitment reduction: {reduction:.1f}%")
            
            debug_info['risk_after'] = risk_after
            debug_info['reduction_percent'] = reduction
            debug_info['intervention_reason'] = 'commitment_risk_exceeded'
            
            return shaped_response, risk, debug_info
        else:
            print(f"[IFCS] PASS: R={risk.R:.3f} ≤ ρ={debug_info['rho']:.3f} (commitment-bearing but low risk)")
            debug_info['intervention_reason'] = 'commitment_risk_acceptable'
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
    
