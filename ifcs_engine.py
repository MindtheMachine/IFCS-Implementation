"""
Inference-Time Commitment Shaping (IFCS) Implementation
Based on: Chatterjee, A. (2026c). Inference-Time Commitment Shaping
"""

import re
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from trilogy_config import (
    UNIVERSAL_MARKERS, AUTHORITY_MARKERS, DOMAIN_KEYWORDS,
    DomainConfig
)


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
        
    def compute_evidential_insufficiency(self, response: str, context: str = "") -> float:
        """Compute ê(z*): Evidential insufficiency
        
        Args:
            response: Response text
            context: Available context/grounding
            
        Returns:
            ê score [0, 1]
        """
        # Extract claims (simplified: sentences with assertions)
        sentences = re.split(r'[.!?]+', response)
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not claims:
            return 0.5  # Neutral
        
        # Check grounding against context
        if context:
            # Count overlapping significant words (nouns, verbs)
            context_words = set(re.findall(r'\b[a-z]{4,}\b', context.lower()))
            
            unsupported_claims = 0
            for claim in claims:
                claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim.lower()))
                overlap = len(claim_words & context_words)
                
                # Claim is unsupported if < 30% overlap
                if overlap / max(len(claim_words), 1) < 0.3:
                    unsupported_claims += 1
            
            e_hat = unsupported_claims / len(claims)
        else:
            # No context: check for post-cutoff or unverifiable claims
            post_cutoff_patterns = [
                r'\b202[5-9]\b',  # Years after cutoff
                r'\b203[0-9]\b',
                r'current(?:ly)?',
                r'recent(?:ly)?',
                r'today',
                r'now',
                r'latest',
                r'just (?:announced|released|published)'
            ]
            
            temporal_claims = sum(1 for pattern in post_cutoff_patterns 
                                 for claim in claims 
                                 if re.search(pattern, claim, re.IGNORECASE))
            
            # Paper-aligned ranges: no grounding implies weak/stale evidence (0.7-0.9)
            base_e = 0.8
            e_hat = min(1.0, max(base_e, temporal_claims / len(claims) + 0.3))
        
        return e_hat
    
    def compute_scope_inflation(self, response: str) -> float:
        """Compute ŝ(z*): Scope inflation
        
        Args:
            response: Response text
            
        Returns:
            ŝ score [0, 1]
        """
        # Extract assertions (sentences)
        sentences = re.split(r'[.!?]+', response)
        assertions = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not assertions:
            return 0.0
        
        # Count universal markers
        universal_count = sum(
            1 for assertion in assertions
            for marker in UNIVERSAL_MARKERS
            if re.search(r'\b' + re.escape(marker) + r'\b', assertion, re.IGNORECASE)
        )
        
        s_hat = universal_count / len(assertions)
        return min(1.0, s_hat)
    
    def compute_authority_cues(self, response: str) -> float:
        """Compute â(z*): Authority cues
        
        Args:
            response: Response text
            
        Returns:
            â score [0, 1]
        """
        # Extract sentences
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not valid_sentences:
            return 0.0
        
        # Count authority markers
        authority_count = sum(
            1 for sentence in valid_sentences
            for marker in AUTHORITY_MARKERS
            if re.search(r'\b' + re.escape(marker) + r'\b', sentence, re.IGNORECASE)
        )
        
        a_hat = authority_count / len(valid_sentences)
        return min(1.0, a_hat)
    
    def compute_temporal_risk(self, response: str, prompt: str) -> float:
        """Compute t̂(z*): Temporal risk
        
        Args:
            response: Response text
            prompt: Original prompt
            
        Returns:
            t̂ score [0, 1]
        """
        # Check for temporal sensitivity in prompt and response
        current_state_patterns = [
            r'\bcurrent\b', r'\bnow\b', r'\btoday\b', r'\blatest\b',
            r'\bprice\b', r'\bmarket\b', r'\belection\b', r'\bnews\b'
        ]
        future_year_pattern = r'\b(202[4-9]|203\d)\b'
        
        combined_text = f"{prompt} {response}".lower()
        
        matches = sum(1 for pattern in current_state_patterns 
                     if re.search(pattern, combined_text))
        if re.search(future_year_pattern, combined_text):
            matches = max(matches, 3)
        
        if matches >= 3:
            return 1.0  # Current state query
        elif matches >= 2:
            return 0.7  # Evolving practices
        elif matches >= 1:
            return 0.4  # Slowly evolving
        else:
            return 0.0  # Stable facts
    
    def compute_early_authority_gradient(self, response: str) -> float:
        """Compute ΔAG: Early authority gradient
        
        Args:
            response: Response text
            
        Returns:
            ΔAG score
        """
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) < 3:
            return 0.0
        
        # Compute authority scores for each sentence
        def sentence_authority(sent: str) -> float:
            markers = sum(1 for m in AUTHORITY_MARKERS 
                         if re.search(r'\b' + re.escape(m) + r'\b', sent, re.IGNORECASE))
            certainty = sum(1 for m in ['definitely', 'certainly', 'clearly', 'obviously']
                           if m in sent.lower())
            return min(1.0, (markers + certainty) / 3.0)
        
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
        # Detect domain (lowest-priority; only used as a fallback)
        domain = self.detect_domain(prompt)
        default_config = self.get_domain_config(None)
        domain_used = None
        rho_used = default_config.rho
        
        # Use default weights first (C6: domain sensitivity should emerge from scores)
        lambda_e, lambda_s, lambda_a, lambda_t = (
            default_config.lambda_e,
            default_config.lambda_s,
            default_config.lambda_a,
            default_config.lambda_t
        )
        
        # Compute components
        e_hat = self.compute_evidential_insufficiency(response, context)
        s_hat = self.compute_scope_inflation(response)
        a_hat = self.compute_authority_cues(response)
        t_hat = self.compute_temporal_risk(response, prompt)
        
        # Blend in prompt-driven priors (promotes expected intervention for risky prompts)
        prompt_risk = self._prompt_risk_components(prompt)
        e_hat = max(e_hat, prompt_risk['e_hat'])
        s_hat = max(s_hat, prompt_risk['s_hat'])
        a_hat = max(a_hat, prompt_risk['a_hat'])
        t_hat = max(t_hat, prompt_risk['t_hat'])
        
        # Compute R(z*) using default config first
        R = (lambda_e * e_hat +
             lambda_s * s_hat +
             lambda_a * a_hat +
             lambda_t * t_hat)

        # Domain is a last-resort override if default did not trigger
        if domain:
            domain_config = self.get_domain_config(domain)
            if R < default_config.rho:
                lambda_e, lambda_s, lambda_a, lambda_t = (
                    domain_config.lambda_e,
                    domain_config.lambda_s,
                    domain_config.lambda_a,
                    domain_config.lambda_t
                )
                R = (lambda_e * e_hat +
                     lambda_s * s_hat +
                     lambda_a * a_hat +
                     lambda_t * t_hat)
                domain_used = domain
                rho_used = domain_config.rho
            print(f"[IFCS] Domain detected: {domain.upper()} (ρ={domain_config.rho:.2f})")

        # Persist the effective domain/rho for the caller
        self._last_domain_used = domain_used
        self._last_rho_used = rho_used
        
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
        # Add conditional prefix if not already present
        conditionals = ['given', 'assuming', 'if', 'provided', 'in cases where']
        
        if not any(c in text.lower()[:100] for c in conditionals):
            text = f"In typical scenarios, {text[0].lower()}{text[1:]}"
        
        # Add limitations suffix
        if 'though' not in text.lower() and 'however' not in text.lower():
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
        # Compute commitment risk
        risk = self.compute_commitment_risk(response, prompt, context)
        domain_used = getattr(self, "_last_domain_used", None)
        rho_used = getattr(self, "_last_rho_used", self.get_domain_config(None).rho)
        
        print(f"[IFCS] Commitment risk: {risk}")
        
        # Determine if intervention needed
        should_intervene = self.should_intervene(risk, sigma, rho_used)
        
        debug_info = {
            'domain': domain_used,
            'risk': risk,
            'rho': rho_used,
            'intervened': should_intervene,
            'sigma': sigma
        }
        
        if should_intervene:
            print(f"[IFCS] INTERVENING: R={risk.R:.3f} > ρ={debug_info['rho']:.3f}")
            
            # Apply transformations
            shaped_response = self.apply_transformation_rules(response, risk)
            
            # Recompute risk after shaping
            risk_after = self.compute_commitment_risk(shaped_response, prompt, context)
            reduction = ((risk.R - risk_after.R) / risk.R * 100) if risk.R > 0 else 0
            
            print(f"[IFCS] Risk after shaping: {risk_after}")
            print(f"[IFCS] Commitment reduction: {reduction:.1f}%")
            
            debug_info['risk_after'] = risk_after
            debug_info['reduction_percent'] = reduction
            
            return shaped_response, risk, debug_info
        else:
            print(f"[IFCS] PASS: R={risk.R:.3f} ≤ ρ={debug_info['rho']:.3f}")
            return response, risk, debug_info
    def detect_domain(self, prompt: str) -> Optional[str]:
        """Detect domain from prompt
        
        Args:
            prompt: User prompt
            
        Returns:
            Domain name or None
        """
        prompt_lower = prompt.lower()
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            if matches >= 2:  # Require at least 2 keyword matches
                return domain
        
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
    
