"""
Trilogy Orchestration System
Coordinates ECR → Control Probe Type-1 → IFCS → Control Probe Type-2
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
import json
import re
from datetime import datetime

from ecr_engine import ECREngine
from control_probe import ControlProbeType1, ControlProbeType2, CommitmentDecision
from ifcs_engine import IFCSEngine


@dataclass
class TrilogyResult:
    """Result from trilogy pipeline"""
    final_response: str
    selected_response: str
    shaped_response: str
    ecr_fired: bool
    cp_type1_fired: bool
    cp_type1_decision: str
    ifcs_fired: bool
    cp_type2_fired: bool
    cp_type2_decision: str
    
    # Detailed metrics
    ecr_metrics: Optional[Dict] = None
    cp_type1_metrics: Optional[Dict] = None
    ifcs_metrics: Optional[Dict] = None
    cp_type2_metrics: Optional[Dict] = None
    
    # Processing info
    num_candidates: int = 0
    selected_candidate_idx: int = 0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, indent=2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class TrilogyOrchestrator:
    """Main orchestrator for ECR-Control Probe-IFCS pipeline"""
    
    def __init__(self, config, llm_call_fn, llm_provider=None):
        """Initialize trilogy system
        
        Args:
            config: TrilogyConfig instance
            llm_call_fn: Function to call LLM (prompt, temperature, max_tokens) -> response
            llm_provider: Optional provider instance (for batch capabilities)
        """
        self.config = config
        self.llm_call_fn = llm_call_fn
        self.llm_provider = llm_provider
        
        # Initialize engines
        self.ecr = ECREngine(config.ecr)
        self.cp_type1 = ControlProbeType1(config.control_probe)
        self.ifcs = IFCSEngine(config.ifcs)
        self.cp_type2 = ControlProbeType2(config.control_probe)
        
        print("[Trilogy] Initialized ECR-Control Probe-IFCS system")
    
    def process(self, prompt: str, context: str = "") -> TrilogyResult:
        """Process prompt through full trilogy pipeline
        
        Pipeline: ECR → CP Type-1 → IFCS → [output] → CP Type-2
        
        Args:
            prompt: User prompt
            context: Optional context/grounding
            
        Returns:
            TrilogyResult with final response and metrics
        """
        start_time = datetime.now()
        
        print("\n" + "="*80)
        print(f"[Trilogy] Processing: {prompt[:100]}...")
        print("="*80)

        block, message, decision = self.cp_type2.should_block_prompt(prompt)
        if block:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return TrilogyResult(
                final_response=message,
                selected_response=message,
                shaped_response=message,
                ecr_fired=False,
                cp_type1_fired=False,
                cp_type1_decision="not_evaluated",
                ifcs_fired=False,
                cp_type2_fired=True,
                cp_type2_decision=decision.value if decision else "reset",
                ecr_metrics=None,
                cp_type1_metrics=None,
                ifcs_metrics=None,
                cp_type2_metrics={'reason': 'topic_gate'},
                num_candidates=0,
                selected_candidate_idx=0,
                processing_time_ms=processing_time
            )
        
        # Stage 1: ECR - Generate and select candidates
        print("\n[Stage 1] ECR: Candidate Generation and Selection")
        print("-" * 80)

        candidate_k = self.config.ecr.K
        if self.config.ecr.adaptive_k:
            structural_signals = self.ifcs.prompt_structural_signals(prompt)
            max_signal = max(structural_signals.values()) if structural_signals else 0.0
            if max_signal >= self.config.ecr.adaptive_k_high_threshold:
                candidate_k = self.config.ecr.adaptive_k_high
                reason = f"high structural risk (max={max_signal:.2f})"
            elif max_signal >= self.config.ecr.adaptive_k_mid_threshold:
                candidate_k = self.config.ecr.adaptive_k_mid
                reason = f"moderate structural risk (max={max_signal:.2f})"
            else:
                candidate_k = self.config.ecr.adaptive_k_low
                reason = f"low structural risk (max={max_signal:.2f})"
            candidate_k = min(candidate_k, self.config.ecr.K)
            print(f"[ECR] Adaptive K={candidate_k} based on {reason}")

        candidates = self.ecr.generate_candidates(
            prompt,
            self.llm_call_fn,
            num_candidates=candidate_k,
            llm_provider=self.llm_provider
        )
        selected_response, ecr_metrics, ecr_debug = self.ecr.select_best_candidate(
            candidates, prompt, self.llm_call_fn
        )
        
        ecr_fired = True
        num_candidates = len(candidates)
        selected_idx = ecr_debug['selected_idx']
        
        print(f"\n[ECR] Selected response (candidate {selected_idx + 1}):")
        print(f"{selected_response[:200]}...\n")
        
        # Stage 2: Control Probe Type-1 - Admissibility gating
        print("[Stage 2] Control Probe Type-1: Admissibility Gating")
        print("-" * 80)
        
        cp1_decision, sigma, cp1_debug = self.cp_type1.evaluate(
            selected_response, 
            prompt,
            ecr_metrics
        )
        
        cp1_fired = (cp1_decision == CommitmentDecision.BLOCK)
        
        if cp1_fired:
            # Output blocked - generate appropriate response
            final_response = self.cp_type1.generate_blocked_response(prompt, cp1_debug)
            
            print(f"[CP Type-1] Output BLOCKED (σ={sigma:.3f} < τ={self.config.control_probe.tau:.3f})")
            
            # Pipeline ends here
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TrilogyResult(
                final_response=final_response,
                selected_response=selected_response,
                shaped_response=selected_response,
                ecr_fired=ecr_fired,
                cp_type1_fired=cp1_fired,
                cp_type1_decision=cp1_decision.value,
                ifcs_fired=False,
                cp_type2_fired=False,
                cp_type2_decision="not_evaluated",
                ecr_metrics=vars(ecr_metrics),
                cp_type1_metrics=cp1_debug,
                num_candidates=num_candidates,
                selected_candidate_idx=selected_idx,
                processing_time_ms=processing_time
            )
        
        print(f"[CP Type-1] PASSED (σ={sigma:.3f} ≥ τ={self.config.control_probe.tau:.3f})")
        
        # Stage 3: IFCS - Commitment shaping
        print("\n[Stage 3] IFCS: Commitment Shaping")
        print("-" * 80)
        
        shaped_response, ifcs_risk, ifcs_debug = self.ifcs.shape_commitment(
            selected_response,
            prompt,
            context,
            sigma
        )
        
        ifcs_fired = ifcs_debug['intervened']
        
        if ifcs_fired:
            print(f"\n[IFCS] Shaped response:")
            print(f"{shaped_response[:200]}...\n")
        
        final_response = shaped_response
        
        # Stage 4: Control Probe Type-2 - Interaction-level monitoring
        print("[Stage 4] Control Probe Type-2: Interaction Monitoring")
        print("-" * 80)
        
        # Add current turn to history
        self.cp_type2.add_turn(prompt, final_response, ifcs_risk.R)
        
        # Evaluate interaction
        cp2_decision, cp2_debug = self.cp_type2.evaluate()
        
        cp2_fired = (cp2_decision in [CommitmentDecision.HALT, CommitmentDecision.RESET])
        
        if cp2_fired:
            if cp2_decision == CommitmentDecision.HALT:
                # Append halt message
                halt_message = self.cp_type2.generate_halt_response(cp2_debug)
                final_response = f"{final_response}\n\n{halt_message}"
                
                print(f"[CP Type-2] HALT triggered")
            
            elif cp2_decision == CommitmentDecision.RESET:
                # Reset and note it
                self.cp_type2.reset()
                print(f"[CP Type-2] RESET triggered")
        else:
            # Check if R_cum exists in debug info (might be insufficient history)
            if 'R_cum' in cp2_debug:
                print(f"[CP Type-2] Interaction OK (R_cum={cp2_debug['R_cum']:.3f})")
            else:
                print(f"[CP Type-2] Interaction OK ({cp2_debug.get('reason', 'N/A')})")
        
        # Complete
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print("\n" + "="*80)
        print(f"[Trilogy] Processing complete ({processing_time:.0f}ms)")
        print("="*80 + "\n")
        
        return TrilogyResult(
            final_response=final_response,
            selected_response=selected_response,
            shaped_response=shaped_response,
            ecr_fired=ecr_fired,
            cp_type1_fired=cp1_fired,
            cp_type1_decision=cp1_decision.value,
            ifcs_fired=ifcs_fired,
            cp_type2_fired=cp2_fired,
            cp_type2_decision=cp2_decision.value,
            ecr_metrics=vars(ecr_metrics),
            cp_type1_metrics=cp1_debug,
            ifcs_metrics=ifcs_debug,
            cp_type2_metrics=cp2_debug,
            num_candidates=num_candidates,
            selected_candidate_idx=selected_idx,
            processing_time_ms=processing_time
        )

    def reset_interaction(self):
        """Reset interaction-level history (Control Probe Type-2)."""
        self.cp_type2.reset()


class BaselineAgent:
    """Baseline agent: plain LLM without trilogy"""
    
    def __init__(self, llm_call_fn):
        """Initialize baseline agent
        
        Args:
            llm_call_fn: Function to call LLM
        """
        self.llm_call_fn = llm_call_fn
        
        print("[Baseline] Initialized baseline (unregulated) agent")
    
    def process(self, prompt: str) -> str:
        """Process prompt with plain LLM
        
        Args:
            prompt: User prompt
            
        Returns:
            Raw LLM response
        """
        print(f"\n[Baseline] Processing: {prompt[:100]}...")
        
        response = self.llm_call_fn(prompt, temperature=None)
        
        print(f"[Baseline] Generated response ({len(response)} chars)")
        
        return response


class ComparisonEngine:
    """Compares baseline and regulated outputs"""
    
    @staticmethod
    def compare(
        prompt: str,
        baseline_response: str,
        regulated_result: TrilogyResult
    ) -> Dict:
        """Compare baseline vs regulated responses
        
        Args:
            prompt: Original prompt
            baseline_response: Unregulated response
            regulated_result: Result from trilogy
            
        Returns:
            Comparison analysis dictionary
        """
        print("\n" + "="*80)
        print("[Comparison] Analyzing baseline vs regulated outputs")
        print("="*80)
        
        regulated_response = regulated_result.final_response
        
        # Basic statistics
        comparison = {
            'prompt': prompt,
            'baseline_length': len(baseline_response),
            'regulated_length': len(regulated_response),
            'length_change_pct': ((len(regulated_response) - len(baseline_response)) / 
                                 len(baseline_response) * 100) if baseline_response else 0,
            
            # Mechanism firing
            'mechanisms_fired': {
                'ECR': regulated_result.ecr_fired,
                'CP_Type1': regulated_result.cp_type1_fired,
                'IFCS': regulated_result.ifcs_fired,
                'CP_Type2': regulated_result.cp_type2_fired
            },
            
            # Decisions
            'cp_type1_decision': regulated_result.cp_type1_decision,
            'cp_type2_decision': regulated_result.cp_type2_decision,
            
            # Processing
            'num_candidates_evaluated': regulated_result.num_candidates,
            'processing_time_ms': regulated_result.processing_time_ms,
        }
        
        # Gate metrics for clarity
        comparison['cp_type1_metrics'] = regulated_result.cp_type1_metrics or {}
        
        # Commitment markers analysis
        baseline_markers = ComparisonEngine._count_commitment_markers(baseline_response)
        regulated_markers = ComparisonEngine._count_commitment_markers(regulated_response)
        selected_markers = ComparisonEngine._count_commitment_markers(regulated_result.selected_response)
        shaped_markers = ComparisonEngine._count_commitment_markers(regulated_result.shaped_response)
        
        comparison['commitment_markers'] = {
            'baseline': baseline_markers,
            'regulated': regulated_markers,
            'reduction': {
                'universal': baseline_markers['universal'] - regulated_markers['universal'],
                'authority': baseline_markers['authority'] - regulated_markers['authority'],
                'certainty': baseline_markers['certainty'] - regulated_markers['certainty']
            }
        }
        comparison['ifcs_marker_delta'] = {
            'selected': selected_markers,
            'shaped': shaped_markers,
            'reduction': {
                'universal': selected_markers['universal'] - shaped_markers['universal'],
                'authority': selected_markers['authority'] - shaped_markers['authority'],
                'certainty': selected_markers['certainty'] - shaped_markers['certainty']
            }
        }
        
        # Key changes summary
        changes = []
        
        if regulated_result.cp_type1_fired:
            changes.append("⚠ Control Probe Type-1 blocked the baseline output (inadmissible)")
        
        if regulated_result.ifcs_fired:
            ifcs_metrics = regulated_result.ifcs_metrics
            if ifcs_metrics and 'reduction_percent' in ifcs_metrics:
                reduction = ifcs_metrics['reduction_percent']
                changes.append(f"✓ IFCS reduced commitment risk by {reduction:.1f}%")
            else:
                changes.append("✓ IFCS reshaped commitment strength")
        
        if regulated_result.cp_type2_fired:
            changes.append(f"⚠ Control Probe Type-2 triggered ({regulated_result.cp_type2_decision})")
        
        if regulated_result.ecr_fired and baseline_response != regulated_response:
            changes.append("✓ ECR selected a different candidate than the baseline sample")
        
        if not any([regulated_result.cp_type1_fired, regulated_result.ifcs_fired, regulated_result.cp_type2_fired]):
            if baseline_response == regulated_response:
                changes.append("→ No intervention needed (baseline was appropriate)")
            elif not changes:
                changes.append("→ No additional intervention beyond ECR selection")
        
        comparison['key_changes'] = changes

        # IFCS risk details (paper-aligned components)
        ifcs_metrics = regulated_result.ifcs_metrics or {}
        risk = ifcs_metrics.get('risk')
        risk_after = ifcs_metrics.get('risk_after')
        if risk:
            comparison['ifcs_risk'] = {
                'e_hat': getattr(risk, 'e_hat', None),
                's_hat': getattr(risk, 's_hat', None),
                'a_hat': getattr(risk, 'a_hat', None),
                't_hat': getattr(risk, 't_hat', None),
                'R': getattr(risk, 'R', None),
                'rho': ifcs_metrics.get('rho'),
                'rho_default': ifcs_metrics.get('rho_default'),
                'rho_reason': ifcs_metrics.get('rho_reason'),
                'structural_signals': ifcs_metrics.get('structural_signals'),
                'threshold_tier': ifcs_metrics.get('threshold_tier'),
                'adaptive_active': ifcs_metrics.get('adaptive_active')
            }
        if risk_after:
            comparison['ifcs_risk_after'] = {
                'e_hat': getattr(risk_after, 'e_hat', None),
                's_hat': getattr(risk_after, 's_hat', None),
                'a_hat': getattr(risk_after, 'a_hat', None),
                't_hat': getattr(risk_after, 't_hat', None),
                'R': getattr(risk_after, 'R', None)
            }
        
        # Generate summary
        comparison['summary'] = ComparisonEngine._generate_summary(comparison)
        
        return comparison
    
    @staticmethod
    def _count_commitment_markers(text: str) -> Dict[str, int]:
        """Count commitment markers in text"""
        from trilogy_config import UNIVERSAL_MARKERS, AUTHORITY_MARKERS
        
        text_lower = text.lower()

        def count_markers(markers):
            total = 0
            for marker in markers:
                pattern = r'\\b' + re.escape(marker) + r'\\b'
                total += len(re.findall(pattern, text_lower))
            return total

        universal_count = count_markers(UNIVERSAL_MARKERS)
        authority_count = count_markers(AUTHORITY_MARKERS)

        certainty_markers = ['definitely', 'certainly', 'clearly', 'obviously']
        certainty_count = count_markers(certainty_markers)
        
        return {
            'universal': universal_count,
            'authority': authority_count,
            'certainty': certainty_count,
            'total': universal_count + authority_count + certainty_count
        }
    
    @staticmethod
    def _generate_summary(comparison: Dict) -> str:
        """Generate human-readable summary"""
        lines = []
        
        lines.append("COMPARISON SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Mechanisms
        lines.append("Mechanisms Triggered:")
        for mechanism, fired in comparison['mechanisms_fired'].items():
            status = "✓ FIRED" if fired else "○ Not triggered"
            lines.append(f"  {mechanism:15s} {status}")
        
        lines.append("")
        
        # Key changes
        if comparison['key_changes']:
            lines.append("Key Changes:")
            for change in comparison['key_changes']:
                lines.append(f"  {change}")
            lines.append("")
        
        # Control Probe Type-1 gate details
        cp1 = comparison.get('cp_type1_metrics', {})
        if cp1:
            sigma = cp1.get('sigma')
            sigma_raw = cp1.get('sigma_raw')
            tau = cp1.get('tau')
            prompt_risk = cp1.get('prompt_risk')
            decision = comparison.get('cp_type1_decision', 'N/A')
            
            lines.append("Admissibility Gate (CP Type-1):")
            if sigma is not None and tau is not None:
                lines.append(f"  Decision: {decision.upper()} (σ={sigma:.3f}, τ={tau:.3f})")
            else:
                lines.append(f"  Decision: {decision.upper()}")
            
            if sigma_raw is not None:
                lines.append(f"  σ_raw: {sigma_raw:.3f}")
            if prompt_risk is not None:
                lines.append(f"  Prompt risk: {prompt_risk:.2f}")
            lines.append("")
        
        # Commitment markers
        markers = comparison['commitment_markers']
        lines.append("Commitment Marker Analysis:")
        lines.append(f"  Universal markers:  {markers['baseline']['universal']:3d} → {markers['regulated']['universal']:3d} "
                    f"({markers['reduction']['universal']:+d})")
        lines.append(f"  Authority markers:  {markers['baseline']['authority']:3d} → {markers['regulated']['authority']:3d} "
                    f"({markers['reduction']['authority']:+d})")
        lines.append(f"  Certainty markers:  {markers['baseline']['certainty']:3d} → {markers['regulated']['certainty']:3d} "
                    f"({markers['reduction']['certainty']:+d})")
        
        lines.append("")
        
        # Statistics
        lines.append("Statistics:")
        lines.append(f"  Response length:    {comparison['baseline_length']:4d} → {comparison['regulated_length']:4d} chars "
                    f"({comparison['length_change_pct']:+.1f}%)")
        lines.append(f"  Processing time:    {comparison['processing_time_ms']:.0f}ms")
        lines.append(f"  Candidates tested:  {comparison['num_candidates_evaluated']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_side_by_side(
        prompt: str,
        baseline: str,
        regulated: str,
        comparison: Dict,
        width: int = 80
    ) -> str:
        """Format baseline and regulated outputs side-by-side
        
        Args:
            prompt: Original prompt
            baseline: Baseline response
            regulated: Regulated response
            comparison: Comparison analysis
            width: Column width
            
        Returns:
            Formatted comparison string
        """
        lines = []
        
        # Header
        lines.append("=" * (width * 2 + 3))
        lines.append(f"{'BASELINE (Unregulated)':^{width}} │ {'REGULATED (ECR-CP-IFCS)':^{width}}")
        lines.append("=" * (width * 2 + 3))
        lines.append("")
        
        # Prompt
        lines.append(f"PROMPT: {prompt}")
        lines.append("")
        lines.append("-" * (width * 2 + 3))
        lines.append("")
        
        # Responses (wrap text)
        baseline_lines = ComparisonEngine._wrap_text(baseline, width)
        regulated_lines = ComparisonEngine._wrap_text(regulated, width)
        
        max_lines = max(len(baseline_lines), len(regulated_lines))
        
        for i in range(max_lines):
            left = baseline_lines[i] if i < len(baseline_lines) else ""
            right = regulated_lines[i] if i < len(regulated_lines) else ""
            lines.append(f"{left:<{width}} │ {right:<{width}}")
        
        lines.append("")
        lines.append("=" * (width * 2 + 3))
        lines.append("")
        
        # Add summary
        lines.append(comparison['summary'])
        
        return "\n".join(lines)
    
    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
