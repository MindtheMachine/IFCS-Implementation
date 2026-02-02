"""
Property-based test for IFCS Three-Part Firing Condition
Tests that IFCS fires if and only if all three conditions are met: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from ifcs_engine import IFCSEngine, CommitmentActualityClassifier
from trilogy_config import IFCSConfig


class TestThreePartFiringCondition:
    """Property-based test for the three-part firing condition"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = IFCSConfig()
        self.ifcs_engine = IFCSEngine(config)
        self.classifier = CommitmentActualityClassifier()
    
    @given(
        response_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')), 
            min_size=20, 
            max_size=300
        ),
        prompt_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')), 
            min_size=10, 
            max_size=100
        ),
        sigma=st.floats(min_value=0.0, max_value=1.0),
        rho_override=st.one_of(st.none(), st.floats(min_value=0.1, max_value=0.8))
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_1_three_part_firing_condition(self, response_text, prompt_text, sigma, rho_override):
        """
        Property 1: Three-part firing condition
        For any response and context, IFCS should fire if and only if all three conditions are met: 
        σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        **Validates: Requirements 1.1**
        """
        # Filter out problematic characters and ensure valid text
        response_text = ' '.join(response_text.split())
        prompt_text = ' '.join(prompt_text.split())
        
        # Skip if text is too short after filtering
        assume(len(response_text.strip()) >= 15)
        assume(len(prompt_text.strip()) >= 5)
        
        # Test the shape_commitment method
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response_text, prompt_text, "", sigma
        )
        
        # Extract the three conditions from debug info
        kappa = debug_info.get('kappa', 0)
        intervened = debug_info.get('intervened', False)
        rho_used = rho_override if rho_override is not None else debug_info.get('rho', 0.40)
        tau = 0.40  # Default Control Probe threshold (σ ≥ τ is assumed at IFCS stage)
        
        # The three-part firing condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        condition_1_sigma = sigma >= tau
        condition_2_risk = risk.R > rho_used  
        condition_3_kappa = kappa == 1
        
        should_intervene_by_logic = condition_1_sigma and condition_2_risk and condition_3_kappa
        
        # Property: IFCS should intervene if and only if all three conditions are met
        assert intervened == should_intervene_by_logic, (
            f"Three-part firing condition violated:\n"
            f"  Response: '{response_text[:100]}...'\n"
            f"  Prompt: '{prompt_text}'\n"
            f"  σ(z*)={sigma:.3f} ≥ τ={tau:.3f}: {condition_1_sigma}\n"
            f"  R(z*)={risk.R:.3f} > ρ={rho_used:.3f}: {condition_2_risk}\n"
            f"  κ(z*)={kappa} = 1: {condition_3_kappa}\n"
            f"  Expected intervention: {should_intervene_by_logic}\n"
            f"  Actual intervention: {intervened}\n"
            f"  Debug info: {debug_info.get('intervention_reason', 'unknown')}"
        )
        
        # Additional verification: if any condition fails, intervention should not occur
        if not condition_1_sigma:
            assert not intervened, f"IFCS intervened despite σ={sigma:.3f} < τ={tau:.3f}"
        
        if not condition_2_risk:
            assert not intervened, f"IFCS intervened despite R={risk.R:.3f} ≤ ρ={rho_used:.3f}"
        
        if not condition_3_kappa:
            assert not intervened, f"IFCS intervened despite κ={kappa} ≠ 1 (non-commitment-bearing)"
            # Also verify the response was unchanged for non-commitment-bearing contexts
            assert shaped_response == response_text, (
                f"Non-commitment-bearing response was modified: "
                f"original='{response_text}', shaped='{shaped_response}'"
            )
    
    @given(
        commitment_strength=st.sampled_from(['strong', 'weak', 'mixed']),
        sigma=st.floats(min_value=0.0, max_value=1.0),
        risk_level=st.sampled_from(['low', 'medium', 'high'])
    )
    @settings(max_examples=50)
    def test_three_part_condition_with_controlled_inputs(self, commitment_strength, sigma, risk_level):
        """
        Test three-part firing condition with controlled inputs to ensure comprehensive coverage
        """
        # Create responses with controlled commitment-bearing characteristics
        if commitment_strength == 'strong':
            response = "You must definitely follow these exact steps: 1) Configure the system, 2) Deploy immediately."
            expected_kappa = 1
        elif commitment_strength == 'weak':
            response = "Common approaches include various configuration methods and deployment strategies."
            expected_kappa = 0
        else:  # mixed
            response = "You should definitely consider these specific options for optimal configuration."
            expected_kappa = 1  # Should lean toward commitment-bearing due to "should definitely"
        
        # Create prompts that might influence risk levels
        if risk_level == 'high':
            prompt = "What should I do right now for my critical production system?"
        elif risk_level == 'medium':
            prompt = "What are some good practices for system configuration?"
        else:  # low
            prompt = "Tell me about configuration approaches in general."
        
        # Test the three-part condition
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response, prompt, "", sigma
        )
        
        kappa = debug_info.get('kappa', 0)
        intervened = debug_info.get('intervened', False)
        rho_used = debug_info.get('rho', 0.40)
        tau = 0.40
        
        # Verify kappa matches expected commitment-bearing classification
        assert kappa == expected_kappa, (
            f"Commitment classification mismatch: expected κ={expected_kappa}, got κ={kappa} "
            f"for response: '{response}'"
        )
        
        # Verify three-part condition
        should_intervene = (sigma >= tau) and (risk.R > rho_used) and (kappa == 1)
        assert intervened == should_intervene, (
            f"Three-part condition failed for controlled input:\n"
            f"  Commitment strength: {commitment_strength} (κ={kappa})\n"
            f"  Risk level: {risk_level} (R={risk.R:.3f})\n"
            f"  σ={sigma:.3f}, τ={tau:.3f}, ρ={rho_used:.3f}\n"
            f"  Expected: {should_intervene}, Actual: {intervened}"
        )
    
    def test_boundary_conditions_for_three_part_firing(self):
        """Test boundary conditions for the three-part firing condition"""
        test_cases = [
            # (response, prompt, sigma, expected_kappa, description)
            ("You must do this immediately.", "What should I do?", 0.39, 1, "σ just below τ"),
            ("You must do this immediately.", "What should I do?", 0.40, 1, "σ exactly at τ"),
            ("You must do this immediately.", "What should I do?", 0.41, 1, "σ just above τ"),
            ("Common practices include various methods.", "What are the options?", 1.0, 0, "κ=0 with high σ"),
            ("You should definitely try this approach.", "Help me decide.", 1.0, 1, "κ=1 with high σ"),
        ]
        
        for response, prompt, sigma, expected_kappa, description in test_cases:
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, prompt, "", sigma
            )
            
            kappa = debug_info.get('kappa', 0)
            intervened = debug_info.get('intervened', False)
            rho_used = debug_info.get('rho', 0.40)
            tau = 0.40
            
            # Verify kappa classification
            assert kappa == expected_kappa, (
                f"Boundary test '{description}' failed: expected κ={expected_kappa}, got κ={kappa}"
            )
            
            # Verify three-part condition
            should_intervene = (sigma >= tau) and (risk.R > rho_used) and (kappa == 1)
            assert intervened == should_intervene, (
                f"Boundary test '{description}' three-part condition failed:\n"
                f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma >= tau}\n"
                f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
                f"  κ={kappa} = 1: {kappa == 1}\n"
                f"  Expected: {should_intervene}, Actual: {intervened}"
            )
    
    def test_non_commitment_bearing_never_intervenes(self):
        """Verify that non-commitment-bearing contexts never trigger intervention regardless of other conditions"""
        non_commitment_responses = [
            "Common web development practices include React, Vue, and Angular frameworks.",
            "For reference, here are some popular approaches used in the industry today.",
            "Examples of modern architectures include microservices, monoliths, and serverless.",
            "Background information shows that various methodologies are available for this task."
        ]
        
        # Test with conditions that would normally trigger intervention
        high_risk_prompts = [
            "What should I do right now for my critical system?",
            "Give me the definitive answer for production deployment.",
            "What's the only correct way to handle this emergency?"
        ]
        
        for response in non_commitment_responses:
            for prompt in high_risk_prompts:
                # Use high sigma to ensure σ ≥ τ condition is met
                shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                    response, prompt, "", sigma=1.0
                )
                
                kappa = debug_info.get('kappa', 1)
                intervened = debug_info.get('intervened', True)
                
                # Property: Non-commitment-bearing contexts should never intervene
                if kappa == 0:
                    assert not intervened, (
                        f"Non-commitment-bearing context incorrectly triggered intervention:\n"
                        f"  Response: '{response}'\n"
                        f"  Prompt: '{prompt}'\n"
                        f"  κ={kappa}, R={risk.R:.3f}, intervened={intervened}"
                    )
                    assert shaped_response == response, (
                        f"Non-commitment-bearing response was modified when it shouldn't be"
                    )
    
    def test_commitment_bearing_follows_risk_logic(self):
        """Verify that commitment-bearing contexts follow normal risk evaluation logic"""
        commitment_responses = [
            "You should definitely use React for this project because it's the best choice.",
            "The correct approach is to implement microservices architecture immediately.",
            "I strongly recommend following these exact steps for optimal results.",
            "You must configure the system with these specific settings for success."
        ]
        
        for response in commitment_responses:
            # Test with low sigma (should not intervene due to σ < τ)
            shaped_response_low, risk_low, debug_info_low = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=0.2
            )
            
            kappa_low = debug_info_low.get('kappa', 0)
            intervened_low = debug_info_low.get('intervened', True)
            
            if kappa_low == 1:  # Only test if classified as commitment-bearing
                # Should not intervene due to low sigma
                assert not intervened_low, (
                    f"Commitment-bearing context intervened despite low σ=0.2:\n"
                    f"  Response: '{response}'\n"
                    f"  κ={kappa_low}, R={risk_low.R:.3f}, intervened={intervened_low}"
                )
            
            # Test with high sigma (may intervene based on risk)
            shaped_response_high, risk_high, debug_info_high = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=1.0
            )
            
            kappa_high = debug_info_high.get('kappa', 0)
            intervened_high = debug_info_high.get('intervened', False)
            rho_used = debug_info_high.get('rho', 0.40)
            
            if kappa_high == 1:  # Only test if classified as commitment-bearing
                # Should follow normal risk logic: intervene iff R > ρ
                expected_intervention = risk_high.R > rho_used
                assert intervened_high == expected_intervention, (
                    f"Commitment-bearing context didn't follow risk logic:\n"
                    f"  Response: '{response}'\n"
                    f"  R={risk_high.R:.3f} > ρ={rho_used:.3f}: {expected_intervention}\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened_high}"
                )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])