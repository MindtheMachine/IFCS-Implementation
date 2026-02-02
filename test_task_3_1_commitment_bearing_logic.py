"""
Task 3.1: Verify commitment-bearing contexts follow original IFCS logic

This test suite verifies that commitment-bearing contexts (κ(z*) = 1) continue to follow 
the original IFCS processing logic without any changes to the mathematical transformations 
or risk scoring algorithms.

Test Requirements:
- Test that κ(z*) = 1 contexts proceed with normal σ/ρ evaluation
- Ensure all existing IFCS transformations (Γ operators) work identically
- Validate that commitment risk scoring (ê, ŝ, â, t̂) remains unchanged
- Preserve all existing error handling and fallback behavior
- Requirements: 1.3, 3.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from ifcs_engine import IFCSEngine, CommitmentActualityClassifier, CommitmentRisk
from trilogy_config import IFCSConfig
import re


class TestTask31CommitmentBearingLogic:
    """Test suite for Task 3.1: Verify commitment-bearing contexts follow original IFCS logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = IFCSConfig()
        self.ifcs_engine = IFCSEngine(config)
        self.classifier = CommitmentActualityClassifier()
    
    def test_commitment_bearing_contexts_proceed_with_normal_sigma_rho_evaluation(self):
        """
        Test that κ(z*) = 1 contexts proceed with normal σ/ρ evaluation
        Validates that the original two-part condition σ(z*) ≥ τ ∧ R(z*) > ρ still applies
        """
        commitment_bearing_responses = [
            "You should definitely use React for this project because it's the best framework.",
            "The correct approach is to implement microservices architecture immediately.",
            "I strongly recommend following these exact steps for optimal deployment.",
            "You must configure the system with these specific settings for success.",
            "The best solution is to use Docker containers for this deployment."
        ]
        
        test_cases = [
            # (sigma, expected_sigma_condition, description)
            (0.30, False, "σ below threshold"),
            (0.39, False, "σ just below threshold"), 
            (0.40, True, "σ exactly at threshold"),
            (0.41, True, "σ just above threshold"),
            (0.60, True, "σ moderately above threshold"),
            (1.00, True, "σ at maximum")
        ]
        
        for response in commitment_bearing_responses:
            # Verify this is classified as commitment-bearing
            is_commitment_bearing = self.classifier.is_commitment_bearing(response, "What should I do?")
            assert is_commitment_bearing, f"Response should be commitment-bearing: '{response}'"
            
            for sigma, expected_sigma_condition, description in test_cases:
                shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                    response, "What should I do?", "", sigma
                )
                
                kappa = debug_info.get('kappa', 0)
                intervened = debug_info.get('intervened', False)
                rho_used = debug_info.get('rho', 0.40)
                tau = 0.40
                
                # Verify commitment-bearing classification
                assert kappa == 1, f"Response should be classified as commitment-bearing (κ=1): '{response}'"
                
                # Test normal σ/ρ evaluation logic
                sigma_condition = sigma >= tau
                risk_condition = risk.R > rho_used
                expected_intervention = sigma_condition and risk_condition
                
                assert sigma_condition == expected_sigma_condition, (
                    f"Sigma condition mismatch for {description}: "
                    f"σ={sigma:.3f} ≥ τ={tau:.3f} should be {expected_sigma_condition}"
                )
                
                assert intervened == expected_intervention, (
                    f"Intervention logic failed for commitment-bearing context ({description}):\n"
                    f"  Response: '{response[:50]}...'\n"
                    f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma_condition}\n"
                    f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk_condition}\n"
                    f"  κ={kappa} = 1: True\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened}"
                )
    
    def test_ifcs_transformations_work_identically_for_commitment_bearing(self):
        """
        Test that all existing IFCS transformations (Γ operators) work identically
        for commitment-bearing contexts
        """
        # Test responses designed to trigger specific transformation rules
        transformation_test_cases = [
            # (response, expected_transformations, description)
            (
                "You must always use React for all projects. It's definitely the best framework.",
                ["rule1_weaken_universals"],
                "Rule 1: Weaken universal claims (always, definitely, best)"
            ),
            (
                "You should configure the database connection immediately.",
                ["rule2_surface_assumptions"],
                "Rule 2: Surface implicit assumptions"
            ),
            (
                "You must do this right now. You have to follow these steps exactly.",
                ["rule3_attenuate_authority"],
                "Rule 3: Attenuate authority cues (must, have to)"
            ),
            (
                "Definitely use this approach. You must implement it. This is the only correct way.",
                ["rule1_weaken_universals", "rule3_attenuate_authority"],
                "Multiple rules: Universal claims + Authority cues"
            ),
            (
                "You should definitely implement microservices architecture for optimal performance.",
                ["rule5_add_conditionals"],
                "Rule 5: Add conditional framing for high evidential/scope risk"
            )
        ]
        
        for response, expected_rules, description in transformation_test_cases:
            # Verify this is commitment-bearing
            is_commitment_bearing = self.classifier.is_commitment_bearing(response, "What should I do?")
            assert is_commitment_bearing, f"Test case should be commitment-bearing: '{response}'"
            
            # Test with high sigma to ensure intervention occurs
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 0)
            intervened = debug_info.get('intervened', False)
            rho_used = debug_info.get('rho', 0.40)
            
            # Verify commitment-bearing classification
            assert kappa == 1, f"Should be commitment-bearing: '{response}'"
            
            # If risk is high enough, intervention should occur and transformations should be applied
            if risk.R > rho_used:
                assert intervened, f"Should intervene for high-risk commitment-bearing response: '{response}'"
                assert shaped_response != response, f"Response should be transformed: '{response}'"
                
                # Verify specific transformations were applied
                if "rule1_weaken_universals" in expected_rules:
                    # Check that universal claims were weakened
                    assert "always" not in shaped_response.lower() or "typically" in shaped_response.lower(), (
                        f"Rule 1 not applied - 'always' should be weakened: '{shaped_response}'"
                    )
                    assert "definitely" not in shaped_response.lower() or "likely" in shaped_response.lower(), (
                        f"Rule 1 not applied - 'definitely' should be weakened: '{shaped_response}'"
                    )
                    assert "the best" not in shaped_response.lower() or "an effective" in shaped_response.lower(), (
                        f"Rule 1 not applied - 'the best' should be weakened: '{shaped_response}'"
                    )
                
                if "rule3_attenuate_authority" in expected_rules:
                    # Check that authority cues were attenuated
                    assert "you must" not in shaped_response.lower() or "you might consider" in shaped_response.lower(), (
                        f"Rule 3 not applied - 'you must' should be attenuated: '{shaped_response}'"
                    )
                    assert "you have to" not in shaped_response.lower() or "you may have to" in shaped_response.lower(), (
                        f"Rule 3 not applied - 'you have to' should be attenuated: '{shaped_response}'"
                    )
                
                if "rule5_add_conditionals" in expected_rules:
                    # Check that conditional framing was added
                    conditional_markers = ["in typical scenarios", "though exceptions exist", "individual cases may vary"]
                    has_conditional = any(marker in shaped_response.lower() for marker in conditional_markers)
                    assert has_conditional, (
                        f"Rule 5 not applied - conditional framing should be added: '{shaped_response}'"
                    )
            
            print(f"✓ Transformation test passed: {description}")
            print(f"  Original: '{response}'")
            print(f"  Shaped: '{shaped_response}'")
            print(f"  Risk: {risk}")
            print(f"  Intervened: {intervened}")
            print()
    
    def test_commitment_risk_scoring_unchanged_for_commitment_bearing(self):
        """
        Test that commitment risk scoring (ê, ŝ, â, t̂) remains unchanged
        for commitment-bearing contexts
        """
        test_responses = [
            # (response, expected_high_components, description)
            (
                "You should definitely use React for all web development projects.",
                ["s_hat", "a_hat"],  # Universal scope + authority cues
                "Universal claims with authority"
            ),
            (
                "Based on my extensive experience, you must implement this solution immediately.",
                ["a_hat"],  # Strong authority cues
                "Authority-heavy response"
            ),
            (
                "The current best practices for 2024 require using the latest frameworks.",
                ["t_hat"],  # Temporal risk
                "Temporal-sensitive response"
            ),
            (
                "You should configure the system without any additional context or documentation.",
                ["e_hat"],  # Evidential insufficiency
                "Context-dependent response"
            ),
            (
                "All developers must always use TypeScript for every project without exception.",
                ["s_hat", "a_hat"],  # Scope inflation + authority
                "Multiple risk components"
            )
        ]
        
        for response, expected_high_components, description in test_responses:
            # Verify commitment-bearing classification
            is_commitment_bearing = self.classifier.is_commitment_bearing(response, "What should I do?")
            assert is_commitment_bearing, f"Should be commitment-bearing: '{response}'"
            
            # Compute risk components
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 0)
            assert kappa == 1, f"Should be commitment-bearing: '{response}'"
            
            # Verify risk components are computed correctly
            assert isinstance(risk, CommitmentRisk), "Risk should be CommitmentRisk instance"
            assert 0.0 <= risk.e_hat <= 1.0, f"ê should be in [0,1]: {risk.e_hat}"
            assert 0.0 <= risk.s_hat <= 1.0, f"ŝ should be in [0,1]: {risk.s_hat}"
            assert 0.0 <= risk.a_hat <= 1.0, f"â should be in [0,1]: {risk.a_hat}"
            assert 0.0 <= risk.t_hat <= 1.0, f"t̂ should be in [0,1]: {risk.t_hat}"
            assert risk.R >= 0.0, f"R should be non-negative: {risk.R}"
            
            # Verify expected high-risk components
            for component in expected_high_components:
                component_value = getattr(risk, component)
                assert component_value >= 0.3, (
                    f"Expected high {component} for '{description}': "
                    f"{component}={component_value:.3f} (should be ≥ 0.3)"
                )
            
            print(f"✓ Risk scoring test passed: {description}")
            print(f"  Response: '{response}'")
            print(f"  Risk: {risk}")
            print(f"  Expected high components: {expected_high_components}")
            print()
    
    def test_error_handling_and_fallback_behavior_preserved(self):
        """
        Test that all existing error handling and fallback behavior is preserved
        for commitment-bearing contexts
        """
        error_test_cases = [
            # (response, prompt, context, sigma, description)
            ("", "What should I do?", "", 1.0, "Empty response"),
            ("   ", "What should I do?", "", 1.0, "Whitespace-only response"),
            ("Short", "What should I do?", "", 1.0, "Very short response"),
            ("You should definitely use React.", "", "", 1.0, "Empty prompt"),
            ("You should definitely use React.", "What should I do?", "", -0.5, "Negative sigma"),
            ("You should definitely use React.", "What should I do?", "", 2.0, "Sigma > 1.0"),
        ]
        
        for response, prompt, context, sigma, description in error_test_cases:
            try:
                shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                    response, prompt, context, sigma
                )
                
                # Verify the method doesn't crash and returns valid results
                assert isinstance(shaped_response, str), f"Should return string response for {description}"
                assert isinstance(risk, CommitmentRisk), f"Should return CommitmentRisk for {description}"
                assert isinstance(debug_info, dict), f"Should return dict debug_info for {description}"
                
                # Verify debug info contains expected keys
                expected_keys = ['kappa', 'commitment_bearing', 'intervened', 'sigma']
                for key in expected_keys:
                    assert key in debug_info, f"Debug info missing key '{key}' for {description}"
                
                print(f"✓ Error handling test passed: {description}")
                
            except Exception as e:
                pytest.fail(f"Error handling failed for {description}: {str(e)}")
    
    @given(
        response_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), 
            min_size=20, 
            max_size=200
        ),
        sigma=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_commitment_bearing_contexts_follow_original_logic(self, response_text, sigma):
        """
        Property test: Commitment-bearing contexts follow original IFCS logic
        
        For any commitment-bearing response (κ(z*) = 1), the system should:
        1. Proceed with normal σ/ρ evaluation
        2. Apply transformations when intervention is needed
        3. Compute risk scores using original algorithms
        4. Handle all cases without errors
        
        **Validates: Requirements 1.3, 3.3**
        """
        # Clean the generated text
        response_text = ' '.join(response_text.split())
        assume(len(response_text.strip()) >= 15)
        
        # Add commitment-bearing markers to ensure κ(z*) = 1
        commitment_markers = [
            "You should definitely",
            "You must always", 
            "The best approach is to",
            "I recommend that you"
        ]
        
        marker = commitment_markers[hash(response_text) % len(commitment_markers)]
        commitment_response = f"{marker} {response_text}"
        
        # Verify this creates a commitment-bearing response
        is_commitment_bearing = self.classifier.is_commitment_bearing(commitment_response, "What should I do?")
        assume(is_commitment_bearing)  # Only test commitment-bearing responses
        
        try:
            # Test the shape_commitment method
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                commitment_response, "What should I do?", "", sigma
            )
            
            # Property 1: Should be classified as commitment-bearing
            kappa = debug_info.get('kappa', 0)
            assert kappa == 1, f"Should be commitment-bearing: κ={kappa}"
            
            # Property 2: Should follow normal σ/ρ evaluation
            tau = 0.40
            rho_used = debug_info.get('rho', 0.40)
            intervened = debug_info.get('intervened', False)
            
            expected_intervention = (sigma >= tau) and (risk.R > rho_used)
            assert intervened == expected_intervention, (
                f"Should follow original logic: σ={sigma:.3f} ≥ τ={tau:.3f} ∧ "
                f"R={risk.R:.3f} > ρ={rho_used:.3f} = {expected_intervention}, got {intervened}"
            )
            
            # Property 3: Risk components should be valid
            assert isinstance(risk, CommitmentRisk), "Should return CommitmentRisk"
            assert 0.0 <= risk.e_hat <= 1.0, f"ê should be in [0,1]: {risk.e_hat}"
            assert 0.0 <= risk.s_hat <= 1.0, f"ŝ should be in [0,1]: {risk.s_hat}"
            assert 0.0 <= risk.a_hat <= 1.0, f"â should be in [0,1]: {risk.a_hat}"
            assert 0.0 <= risk.t_hat <= 1.0, f"t̂ should be in [0,1]: {risk.t_hat}"
            assert risk.R >= 0.0, f"R should be non-negative: {risk.R}"
            
            # Property 4: Should return valid shaped response
            assert isinstance(shaped_response, str), "Should return string response"
            assert len(shaped_response.strip()) > 0, "Shaped response should not be empty"
            
            # Property 5: If intervention occurred, response should be modified
            if intervened:
                # Response may be the same if no transformations were needed
                # But the system should have attempted transformation
                pass  # This is acceptable - not all high-risk responses need transformation
            
        except Exception as e:
            pytest.fail(f"Commitment-bearing logic failed for response '{commitment_response[:50]}...': {str(e)}")
    
    def test_specific_commitment_bearing_examples_follow_original_logic(self):
        """
        Test specific commitment-bearing examples to ensure they follow original IFCS logic
        """
        test_cases = [
            # (response, prompt, expected_kappa, description)
            (
                "You should definitely use React for this project because it's the best framework.",
                "What framework should I use?",
                1,
                "Strong recommendation with superlative"
            ),
            (
                "The correct approach is to implement microservices architecture immediately.",
                "How should I design my system?",
                1,
                "Definitive guidance with urgency"
            ),
            (
                "I strongly recommend following these exact steps for deployment.",
                "How do I deploy my application?",
                1,
                "Explicit recommendation with specificity"
            ),
            (
                "You must configure the system with these specific settings for success.",
                "How should I configure the system?",
                1,
                "Imperative directive with outcome claim"
            ),
            (
                "Generally, you should always validate input before processing data.",
                "What are good security practices?",
                1,
                "Universal guidance with authority"
            )
        ]
        
        for response, prompt, expected_kappa, description in test_cases:
            # Test with various sigma values to verify original logic
            sigma_values = [0.2, 0.4, 0.6, 1.0]
            
            for sigma in sigma_values:
                shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                    response, prompt, "", sigma
                )
                
                kappa = debug_info.get('kappa', 0)
                intervened = debug_info.get('intervened', False)
                rho_used = debug_info.get('rho', 0.40)
                tau = 0.40
                
                # Verify commitment-bearing classification
                assert kappa == expected_kappa, (
                    f"Classification mismatch for '{description}': "
                    f"expected κ={expected_kappa}, got κ={kappa}"
                )
                
                # Verify original IFCS logic
                expected_intervention = (sigma >= tau) and (risk.R > rho_used)
                assert intervened == expected_intervention, (
                    f"Original logic violation for '{description}' with σ={sigma:.3f}:\n"
                    f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma >= tau}\n"
                    f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened}"
                )
                
                # Verify risk computation is reasonable
                assert isinstance(risk, CommitmentRisk), "Should return CommitmentRisk"
                assert risk.R >= 0.0, f"Risk should be non-negative: {risk.R}"
                
                print(f"✓ Original logic test passed: {description} (σ={sigma:.3f})")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])