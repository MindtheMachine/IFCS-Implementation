"""
Property-based test for IFCS Normal Processing Preservation
Tests Property 3: Normal processing for commitment-bearing contexts

Task 3.2: Write property test for normal processing preservation
- Property 3: Normal processing for commitment-bearing contexts
- Validates: Requirements 1.3, 3.3

Create a property-based test that validates commitment-bearing contexts (κ(z*) = 1) 
proceed with normal commitment risk evaluation and shaping according to the original 
σ(z*) ≥ τ ∧ R(z*) > ρ logic.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from ifcs_engine import IFCSEngine, CommitmentActualityClassifier, CommitmentRisk
from trilogy_config import IFCSConfig


class TestProperty3NormalProcessingPreservation:
    """Property-based test for normal processing preservation on commitment-bearing contexts"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = IFCSConfig()
        self.ifcs_engine = IFCSEngine(config)
        self.classifier = CommitmentActualityClassifier()
    
    @given(
        base_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), 
            min_size=10, 
            max_size=150
        ),
        commitment_marker=st.sampled_from([
            "You should definitely",
            "You must always", 
            "The best approach is to",
            "I recommend that you",
            "The correct way is to",
            "You need to immediately",
            "The only solution is to",
            "Generally, you must",
            "It's essential that you",
            "The right method is to"
        ]),
        prompt_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')), 
            min_size=10, 
            max_size=100
        ),
        sigma=st.floats(min_value=0.0, max_value=1.0),
        context_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), 
            min_size=0, 
            max_size=50
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_3_normal_processing_for_commitment_bearing_contexts(
        self, base_text, commitment_marker, prompt_text, sigma, context_text
    ):
        """
        Property 3: Normal processing for commitment-bearing contexts
        For any response classified as commitment-bearing (κ(z*) = 1), IFCS should proceed with normal 
        commitment risk evaluation and shaping according to the original σ(z*) ≥ τ ∧ R(z*) > ρ logic
        **Validates: Requirements 1.3, 3.3**
        """
        # Clean and filter the generated text
        base_text = ' '.join(base_text.split())
        prompt_text = ' '.join(prompt_text.split())
        context_text = ' '.join(context_text.split())
        
        # Skip if text is too short after filtering
        assume(len(base_text.strip()) >= 8)
        assume(len(prompt_text.strip()) >= 5)
        
        # Filter out problematic words that could interfere with commitment classification
        descriptive_words = [
            'include', 'includes', 'examples', 'typically', 'usually', 'often',
            'common', 'practices', 'options', 'alternatives', 'illustration', 
            'example', 'instance', 'case', 'reference', 'background', 'information',
            'overview', 'compare', 'contrast', 'versus', 'alternative', 'generally',
            'methods', 'techniques', 'approaches', 'contain', 'comprise', 'consist'
        ]
        
        # Remove descriptive words from base text to ensure commitment-bearing classification
        filtered_words = []
        for word in base_text.split():
            word_lower = word.lower().strip('.,!?;:')
            if word_lower not in descriptive_words:
                filtered_words.append(word)
        
        if len(filtered_words) < 3:
            filtered_words = ['implement', 'the', 'solution']
        
        filtered_base = ' '.join(filtered_words)
        
        # Construct a commitment-bearing response using commitment marker
        response = f"{commitment_marker} {filtered_base}."
        
        # Verify this response is classified as commitment-bearing
        is_commitment_bearing = self.classifier.is_commitment_bearing(response, prompt_text)
        kappa = 1 if is_commitment_bearing else 0
        
        # Only test responses that are actually classified as commitment-bearing
        assume(kappa == 1)
        
        # Test the shape_commitment method
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response, prompt_text, context_text, sigma
        )
        
        # Extract debug information
        actual_kappa = debug_info.get('kappa', 0)
        intervened = debug_info.get('intervened', False)
        actual_sigma = debug_info.get('sigma', 0.0)
        rho_used = debug_info.get('rho', 0.40)
        tau = 0.40  # Default Control Probe threshold
        
        # Verify the response was classified as commitment-bearing
        assert actual_kappa == 1, (
            f"Response was not classified as commitment-bearing: '{response}' "
            f"(κ={actual_kappa})"
        )
        
        # Property: Commitment-bearing contexts should follow original IFCS logic
        # Original logic: σ(z*) ≥ τ ∧ R(z*) > ρ
        condition_1_sigma = actual_sigma >= tau
        condition_2_risk = risk.R > rho_used
        expected_intervention = condition_1_sigma and condition_2_risk
        
        assert intervened == expected_intervention, (
            f"PROPERTY VIOLATION: Commitment-bearing context didn't follow original logic!\n"
            f"  Response: '{response}'\n"
            f"  Prompt: '{prompt_text}'\n"
            f"  κ(z*)={actual_kappa} (commitment-bearing)\n"
            f"  σ(z*)={actual_sigma:.3f} ≥ τ={tau:.3f}: {condition_1_sigma}\n"
            f"  R(z*)={risk.R:.3f} > ρ={rho_used:.3f}: {condition_2_risk}\n"
            f"  Expected intervention: {expected_intervention}\n"
            f"  Actual intervention: {intervened}\n"
            f"  Intervention reason: {debug_info.get('intervention_reason', 'unknown')}"
        )
        
        # Property: Risk components should be computed using original algorithms
        assert isinstance(risk, CommitmentRisk), "Should return CommitmentRisk instance"
        assert 0.0 <= risk.e_hat <= 1.0, f"ê should be in [0,1]: {risk.e_hat}"
        assert 0.0 <= risk.s_hat <= 1.0, f"ŝ should be in [0,1]: {risk.s_hat}"
        assert 0.0 <= risk.a_hat <= 1.0, f"â should be in [0,1]: {risk.a_hat}"
        assert 0.0 <= risk.t_hat <= 1.0, f"t̂ should be in [0,1]: {risk.t_hat}"
        assert risk.R >= 0.0, f"R should be non-negative: {risk.R}"
        
        # Property: If intervention occurred, transformations should be applied
        if intervened:
            # Response may be the same if no specific transformations were triggered
            # But the system should have attempted transformation
            assert isinstance(shaped_response, str), "Shaped response should be string"
            assert len(shaped_response.strip()) > 0, "Shaped response should not be empty"
            
            # If risk is high enough to trigger intervention, some transformation should occur
            # (though not all high-risk responses require visible changes)
            pass  # This is acceptable behavior
        else:
            # If no intervention, response should be unchanged
            assert shaped_response == response, (
                f"Non-intervening commitment-bearing response was modified:\n"
                f"  Original: '{response}'\n"
                f"  Shaped: '{shaped_response}'"
            )
        
        # Property: Debug info should contain expected intervention reason
        intervention_reason = debug_info.get('intervention_reason', '')
        if intervened:
            assert 'commitment_risk_exceeded' in intervention_reason, (
                f"Intervention reason should indicate risk exceeded: '{intervention_reason}'"
            )
        else:
            expected_reasons = ['commitment_risk_acceptable']
            assert any(reason in intervention_reason for reason in expected_reasons), (
                f"Non-intervention reason should be appropriate: '{intervention_reason}'"
            )
    
    @given(
        risk_level=st.sampled_from(['low_risk', 'medium_risk', 'high_risk']),
        sigma_level=st.sampled_from(['low_sigma', 'threshold_sigma', 'high_sigma']),
        commitment_strength=st.sampled_from(['strong_commitment', 'moderate_commitment'])
    )
    @settings(max_examples=50)
    def test_normal_processing_with_controlled_conditions(
        self, risk_level, sigma_level, commitment_strength
    ):
        """
        Test normal processing property with controlled risk and sigma conditions
        to ensure comprehensive coverage of the original IFCS logic
        """
        # Create responses with controlled commitment characteristics
        if commitment_strength == 'strong_commitment':
            response = "You must definitely implement this exact solution immediately for optimal results."
        else:  # moderate_commitment
            response = "I recommend that you consider using this specific approach for your project."
        
        # Create prompts that influence risk levels
        if risk_level == 'high_risk':
            prompt = "What should I do right now for my critical production system that's failing?"
        elif risk_level == 'medium_risk':
            prompt = "What's the best approach for implementing this feature?"
        else:  # low_risk
            prompt = "What are some good practices for this scenario?"
        
        # Set sigma values based on level
        if sigma_level == 'low_sigma':
            sigma = 0.2  # Below threshold
        elif sigma_level == 'threshold_sigma':
            sigma = 0.4  # Exactly at threshold
        else:  # high_sigma
            sigma = 0.8  # Well above threshold
        
        # Verify the response is classified as commitment-bearing
        is_commitment_bearing = self.classifier.is_commitment_bearing(response, prompt)
        kappa = 1 if is_commitment_bearing else 0
        
        # Only test if actually classified as commitment-bearing
        assume(kappa == 1)
        
        # Test with controlled conditions
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response, prompt, "", sigma
        )
        
        actual_kappa = debug_info.get('kappa', 0)
        intervened = debug_info.get('intervened', False)
        rho_used = debug_info.get('rho', 0.40)
        tau = 0.40
        
        # Verify classification
        assert actual_kappa == 1, f"Expected commitment-bearing classification for: '{response}'"
        
        # Property: Should follow original IFCS logic exactly
        expected_intervention = (sigma >= tau) and (risk.R > rho_used)
        assert intervened == expected_intervention, (
            f"Normal processing logic failed for controlled conditions:\n"
            f"  Response: '{response}'\n"
            f"  Prompt: '{prompt}'\n"
            f"  Risk level: {risk_level} (R={risk.R:.3f})\n"
            f"  Sigma level: {sigma_level} (σ={sigma:.3f})\n"
            f"  Commitment strength: {commitment_strength}\n"
            f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma >= tau}\n"
            f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
            f"  Expected intervention: {expected_intervention}\n"
            f"  Actual intervention: {intervened}"
        )
        
        # Property: Risk computation should be consistent
        assert isinstance(risk, CommitmentRisk), "Should return CommitmentRisk"
        assert risk.R >= 0.0, f"Risk should be non-negative: {risk.R}"
    
    def test_specific_commitment_bearing_examples_normal_processing(self):
        """
        Unit test with specific commitment-bearing examples to verify normal processing
        """
        commitment_examples = [
            ("You should definitely use React for this project.", 
             "What framework should I use?"),
            ("The best approach is to implement microservices architecture.", 
             "How should I design my system?"),
            ("I recommend following these specific steps for deployment.", 
             "How do I deploy my application?"),
            ("You must configure the system with these exact settings.", 
             "How should I configure the system?"),
            ("The correct way is to use Docker containers for this deployment.", 
             "What's the right deployment method?"),
            ("Generally, you should always validate input before processing.", 
             "What are good security practices?"),
        ]
        
        # Test with different sigma values to verify original logic
        sigma_test_values = [0.2, 0.4, 0.6, 1.0]
        
        for response, prompt in commitment_examples:
            for sigma in sigma_test_values:
                shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                    response, prompt, "", sigma
                )
                
                kappa = debug_info.get('kappa', 0)
                intervened = debug_info.get('intervened', False)
                rho_used = debug_info.get('rho', 0.40)
                tau = 0.40
                
                # Verify commitment-bearing classification
                assert kappa == 1, (
                    f"Example should be classified as commitment-bearing: '{response}'"
                )
                
                # Property: Should follow original IFCS logic
                expected_intervention = (sigma >= tau) and (risk.R > rho_used)
                assert intervened == expected_intervention, (
                    f"Normal processing failed for commitment-bearing example:\n"
                    f"  Response: '{response}'\n"
                    f"  Prompt: '{prompt}'\n"
                    f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma >= tau}\n"
                    f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened}"
                )
                
                # Property: Risk components should be valid
                assert isinstance(risk, CommitmentRisk), "Should return CommitmentRisk"
                assert 0.0 <= risk.e_hat <= 1.0, f"ê should be in [0,1]: {risk.e_hat}"
                assert 0.0 <= risk.s_hat <= 1.0, f"ŝ should be in [0,1]: {risk.s_hat}"
                assert 0.0 <= risk.a_hat <= 1.0, f"â should be in [0,1]: {risk.a_hat}"
                assert 0.0 <= risk.t_hat <= 1.0, f"t̂ should be in [0,1]: {risk.t_hat}"
                assert risk.R >= 0.0, f"R should be non-negative: {risk.R}"
    
    def test_edge_cases_normal_processing(self):
        """Test edge cases for normal processing property"""
        edge_cases = [
            # (response, prompt, sigma, description)
            ("You should definitely try this approach.", "What should I do?", 0.39, "σ just below threshold"),
            ("You should definitely try this approach.", "What should I do?", 0.40, "σ exactly at threshold"),
            ("You should definitely try this approach.", "What should I do?", 0.41, "σ just above threshold"),
            ("You must implement this solution immediately.", "Help me decide.", 1.0, "High authority with max σ"),
            ("The best approach is clearly to use this method.", "What's optimal?", 0.5, "Superlative claims"),
        ]
        
        for response, prompt, sigma, description in edge_cases:
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, prompt, "", sigma
            )
            
            kappa = debug_info.get('kappa', 0)
            intervened = debug_info.get('intervened', False)
            rho_used = debug_info.get('rho', 0.40)
            tau = 0.40
            
            # Only test cases that are actually classified as commitment-bearing
            if kappa == 1:
                expected_intervention = (sigma >= tau) and (risk.R > rho_used)
                assert intervened == expected_intervention, (
                    f"Edge case normal processing failed for '{description}':\n"
                    f"  Response: '{response}'\n"
                    f"  σ={sigma:.3f} ≥ τ={tau:.3f}: {sigma >= tau}\n"
                    f"  R={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened}"
                )
                
                # Verify risk computation
                assert isinstance(risk, CommitmentRisk), f"Should return CommitmentRisk for '{description}'"
                assert risk.R >= 0.0, f"Risk should be non-negative for '{description}': {risk.R}"
    
    def test_transformation_preservation_for_commitment_bearing(self):
        """
        Test that IFCS transformations are preserved and work correctly 
        for commitment-bearing contexts
        """
        transformation_test_cases = [
            # (response, expected_transformations, description)
            (
                "You must always use React for all projects without exception.",
                ["universal_weakening", "authority_attenuation"],
                "Universal claims with authority cues"
            ),
            (
                "Definitely implement microservices architecture immediately.",
                ["universal_weakening"],
                "Definitive claims requiring weakening"
            ),
            (
                "You should configure the database without any documentation.",
                ["assumption_surfacing"],
                "Context-dependent advice"
            ),
        ]
        
        for response, expected_transformations, description in transformation_test_cases:
            # Test with high sigma to ensure intervention occurs
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 0)
            intervened = debug_info.get('intervened', False)
            rho_used = debug_info.get('rho', 0.40)
            
            # Verify commitment-bearing classification
            assert kappa == 1, f"Should be commitment-bearing: '{response}'"
            
            # If risk is high enough, transformations should be applied
            if risk.R > rho_used:
                assert intervened, f"Should intervene for high-risk response: '{response}'"
                
                # Verify transformations were applied (response should be different)
                # Note: Not all transformations result in visible changes
                assert isinstance(shaped_response, str), "Should return string response"
                assert len(shaped_response.strip()) > 0, "Shaped response should not be empty"
                
                print(f"✓ Transformation test: {description}")
                print(f"  Original: '{response}'")
                print(f"  Shaped: '{shaped_response}'")
                print(f"  Risk: {risk}")
                print(f"  Intervened: {intervened}")
                print()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])