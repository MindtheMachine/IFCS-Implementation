"""
Property-based test for IFCS Non-Intervention Behavior
Tests Property 2: Non-intervention on non-commitment-bearing contexts

Task 2.3: Write property test for non-intervention behavior
- Property 2: Non-intervention on non-commitment-bearing contexts
- Validates: Requirements 1.2, 3.2

Create a property-based test that validates non-commitment-bearing contexts (κ(z*) = 0) 
never trigger IFCS intervention regardless of commitment risk scores, returning the 
original response unchanged with intervened=False.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from ifcs_engine import IFCSEngine, CommitmentActualityClassifier
from trilogy_config import IFCSConfig


class TestProperty2NonIntervention:
    """Property-based test for non-intervention on non-commitment-bearing contexts"""
    
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
        descriptive_marker=st.sampled_from([
            "Common practices include",
            "Options typically are",
            "For reference, here are",
            "Examples include",
            "This usually involves",
            "Background information shows",
            "Types commonly found are",
            "Alternatives often include",
            "Methods generally used are",
            "Approaches commonly seen include",
            "Techniques typically involve",
            "Strategies often employed are"
        ]),
        prompt_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')), 
            min_size=10, 
            max_size=100
        ),
        sigma=st.floats(min_value=0.4, max_value=1.0),  # High sigma to ensure σ ≥ τ condition
        context_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), 
            min_size=0, 
            max_size=50
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_property_2_non_intervention_on_non_commitment_bearing_contexts(
        self, base_text, descriptive_marker, prompt_text, sigma, context_text
    ):
        """
        Property 2: Non-intervention on non-commitment-bearing contexts
        For any response classified as non-commitment-bearing (κ(z*) = 0), IFCS should not intervene 
        regardless of commitment risk scores, returning the original response unchanged with intervened=False
        **Validates: Requirements 1.2, 3.2**
        """
        # Clean and filter the generated text
        base_text = ' '.join(base_text.split())
        prompt_text = ' '.join(prompt_text.split())
        context_text = ' '.join(context_text.split())
        
        # Skip if text is too short after filtering
        assume(len(base_text.strip()) >= 8)
        assume(len(prompt_text.strip()) >= 5)
        
        # Filter out problematic words that could interfere with classification
        problematic_words = [
            'should', 'must', 'best', 'clearly', 'definitely', 'certainly', 
            'recommend', 'advise', 'suggest', 'need', 'have', 'ought', 'require',
            'optimal', 'ideal', 'perfect', 'right', 'correct', 'only', 'sole'
        ]
        
        # Remove problematic words from base text
        filtered_words = []
        for word in base_text.split():
            word_lower = word.lower().strip('.,!?;:')
            if word_lower not in problematic_words:
                filtered_words.append(word)
        
        if len(filtered_words) < 3:
            filtered_words = ['various', 'options', 'and', 'methods']
        
        filtered_base = ' '.join(filtered_words)
        
        # Construct a non-commitment-bearing response using descriptive marker
        response = f"{descriptive_marker} {filtered_base}."
        
        # Verify this response is classified as non-commitment-bearing
        is_commitment_bearing = self.classifier.is_commitment_bearing(response, prompt_text)
        kappa = 1 if is_commitment_bearing else 0
        
        # Only test responses that are actually classified as non-commitment-bearing
        assume(kappa == 0)
        
        # Test the shape_commitment method with high sigma to ensure σ ≥ τ condition
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response, prompt_text, context_text, sigma
        )
        
        # Extract debug information
        actual_kappa = debug_info.get('kappa', 1)
        intervened = debug_info.get('intervened', True)
        actual_sigma = debug_info.get('sigma', 0.0)
        rho_used = debug_info.get('rho', 0.40)
        
        # Verify the response was classified as non-commitment-bearing
        assert actual_kappa == 0, (
            f"Response was not classified as non-commitment-bearing: '{response}' "
            f"(κ={actual_kappa})"
        )
        
        # Property: Non-commitment-bearing contexts should NEVER intervene
        # regardless of commitment risk scores
        assert intervened == False, (
            f"PROPERTY VIOLATION: Non-commitment-bearing context triggered intervention!\n"
            f"  Response: '{response}'\n"
            f"  Prompt: '{prompt_text}'\n"
            f"  κ(z*)={actual_kappa} (should be 0)\n"
            f"  σ(z*)={actual_sigma:.3f} (≥ τ=0.40: {actual_sigma >= 0.40})\n"
            f"  R(z*)={risk.R:.3f} (> ρ={rho_used:.3f}: {risk.R > rho_used})\n"
            f"  Intervened: {intervened} (should be False)\n"
            f"  Intervention reason: {debug_info.get('intervention_reason', 'unknown')}"
        )
        
        # Property: Non-commitment-bearing responses should be returned unchanged
        assert shaped_response == response, (
            f"PROPERTY VIOLATION: Non-commitment-bearing response was modified!\n"
            f"  Original: '{response}'\n"
            f"  Shaped: '{shaped_response}'\n"
            f"  κ(z*)={actual_kappa} (non-commitment-bearing should not be modified)"
        )
        
        # Additional verification: intervention_reason should indicate non-commitment-bearing
        intervention_reason = debug_info.get('intervention_reason', '')
        assert 'non-commitment-bearing' in intervention_reason, (
            f"Intervention reason should indicate non-commitment-bearing: '{intervention_reason}'"
        )
    
    @given(
        commitment_strength=st.sampled_from(['weak_descriptive', 'strong_descriptive']),
        risk_conditions=st.sampled_from(['high_risk', 'low_risk']),
        sigma=st.floats(min_value=0.4, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_non_intervention_with_controlled_risk_conditions(
        self, commitment_strength, risk_conditions, sigma
    ):
        """
        Test non-intervention property with controlled risk conditions to ensure 
        comprehensive coverage of different risk scenarios
        """
        # Create responses with controlled descriptive characteristics
        if commitment_strength == 'weak_descriptive':
            response = "Common approaches include various configuration methods and deployment strategies."
        else:  # strong_descriptive
            response = "For reference, here are examples of popular frameworks: React, Vue, Angular, and others."
        
        # Create prompts that might influence risk levels
        if risk_conditions == 'high_risk':
            prompt = "What should I do right now for my critical production system that's failing?"
        else:  # low_risk
            prompt = "Tell me about general approaches to web development."
        
        # Verify the response is classified as non-commitment-bearing
        is_commitment_bearing = self.classifier.is_commitment_bearing(response, prompt)
        kappa = 1 if is_commitment_bearing else 0
        
        # Only test if actually classified as non-commitment-bearing
        assume(kappa == 0)
        
        # Test with high sigma to ensure σ ≥ τ condition is met
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response, prompt, "", sigma
        )
        
        actual_kappa = debug_info.get('kappa', 1)
        intervened = debug_info.get('intervened', True)
        rho_used = debug_info.get('rho', 0.40)
        
        # Verify classification
        assert actual_kappa == 0, f"Expected non-commitment-bearing classification for: '{response}'"
        
        # Property: Non-intervention regardless of risk conditions
        assert intervened == False, (
            f"Non-commitment-bearing context intervened under {risk_conditions} conditions:\n"
            f"  Response: '{response}'\n"
            f"  Prompt: '{prompt}'\n"
            f"  Risk level: {risk_conditions}\n"
            f"  R(z*)={risk.R:.3f}, ρ={rho_used:.3f}\n"
            f"  σ(z*)={sigma:.3f}\n"
            f"  κ(z*)={actual_kappa}"
        )
        
        # Property: Response unchanged
        assert shaped_response == response, (
            f"Non-commitment-bearing response was modified under {risk_conditions} conditions"
        )
    
    def test_specific_non_commitment_bearing_examples_never_intervene(self):
        """
        Unit test with specific examples to ensure they never trigger intervention
        even under high-risk conditions
        """
        non_commitment_examples = [
            ("Common web development practices include React, Vue, Angular.", 
             "What are the current best practices for web development?"),
            ("For reference, here are some popular frameworks used in the industry.", 
             "What should I use for my critical production system?"),
            ("Examples of modern approaches include component-based architectures.", 
             "Give me the definitive answer for my emergency deployment."),
            ("Background information shows various methodologies are available.", 
             "What's the only correct way to handle this crisis?"),
            ("Types commonly found are client-side and server-side rendering.", 
             "What must I do right now to fix my broken system?"),
            ("Alternatives often include microservices, monoliths, and serverless.", 
             "You need to tell me the best architecture immediately."),
        ]
        
        for response, prompt in non_commitment_examples:
            # Test with maximum sigma to ensure σ ≥ τ condition
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, prompt, "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 1)
            intervened = debug_info.get('intervened', True)
            rho_used = debug_info.get('rho', 0.40)
            
            # Verify classification as non-commitment-bearing
            assert kappa == 0, (
                f"Example should be classified as non-commitment-bearing: '{response}'"
            )
            
            # Property: Never intervene on non-commitment-bearing contexts
            assert intervened == False, (
                f"Non-commitment-bearing example incorrectly triggered intervention:\n"
                f"  Response: '{response}'\n"
                f"  Prompt: '{prompt}'\n"
                f"  R(z*)={risk.R:.3f} > ρ={rho_used:.3f}: {risk.R > rho_used}\n"
                f"  σ(z*)=1.0 ≥ τ=0.40: True\n"
                f"  κ(z*)={kappa} = 1: False\n"
                f"  Should intervene: False, Actually intervened: {intervened}"
            )
            
            # Property: Response unchanged
            assert shaped_response == response, (
                f"Non-commitment-bearing response was incorrectly modified:\n"
                f"  Original: '{response}'\n"
                f"  Shaped: '{shaped_response}'"
            )
    
    def test_edge_cases_non_intervention(self):
        """Test edge cases for non-intervention property"""
        edge_cases = [
            # Very short descriptive responses
            ("Examples include A, B, C.", "What should I choose?", False),
            # Mixed signals but descriptive dominant
            ("Common practices include React. You might consider Vue too.", "What's best?", False),
            # Descriptive with technical terms
            ("Frameworks typically used are React, Angular, Vue, and Svelte.", "What must I use?", False),
            # Informational listing
            ("Options generally available: client-side rendering, server-side rendering.", "What's required?", False),
        ]
        
        for response, prompt, expected_intervention in edge_cases:
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, prompt, "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 1)
            intervened = debug_info.get('intervened', True)
            
            # Only test cases that are actually classified as non-commitment-bearing
            if kappa == 0:
                assert intervened == expected_intervention, (
                    f"Edge case intervention mismatch:\n"
                    f"  Response: '{response}'\n"
                    f"  Expected intervention: {expected_intervention}\n"
                    f"  Actual intervention: {intervened}\n"
                    f"  κ(z*)={kappa}"
                )
                
                if not expected_intervention:
                    assert shaped_response == response, (
                        f"Non-intervening edge case response was modified: '{response}'"
                    )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])