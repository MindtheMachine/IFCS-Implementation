"""
Property-based tests for IFCS Commitment-Actuality Gate
Tests the κ(z*) classification logic for commitment-bearing vs non-commitment-bearing contexts
"""

import pytest
from hypothesis import given, strategies as st, settings
from ifcs_engine import CommitmentActualityClassifier


class TestCommitmentActualityClassifier:
    """Test suite for CommitmentActualityClassifier"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = CommitmentActualityClassifier()
    
    # Property 4: Commitment-bearing classification accuracy
    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=200))
    @settings(max_examples=100)
    def test_property_4_commitment_bearing_classification(self, response_text):
        """
        Property 4: Commitment-bearing classification accuracy
        For any response containing normative/directive language, single-path resolution, 
        action-guiding advice, or general reliance claims, the classifier should correctly 
        identify it as commitment-bearing (κ(z*) = 1)
        **Validates: Requirements 1.4, 2.1, 2.2, 2.3, 2.4**
        """
        # Filter out text that might contain commitment-bearing or descriptive words
        problematic_words = ['should', 'must', 'best', 'clearly', 'definitely', 'certainly', 
                           'include', 'includes', 'examples', 'typically', 'usually', 'often',
                           'common', 'practices', 'options', 'alternatives']
        filtered_text = ' '.join([word for word in response_text.split() 
                                 if word.lower() not in problematic_words])
        
        if len(filtered_text.strip()) < 10:
            filtered_text = "neutral content here"
        
        # Add commitment-bearing markers to the response
        commitment_markers = [
            "You should definitely",
            "You must always", 
            "The best approach is to",
            "I recommend that you",
            "The right way to do this is",
            "You need to follow these steps",
            "The only correct method is",
            "Generally, you must"
        ]
        
        for marker in commitment_markers:
            modified_response = f"{marker} {filtered_text}"
            result = self.classifier.is_commitment_bearing(modified_response, "")
            
            # Property: Responses with commitment-bearing markers should be classified as κ=1
            assert result == True, f"Failed to classify commitment-bearing response with marker '{marker}'"
    
    # Property 5: Non-commitment-bearing classification accuracy  
    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')), min_size=10, max_size=200))
    @settings(max_examples=100)
    def test_property_5_non_commitment_bearing_classification(self, response_text):
        """
        Property 5: Non-commitment-bearing classification accuracy
        For any response that is purely descriptive, enumerative, summarizing, or 
        informational listing without stance-taking, the classifier should correctly 
        identify it as non-commitment-bearing (κ(z*) = 0)
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        # Filter out text that might contain commitment-bearing or descriptive words
        problematic_words = ['should', 'must', 'best', 'clearly', 'definitely', 'certainly',
                           'include', 'includes', 'examples', 'typically', 'usually', 'often',
                           'common', 'practices', 'options', 'alternatives']
        filtered_text = ' '.join([word for word in response_text.split() 
                                 if word.lower() not in problematic_words])
        
        if len(filtered_text.strip()) < 10:
            filtered_text = "various options and methods"
        
        # Add non-commitment-bearing markers to the response
        descriptive_markers = [
            "Common practices include",
            "Options typically are",
            "For reference, here are",
            "Examples include",
            "This usually involves",
            "Background information shows",
            "Types commonly found are",
            "Alternatives often include"
        ]
        
        for marker in descriptive_markers:
            modified_response = f"{marker} {filtered_text}"
            result = self.classifier.is_commitment_bearing(modified_response, "")
            
            # Property: Responses with descriptive markers should be classified as κ=0
            assert result == False, f"Failed to classify non-commitment-bearing response with marker '{marker}'"
    
    def test_specific_commitment_bearing_examples(self):
        """Unit tests for specific commitment-bearing examples"""
        commitment_bearing_examples = [
            ("You should use React for this project because it's the best framework.", ""),
            ("The best approach is to implement X by following these steps.", ""),
            ("I recommend following these steps: 1) A, 2) B, 3) C", ""),
            ("You must configure the system to handle this properly.", ""),
            ("The right way to solve this is definitely through method X.", ""),
            ("Generally, you should always validate input before processing.", "")
        ]
        
        for response, prompt in commitment_bearing_examples:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == True, f"Failed to classify commitment-bearing: '{response}'"
    
    def test_specific_non_commitment_bearing_examples(self):
        """Unit tests for specific non-commitment-bearing examples"""
        non_commitment_bearing_examples = [
            ("Common web development practices include React, Vue, Angular.", "What are the current best practices for web development?"),
            ("Current best practices for web development typically involve modern frameworks.", ""),
            ("Popular frameworks include: React (component-based), Vue (progressive), Angular (full-featured)", ""),
            ("Web development approaches vary depending on requirements.", ""),
            ("For reference, here are some common approaches used in the industry.", ""),
            ("Background information includes various methodologies and tools.", ""),
            ("Examples of frameworks include React, Vue, and Angular.", ""),
            ("Options typically include client-side and server-side rendering.", "")
        ]
        
        for response, prompt in non_commitment_bearing_examples:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == False, f"Failed to classify non-commitment-bearing: '{response}'"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            ("", "", False),  # Empty response
            ("   ", "", False),  # Whitespace only
            ("Short text", "", False),  # Very short response
            ("You might consider trying this approach, though alternatives exist.", "", False),  # Qualified recommendation
            ("The answer is complex and depends on many factors.", "", False),  # Ambiguous resolution
        ]
        
        for response, prompt, expected in edge_cases:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == expected, f"Edge case failed: '{response}' expected {expected}, got {result}"
    
    def test_mixed_signals(self):
        """Test responses with both commitment-bearing and descriptive elements"""
        mixed_examples = [
            # More commitment signals than descriptive
            ("You should definitely use React. Common alternatives include Vue and Angular.", "", True),
            # More descriptive signals than commitment  
            ("Common practices include React, Vue, Angular. You might consider React for this case.", "", False),
            # Equal signals - should default to commitment-bearing for safety
            ("You should use React. Common alternatives include Vue.", "", True),
        ]
        
        for response, prompt, expected in mixed_examples:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == expected, f"Mixed signals test failed: '{response}' expected {expected}, got {result}"


class TestIFCSIntegration:
    """Integration tests for IFCS with commitment-actuality gate"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from trilogy_config import IFCSConfig
        from ifcs_engine import IFCSEngine
        
        config = IFCSConfig()
        self.ifcs_engine = IFCSEngine(config)
    
    @given(
        st.text(min_size=20, max_size=200),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_property_1_three_part_firing_condition(self, response_text, sigma, rho):
        """
        Property 1: Three-part firing condition
        For any response and context, IFCS should fire if and only if all three conditions are met: 
        σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
        **Validates: Requirements 1.1**
        """
        prompt = "Test prompt"
        
        # Test the shape_commitment method
        shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
            response_text, prompt, "", sigma
        )
        
        kappa = debug_info.get('kappa', 0)
        intervened = debug_info.get('intervened', False)
        
        # Property: IFCS should intervene iff σ ≥ τ ∧ R > ρ ∧ κ = 1
        tau = 0.40  # Default Control Probe threshold
        should_intervene_by_logic = (sigma >= tau and risk.R > rho and kappa == 1)
        
        assert intervened == should_intervene_by_logic, (
            f"Three-part firing condition violated: "
            f"σ={sigma:.3f}, τ={tau:.3f}, R={risk.R:.3f}, ρ={rho:.3f}, κ={kappa}, "
            f"expected_intervention={should_intervene_by_logic}, actual_intervention={intervened}"
        )
    
    def test_property_2_non_intervention_on_non_commitment_bearing(self):
        """
        Property 2: Non-intervention on non-commitment-bearing contexts
        For any response classified as non-commitment-bearing (κ(z*) = 0), IFCS should not intervene 
        regardless of commitment risk scores, returning the original response unchanged with intervened=False
        **Validates: Requirements 1.2, 3.2**
        """
        non_commitment_examples = [
            "Common web development practices include React, Vue, Angular.",
            "For reference, here are some popular frameworks used in the industry.",
            "Examples of modern approaches include component-based architectures.",
            "Background information shows various methodologies are available."
        ]
        
        for response in non_commitment_examples:
            # Use high sigma and low rho to ensure R > ρ condition would normally trigger
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, "What are the best practices?", "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 1)
            intervened = debug_info.get('intervened', True)
            
            # Property: Non-commitment-bearing contexts should never intervene
            if kappa == 0:
                assert intervened == False, f"Non-commitment-bearing context intervened: '{response}'"
                assert shaped_response == response, f"Non-commitment-bearing response was modified: '{response}'"
    
    def test_property_3_normal_processing_for_commitment_bearing(self):
        """
        Property 3: Normal processing for commitment-bearing contexts
        For any response classified as commitment-bearing (κ(z*) = 1), IFCS should proceed with normal 
        commitment risk evaluation and shaping according to the original σ(z*) ≥ τ ∧ R(z*) > ρ logic
        **Validates: Requirements 1.3, 3.3**
        """
        commitment_examples = [
            "You should definitely use React for this project.",
            "The best approach is to implement microservices architecture.",
            "I recommend following these specific steps for deployment.",
            "You must configure the system with these exact settings."
        ]
        
        for response in commitment_examples:
            # Test with conditions that should trigger intervention
            shaped_response, risk, debug_info = self.ifcs_engine.shape_commitment(
                response, "What should I do?", "", sigma=1.0
            )
            
            kappa = debug_info.get('kappa', 0)
            intervened = debug_info.get('intervened', False)
            rho_used = debug_info.get('rho', 0.40)
            
            # Property: Commitment-bearing contexts should follow normal IFCS logic
            if kappa == 1:
                expected_intervention = risk.R > rho_used
                assert intervened == expected_intervention, (
                    f"Commitment-bearing context logic failed: '{response}', "
                    f"R={risk.R:.3f}, ρ={rho_used:.3f}, expected={expected_intervention}, actual={intervened}"
                )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])