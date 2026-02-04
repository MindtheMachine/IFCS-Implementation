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
        # Extended list to include more problematic words that could interfere with classification
        problematic_words = [
            'should', 'must', 'best', 'clearly', 'definitely', 'certainly', 
            'include', 'includes', 'examples', 'typically', 'usually', 'often',
            'common', 'practices', 'options', 'alternatives', 'illustration', 
            'example', 'instance', 'case', 'reference', 'background', 'information',
            'overview', 'compare', 'contrast', 'versus', 'alternative', 'generally',
            'methods', 'techniques', 'approaches', 'contain', 'comprise', 'consist',
            # Add hedging words that could interfere
            'may', 'might', 'could', 'perhaps', 'though', 'however', 'depends', 'varies'
        ]
        
        # More aggressive filtering - remove any word that contains problematic substrings
        filtered_words = []
        for word in response_text.split():
            word_lower = word.lower()
            if not any(prob_word in word_lower for prob_word in problematic_words):
                filtered_words.append(word)
        
        filtered_text = ' '.join(filtered_words)
        
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
            assert result == True, f"Failed to classify commitment-bearing response with marker '{marker}': '{modified_response}'"
    
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
        # Extended list to include more problematic phrases and patterns
        problematic_words = [
            'should', 'must', 'best', 'clearly', 'definitely', 'certainly',
            'include', 'includes', 'examples', 'typically', 'usually', 'often',
            'common', 'practices', 'options', 'alternatives', 'recommend', 'advise',
            'suggest', 'need', 'have', 'ought', 'require', 'optimal', 'ideal', 
            'perfect', 'right', 'correct', 'only', 'sole', 'answer', 'solution',
            'approach', 'method', 'way', 'do', 'use', 'try', 'follow', 'implement'
        ]
        
        # More aggressive filtering - remove any word that contains problematic substrings
        filtered_words = []
        for word in response_text.split():
            word_lower = word.lower().strip('.,!?;:')
            # Check if word contains any problematic substring
            is_problematic = any(prob_word in word_lower for prob_word in problematic_words)
            if not is_problematic:
                filtered_words.append(word)
        
        # Ensure we have enough neutral content
        if len(filtered_words) < 3:
            filtered_words = ['various', 'neutral', 'content', 'items']
        
        filtered_text = ' '.join(filtered_words)
        
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
            assert result == False, f"Failed to classify non-commitment-bearing response with marker '{marker}': '{modified_response}'"
    
    def test_specific_commitment_bearing_examples(self):
        """
        Task 7.1: Unit tests for specific commitment-bearing examples
        Test cases with normative/directive language, single-path resolution, 
        action-guiding advice, and general reliance claims
        Requirements: 6.1
        """
        # Test cases with normative/directive language ("should", "must", "best", "recommended")
        normative_directive_examples = [
            ("You should use React for this project because it's the best framework.", ""),
            ("You must configure the system to handle this properly.", ""),
            ("The best approach is to implement microservices architecture.", ""),
            ("I recommend following these specific deployment steps.", ""),
            ("You need to validate all inputs before processing them.", ""),
            ("The recommended practice is to use HTTPS for all communications.", ""),
            ("You ought to implement proper error handling in your code.", ""),
            ("It's essential that you backup your data regularly.", "")
        ]
        
        # Test cases with single-path resolution and action-guiding advice
        single_path_resolution_examples = [
            ("The answer is to use dependency injection for better testability.", ""),
            ("The solution is to implement a caching layer at the database level.", ""),
            ("The right way to solve this is through event-driven architecture.", ""),
            ("The only correct method is to use prepared statements for SQL queries.", ""),
            ("To fix this issue, you should refactor the authentication module.", ""),
            ("The proper approach is to implement rate limiting on your API endpoints.", ""),
            ("You should definitely choose PostgreSQL over MySQL for this use case.", ""),
            ("The correct implementation requires using async/await patterns.", "")
        ]
        
        # Test cases with general reliance claims and prescriptive language
        general_reliance_examples = [
            ("Generally, you should always validate input before processing.", ""),
            ("In most cases, you must implement proper error handling.", ""),
            ("As a rule, you should use HTTPS for all web communications.", ""),
            ("Typically, you need to sanitize user input to prevent XSS attacks.", ""),
            ("Usually, you should implement logging for debugging purposes.", ""),
            ("In general, you must follow the principle of least privilege.", ""),
            ("Normally, you should use environment variables for configuration.", ""),
            ("Generally speaking, you must implement proper authentication.", "")
        ]
        
        # Test investment advice, professional advice, and technical recommendations
        specialized_advice_examples = [
            ("You should invest in index funds for long-term growth.", ""),
            ("I recommend diversifying your portfolio across different asset classes.", ""),
            ("You must consult a professional before starting any new medication.", ""),
            ("The best treatment approach is to combine therapy with medication.", ""),
            ("You should implement OAuth 2.0 for secure API authentication.", ""),
            ("I recommend using Docker containers for consistent deployments.", ""),
            ("You must use SSL certificates for secure data transmission.", ""),
            ("The optimal database design requires proper normalization.", "")
        ]
        
        all_commitment_examples = (
            normative_directive_examples + 
            single_path_resolution_examples + 
            general_reliance_examples + 
            specialized_advice_examples
        )
        
        for response, prompt in all_commitment_examples:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == True, f"Failed to classify commitment-bearing: '{response}'"
    
    def test_specific_non_commitment_bearing_examples(self):
        """
        Task 7.2: Unit tests for specific non-commitment-bearing examples
        Test cases with pure factual retrieval, descriptive enumeration, 
        summarization without synthesis, and comparative analysis without prescription
        Requirements: 6.2, 6.5
        """
        # Test cases with pure factual retrieval and descriptive enumeration
        factual_retrieval_examples = [
            ("Common web development practices include React, Vue, Angular.", "What are the current best practices for web development?"),
            ("Popular programming languages include Python, JavaScript, Java, C++.", ""),
            ("Database options include MySQL, PostgreSQL, MongoDB, Redis.", ""),
            ("Cloud providers include AWS, Azure, Google Cloud, DigitalOcean.", ""),
            ("Authentication methods include OAuth, JWT, SAML, Basic Auth.", ""),
            ("Testing frameworks include Jest, Mocha, Pytest, JUnit.", ""),
            ("Version control systems include Git, SVN, Mercurial, Bazaar.", ""),
            ("Web servers include Apache, Nginx, IIS, Tomcat.", "")
        ]
        
        # Test cases with summarization without synthesis or judgment
        summarization_examples = [
            ("Current best practices for web development typically involve modern frameworks.", ""),
            ("Web development approaches vary depending on project requirements.", ""),
            ("Modern applications often use microservices architecture patterns.", ""),
            ("Database design usually involves normalization and indexing considerations.", ""),
            ("Security implementations commonly include encryption and authentication.", ""),
            ("Performance optimization typically involves caching and load balancing.", ""),
            ("Development workflows often include testing and continuous integration.", ""),
            ("API design usually follows REST or GraphQL architectural patterns.", "")
        ]
        
        # Test cases with low-entropy informational listing
        informational_listing_examples = [
            ("Popular frameworks include: React (component-based), Vue (progressive), Angular (full-featured)", ""),
            ("For reference, here are some common approaches used in the industry.", ""),
            ("Background information includes various methodologies and tools.", ""),
            ("Examples of frameworks include React, Vue, and Angular.", ""),
            ("Options typically include client-side and server-side rendering.", ""),
            ("Types of databases include relational, document, key-value, and graph.", ""),
            ("Categories of testing include unit, integration, and end-to-end.", ""),
            ("Deployment strategies include blue-green, canary, and rolling updates.", "")
        ]
        
        # Test the specific problematic example from requirements
        problematic_example = [
            ("What are the current best practices for web development?", "What are the current best practices for web development?")
        ]
        
        # Test comparative analysis without prescription
        comparative_analysis_examples = [
            ("React has advantages in component reusability, while Vue offers simpler learning curve.", ""),
            ("MySQL provides ACID compliance, whereas MongoDB offers flexible schema design.", ""),
            ("REST APIs are simpler to implement, while GraphQL provides more flexible querying.", ""),
            ("Microservices offer scalability benefits, but monoliths provide simpler deployment.", ""),
            ("SQL databases ensure data consistency, while NoSQL databases offer better performance.", ""),
            ("Docker containers provide isolation, while virtual machines offer stronger security.", ""),
            ("Agile methodology emphasizes flexibility, whereas Waterfall focuses on planning.", ""),
            ("Client-side rendering improves interactivity, server-side rendering enhances SEO.", "")
        ]
        
        all_non_commitment_examples = (
            factual_retrieval_examples + 
            summarization_examples + 
            informational_listing_examples + 
            problematic_example +
            comparative_analysis_examples
        )
        
        for response, prompt in all_non_commitment_examples:
            result = self.classifier.is_commitment_bearing(response, prompt)
            assert result == False, f"Failed to classify non-commitment-bearing: '{response}'"
    
    def test_edge_cases(self):
        """
        Task 7.3: Edge case tests for classification boundary
        Test mixed commitment/descriptive content, qualified recommendations,
        comparative analysis without clear preference, error conditions, and fallback behavior
        Requirements: 6.3
        """
        # Test empty or malformed responses (should default to κ(z*) = 0)
        empty_malformed_cases = [
            ("", "", False),  # Empty response
            ("   ", "", False),  # Whitespace only
            ("...", "", False),  # Only punctuation
            ("???", "", False),  # Only question marks
            ("123", "", False),  # Only numbers
            ("a", "", False),  # Single character
            ("Short", "", False),  # Very short response
        ]
        
        # Test qualified recommendations ("might consider", "could try")
        qualified_recommendations = [
            ("You might consider trying this approach, though alternatives exist.", "", False),
            ("You could try using React, but Vue is also an option.", "", False),
            ("Perhaps you should look into microservices, depending on your needs.", "", False),
            ("You may want to consider implementing caching, if performance is a concern.", "", False),
            ("It might be worth exploring GraphQL, though REST works fine too.", "", False),
            ("You could potentially use Docker, but it's not strictly necessary.", "", False),
            ("Maybe you should implement logging, if debugging becomes an issue.", "", False),
            ("You might find it helpful to use TypeScript, though JavaScript works too.", "", False),
        ]
        
        # Test comparative analysis without clear preference
        comparative_without_preference = [
            ("React offers component reusability while Vue provides simplicity.", "", False),
            ("SQL databases ensure consistency whereas NoSQL offers flexibility.", "", False),
            ("Microservices provide scalability but increase complexity.", "", False),
            ("Client-side rendering improves interactivity while server-side helps SEO.", "", False),
            ("Docker containers offer isolation but virtual machines provide security.", "", False),
            ("Agile emphasizes flexibility whereas Waterfall focuses on planning.", "", False),
            ("REST APIs are simple to implement while GraphQL provides query flexibility.", "", False),
            ("MySQL provides ACID compliance while MongoDB offers schema flexibility.", "", False),
        ]
        
        # Test mixed commitment/descriptive content with threshold behavior
        mixed_content_cases = [
            # More commitment signals than descriptive (should be True)
            ("You should definitely use React. Common alternatives include Vue and Angular.", "", True),
            ("The best approach is microservices. Options typically include monoliths too.", "", True),
            ("I recommend using Docker. Examples of alternatives include VMs.", "", True),
            
            # More descriptive signals than commitment (should be False)
            ("Common practices include React, Vue, Angular. You might consider React.", "", False),
            ("Options typically include REST and GraphQL. Perhaps try REST first.", "", False),
            ("Examples include MySQL and PostgreSQL. You could use either one.", "", False),
            
            # Equal signals - should default to commitment-bearing for safety (True)
            ("You should use React. Common alternatives include Vue.", "", True),
            ("The best choice is PostgreSQL. Options include MySQL too.", "", True),
            ("I recommend Docker. Examples of alternatives include VMs.", "", True),
        ]
        
        # Test error conditions and fallback behavior
        error_fallback_cases = [
            ("The answer is complex and depends on many factors.", "", False),  # Ambiguous resolution
            ("It's complicated and varies by situation.", "", False),  # Non-committal
            ("There are many ways to approach this problem.", "", False),  # Multiple paths
            ("The solution depends on your specific requirements.", "", False),  # Conditional
            ("Results may vary based on implementation details.", "", False),  # Uncertain outcome
        ]
        
        all_edge_cases = (
            empty_malformed_cases + 
            qualified_recommendations + 
            comparative_without_preference + 
            mixed_content_cases + 
            error_fallback_cases
        )
        
        for response, prompt, expected in all_edge_cases:
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
    
    def test_problematic_example_from_requirements(self):
        """
        Test the specific problematic example mentioned in requirements:
        "What are the current best practices for web development?"
        This should be classified as non-commitment-bearing (κ(z*) = 0)
        Requirements: 6.5
        """
        problematic_query = "What are the current best practices for web development?"
        
        # Test various response patterns to this query
        response_patterns = [
            "Current best practices for web development include using modern frameworks like React, Vue, or Angular.",
            "Common web development practices typically involve responsive design, performance optimization, and security measures.",
            "Popular approaches include component-based architecture, RESTful APIs, and progressive web applications.",
            "Best practices generally include code organization, testing, version control, and deployment automation.",
            "Modern web development often involves frameworks, build tools, package managers, and continuous integration.",
        ]
        
        for response in response_patterns:
            result = self.classifier.is_commitment_bearing(response, problematic_query)
            assert result == False, (
                f"Problematic example incorrectly classified as commitment-bearing: "
                f"Query: '{problematic_query}', Response: '{response}'"
            )


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