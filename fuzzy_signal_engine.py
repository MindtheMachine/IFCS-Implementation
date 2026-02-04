"""
Fuzzy Signal Engine for Enhanced Signal Strength Computation
Implements fuzzy logic processing for structural risk signals
"""

from typing import Dict, List, Tuple, Optional
from fuzzy_membership import (
    TriangularMF, TrapezoidalMF, 
    create_low_medium_high_triangular, create_low_medium_high_trapezoidal
)
import json
import os


class FuzzySignalEngine:
    """Fuzzy logic engine for signal strength computation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize fuzzy signal engine
        
        Args:
            config_path: Optional path to configuration file for membership functions
        """
        self.membership_functions = {}
        self.fuzzy_rules = {}
        self.signal_weights = {}
        
        # Load configuration or use defaults
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._initialize_default_membership_functions()
            self._initialize_default_fuzzy_rules()
            self._initialize_default_weights()
    
    def _initialize_default_membership_functions(self):
        """Initialize default triangular membership functions for all signal types"""
        
        # Jurisdictional risk (permission-seeking language)
        self.membership_functions['jurisdictional'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.3),
            'MEDIUM': TriangularMF(0.2, 0.5, 0.8),
            'HIGH': TriangularMF(0.7, 1.0, 1.0)
        }
        
        # Policy risk (policy-related language)
        self.membership_functions['policy'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.4),
            'MEDIUM': TriangularMF(0.3, 0.6, 0.9),
            'HIGH': TriangularMF(0.8, 1.0, 1.0)
        }
        
        # Binary framing risk
        self.membership_functions['binary'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.3),
            'MEDIUM': TriangularMF(0.2, 0.5, 0.8),
            'HIGH': TriangularMF(0.7, 1.0, 1.0)
        }
        
        # Personal data risk
        self.membership_functions['personal_data'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.4),
            'MEDIUM': TriangularMF(0.3, 0.6, 0.9),
            'HIGH': TriangularMF(0.8, 1.0, 1.0)
        }
        
        # Consequence risk
        self.membership_functions['consequence'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.4),
            'MEDIUM': TriangularMF(0.3, 0.6, 0.9),
            'HIGH': TriangularMF(0.8, 1.0, 1.0)
        }
        
        # Temporal risk (already handled by signal estimator, but included for completeness)
        self.membership_functions['temporal'] = {
            'LOW': TriangularMF(0.0, 0.0, 0.3),
            'MEDIUM': TriangularMF(0.2, 0.5, 0.8),
            'HIGH': TriangularMF(0.7, 1.0, 1.0)
        }
    
    def _initialize_default_fuzzy_rules(self):
        """Initialize default fuzzy rules for signal aggregation"""
        
        # Define fuzzy rule weights for each signal type and level
        # These determine how much each fuzzy level contributes to final signal
        self.fuzzy_rules = {
            'jurisdictional': {
                'LOW': 0.1,      # Low jurisdictional risk contributes 0.1
                'MEDIUM': 0.4,   # Medium jurisdictional risk contributes 0.4
                'HIGH': 0.8      # High jurisdictional risk contributes 0.8
            },
            'policy': {
                'LOW': 0.1,
                'MEDIUM': 0.35,
                'HIGH': 0.7
            },
            'binary': {
                'LOW': 0.05,
                'MEDIUM': 0.3,
                'HIGH': 0.6
            },
            'personal_data': {
                'LOW': 0.1,
                'MEDIUM': 0.4,
                'HIGH': 0.8
            },
            'consequence': {
                'LOW': 0.1,
                'MEDIUM': 0.4,
                'HIGH': 0.75
            },
            'temporal': {
                'LOW': 0.05,
                'MEDIUM': 0.25,
                'HIGH': 0.5
            }
        }
    
    def _initialize_default_weights(self):
        """Initialize default weights for combining different signal types"""
        
        # Weights for combining different signal types into final structural risk
        self.signal_weights = {
            'jurisdictional': 0.25,   # Permission-seeking is important
            'policy': 0.20,           # Policy questions are significant
            'binary': 0.15,           # Binary framing is moderate risk
            'personal_data': 0.20,    # Personal data is important
            'consequence': 0.15,      # Consequence focus is moderate
            'temporal': 0.05          # Temporal risk is lower weight (handled elsewhere)
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.signal_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            for signal_type in self.signal_weights:
                self.signal_weights[signal_type] /= total_weight
    
    def compute_memberships(self, intent_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compute fuzzy membership values for each signal type
        
        Args:
            intent_scores: Dictionary mapping signal types to raw scores [0,1]
            
        Returns:
            Dictionary mapping signal types to membership levels (LOW/MEDIUM/HIGH)
        """
        memberships = {}
        
        for signal_type, score in intent_scores.items():
            if signal_type in self.membership_functions:
                # Compute membership for each level
                signal_memberships = {}
                for level, mf in self.membership_functions[signal_type].items():
                    signal_memberships[level] = mf.membership(score)
                
                memberships[signal_type] = signal_memberships
        
        return memberships
    
    def aggregate_signals(self, memberships: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate fuzzy memberships using weighted rules
        
        Args:
            memberships: Dictionary of membership values for each signal type and level
            
        Returns:
            Dictionary mapping signal types to aggregated signal strengths [0,1]
        """
        aggregated = {}
        
        for signal_type, levels in memberships.items():
            if signal_type in self.fuzzy_rules:
                # Weighted aggregation using fuzzy rules
                weighted_sum = 0.0
                total_membership = 0.0
                
                for level, membership_val in levels.items():
                    if level in self.fuzzy_rules[signal_type]:
                        rule_weight = self.fuzzy_rules[signal_type][level]
                        weighted_sum += membership_val * rule_weight
                        total_membership += membership_val
                
                # Normalize by total membership to get final signal strength
                if total_membership > 0:
                    aggregated[signal_type] = min(1.0, weighted_sum / total_membership)
                else:
                    aggregated[signal_type] = 0.0
            else:
                # Fallback: simple weighted average if no rules defined
                aggregated[signal_type] = sum(levels.values()) / len(levels) if levels else 0.0
        
        return aggregated
    
    def compute_structural_risk(self, signal_strengths: Dict[str, float]) -> float:
        """Compute overall structural risk from individual signal strengths
        
        Args:
            signal_strengths: Dictionary mapping signal types to strength values [0,1]
            
        Returns:
            Overall structural risk score [0,1]
        """
        if not signal_strengths:
            return 0.0
        
        # Weighted combination of signal strengths
        weighted_risk = 0.0
        total_weight = 0.0
        
        for signal_type, strength in signal_strengths.items():
            if signal_type in self.signal_weights:
                weight = self.signal_weights[signal_type]
                weighted_risk += strength * weight
                total_weight += weight
        
        # Normalize by total weight used
        if total_weight > 0:
            return min(1.0, weighted_risk / total_weight)
        else:
            # Fallback: simple average
            return sum(signal_strengths.values()) / len(signal_strengths)
    
    def process_signals(self, intent_scores: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """Complete signal processing pipeline
        
        Args:
            intent_scores: Raw intent scores from semantic analysis
            
        Returns:
            Tuple of (individual_signal_strengths, overall_structural_risk)
        """
        # Step 1: Compute fuzzy memberships
        memberships = self.compute_memberships(intent_scores)
        
        # Step 2: Aggregate using fuzzy rules
        signal_strengths = self.aggregate_signals(memberships)
        
        # Step 3: Compute overall structural risk
        structural_risk = self.compute_structural_risk(signal_strengths)
        
        return signal_strengths, structural_risk
    
    def get_signal_details(self, intent_scores: Dict[str, float]) -> Dict:
        """Get detailed breakdown of signal processing for debugging
        
        Args:
            intent_scores: Raw intent scores from semantic analysis
            
        Returns:
            Dictionary with detailed processing information
        """
        memberships = self.compute_memberships(intent_scores)
        signal_strengths = self.aggregate_signals(memberships)
        structural_risk = self.compute_structural_risk(signal_strengths)
        
        return {
            'intent_scores': intent_scores,
            'fuzzy_memberships': memberships,
            'signal_strengths': signal_strengths,
            'structural_risk': structural_risk,
            'weights_used': self.signal_weights
        }
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load membership functions (simplified - would need full implementation)
            if 'membership_functions' in config:
                # This would require serialization/deserialization of MF objects
                pass
            
            # Load fuzzy rules
            if 'fuzzy_rules' in config:
                self.fuzzy_rules = config['fuzzy_rules']
            
            # Load signal weights
            if 'signal_weights' in config:
                self.signal_weights = config['signal_weights']
                # Normalize weights
                total_weight = sum(self.signal_weights.values())
                if total_weight > 0:
                    for signal_type in self.signal_weights:
                        self.signal_weights[signal_type] /= total_weight
        
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            # Fall back to defaults
            self._initialize_default_membership_functions()
            self._initialize_default_fuzzy_rules()
            self._initialize_default_weights()
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file
        
        Args:
            config_path: Path to save configuration file
        """
        config = {
            'fuzzy_rules': self.fuzzy_rules,
            'signal_weights': self.signal_weights,
            # Note: membership_functions would need custom serialization
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config to {config_path}: {e}")
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check that all signal types have membership functions
        required_signals = ['jurisdictional', 'policy', 'binary', 'personal_data', 'consequence', 'temporal']
        for signal_type in required_signals:
            if signal_type not in self.membership_functions:
                errors.append(f"Missing membership functions for signal type: {signal_type}")
            else:
                # Check that each signal type has LOW/MEDIUM/HIGH levels
                required_levels = ['LOW', 'MEDIUM', 'HIGH']
                for level in required_levels:
                    if level not in self.membership_functions[signal_type]:
                        errors.append(f"Missing {level} membership function for {signal_type}")
        
        # Check that fuzzy rules exist for all signal types
        for signal_type in required_signals:
            if signal_type not in self.fuzzy_rules:
                errors.append(f"Missing fuzzy rules for signal type: {signal_type}")
        
        # Check that signal weights sum to approximately 1.0
        total_weight = sum(self.signal_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Signal weights sum to {total_weight:.3f}, should be 1.0")
        
        return errors


# Global instance for system-wide use
fuzzy_signal_engine = FuzzySignalEngine()


# Utility functions for testing and validation

def test_fuzzy_engine():
    """Test basic functionality of fuzzy signal engine"""
    engine = FuzzySignalEngine()
    
    # Test with sample intent scores
    test_scores = {
        'jurisdictional': 0.6,
        'policy': 0.3,
        'binary': 0.8,
        'personal_data': 0.4,
        'consequence': 0.2,
        'temporal': 0.1
    }
    
    print("Testing Fuzzy Signal Engine...")
    print(f"Input scores: {test_scores}")
    
    # Process signals
    signal_strengths, structural_risk = engine.process_signals(test_scores)
    
    print(f"Signal strengths: {signal_strengths}")
    print(f"Overall structural risk: {structural_risk:.3f}")
    
    # Get detailed breakdown
    details = engine.get_signal_details(test_scores)
    print(f"Fuzzy memberships: {details['fuzzy_memberships']}")
    
    # Validate configuration
    errors = engine.validate_configuration()
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid")
    
    return True


if __name__ == "__main__":
    test_fuzzy_engine()