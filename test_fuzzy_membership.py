"""
Unit tests for fuzzy membership functions
Tests triangular, trapezoidal, and Gaussian membership functions
"""

import pytest
import math
from fuzzy_membership import (
    TriangularMF, TrapezoidalMF, GaussianMF,
    create_low_medium_high_triangular, create_low_medium_high_trapezoidal,
    validate_membership_function
)


class TestTriangularMF:
    """Test cases for TriangularMF class"""
    
    def test_triangular_initialization(self):
        """Test proper initialization of triangular membership function"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        assert mf.a == 0.0
        assert mf.b == 0.5
        assert mf.c == 1.0
    
    def test_triangular_invalid_parameters(self):
        """Test that invalid parameters raise ValueError"""
        with pytest.raises(ValueError):
            TriangularMF(1.0, 0.5, 0.0)  # a > b
        
        with pytest.raises(ValueError):
            TriangularMF(0.0, 1.0, 0.5)  # b > c
    
    def test_triangular_membership_peak(self):
        """Test membership at peak point"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        assert mf.membership(0.5) == 1.0
    
    def test_triangular_membership_boundaries(self):
        """Test membership at boundary points"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        assert mf.membership(0.0) == 0.0
        assert mf.membership(1.0) == 0.0
        assert mf.membership(-0.1) == 0.0  # Outside left
        assert mf.membership(1.1) == 0.0   # Outside right
    
    def test_triangular_membership_slopes(self):
        """Test membership on ascending and descending slopes"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        
        # Ascending slope
        assert mf.membership(0.25) == 0.5
        assert mf.membership(0.1) == 0.2
        
        # Descending slope
        assert mf.membership(0.75) == 0.5
        assert abs(mf.membership(0.9) - 0.2) < 1e-10  # Use approximate equality
    
    def test_triangular_edge_cases(self):
        """Test edge cases like zero-width triangles"""
        # Point triangle (all points same)
        mf = TriangularMF(0.5, 0.5, 0.5)
        assert mf.membership(0.5) == 1.0
        assert mf.membership(0.4) == 0.0
        assert mf.membership(0.6) == 0.0
    
    def test_triangular_range_validation(self):
        """Test that all membership values are in [0,1] range"""
        mf = TriangularMF(0.2, 0.6, 0.9)
        test_points = [i * 0.1 for i in range(-5, 15)]  # -0.5 to 1.4
        
        for point in test_points:
            membership = mf.membership(point)
            assert 0.0 <= membership <= 1.0, f"Membership {membership} out of range for input {point}"


class TestTrapezoidalMF:
    """Test cases for TrapezoidalMF class"""
    
    def test_trapezoidal_initialization(self):
        """Test proper initialization of trapezoidal membership function"""
        mf = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
        assert mf.a == 0.0
        assert mf.b == 0.3
        assert mf.c == 0.7
        assert mf.d == 1.0
    
    def test_trapezoidal_invalid_parameters(self):
        """Test that invalid parameters raise ValueError"""
        with pytest.raises(ValueError):
            TrapezoidalMF(1.0, 0.5, 0.7, 0.9)  # a > b
        
        with pytest.raises(ValueError):
            TrapezoidalMF(0.0, 0.5, 0.3, 1.0)  # b > c
        
        with pytest.raises(ValueError):
            TrapezoidalMF(0.0, 0.3, 0.7, 0.5)  # c > d
    
    def test_trapezoidal_membership_flat_top(self):
        """Test membership in flat top region"""
        mf = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
        assert mf.membership(0.3) == 1.0
        assert mf.membership(0.5) == 1.0
        assert mf.membership(0.7) == 1.0
    
    def test_trapezoidal_membership_boundaries(self):
        """Test membership at boundary points"""
        mf = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
        assert mf.membership(0.0) == 0.0
        assert mf.membership(1.0) == 0.0
        assert mf.membership(-0.1) == 0.0  # Outside left
        assert mf.membership(1.1) == 0.0   # Outside right
    
    def test_trapezoidal_membership_slopes(self):
        """Test membership on ascending and descending slopes"""
        mf = TrapezoidalMF(0.0, 0.4, 0.6, 1.0)
        
        # Ascending slope
        assert mf.membership(0.2) == 0.5
        assert mf.membership(0.1) == 0.25
        
        # Descending slope
        assert abs(mf.membership(0.8) - 0.5) < 1e-10  # Use approximate equality
        assert abs(mf.membership(0.9) - 0.25) < 1e-10  # Use approximate equality
    
    def test_trapezoidal_degenerate_to_triangle(self):
        """Test trapezoidal that degenerates to triangle (b == c)"""
        mf = TrapezoidalMF(0.0, 0.5, 0.5, 1.0)
        assert mf.membership(0.5) == 1.0
        assert mf.membership(0.25) == 0.5
        assert mf.membership(0.75) == 0.5
    
    def test_trapezoidal_range_validation(self):
        """Test that all membership values are in [0,1] range"""
        mf = TrapezoidalMF(0.1, 0.3, 0.7, 0.9)
        test_points = [i * 0.1 for i in range(-5, 15)]  # -0.5 to 1.4
        
        for point in test_points:
            membership = mf.membership(point)
            assert 0.0 <= membership <= 1.0, f"Membership {membership} out of range for input {point}"


class TestGaussianMF:
    """Test cases for GaussianMF class"""
    
    def test_gaussian_initialization(self):
        """Test proper initialization of Gaussian membership function"""
        mf = GaussianMF(0.5, 0.2)
        assert mf.center == 0.5
        assert mf.sigma == 0.2
    
    def test_gaussian_invalid_sigma(self):
        """Test that invalid sigma raises ValueError"""
        with pytest.raises(ValueError):
            GaussianMF(0.5, 0.0)  # sigma = 0
        
        with pytest.raises(ValueError):
            GaussianMF(0.5, -0.1)  # sigma < 0
    
    def test_gaussian_membership_center(self):
        """Test membership at center point"""
        mf = GaussianMF(0.5, 0.2)
        assert mf.membership(0.5) == 1.0
    
    def test_gaussian_membership_symmetry(self):
        """Test that Gaussian is symmetric around center"""
        mf = GaussianMF(0.5, 0.2)
        
        # Test symmetry
        assert abs(mf.membership(0.3) - mf.membership(0.7)) < 1e-10
        assert abs(mf.membership(0.4) - mf.membership(0.6)) < 1e-10
    
    def test_gaussian_membership_decay(self):
        """Test that membership decreases with distance from center"""
        mf = GaussianMF(0.5, 0.2)
        
        center_val = mf.membership(0.5)
        near_val = mf.membership(0.6)
        far_val = mf.membership(0.8)
        
        assert center_val > near_val > far_val
    
    def test_gaussian_range_validation(self):
        """Test that all membership values are in [0,1] range"""
        mf = GaussianMF(0.5, 0.3)
        test_points = [i * 0.1 for i in range(-10, 20)]  # -1.0 to 1.9
        
        for point in test_points:
            membership = mf.membership(point)
            assert 0.0 <= membership <= 1.0, f"Membership {membership} out of range for input {point}"


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_create_low_medium_high_triangular(self):
        """Test creation of standard LOW/MEDIUM/HIGH triangular functions"""
        mf_set = create_low_medium_high_triangular(0.0, 1.0)
        
        assert 'LOW' in mf_set
        assert 'MEDIUM' in mf_set
        assert 'HIGH' in mf_set
        
        # Test that LOW has high membership at low values
        assert mf_set['LOW'].membership(0.1) > 0.8
        
        # Test that HIGH has high membership at high values
        assert mf_set['HIGH'].membership(0.9) > 0.8
        
        # Test that MEDIUM peaks around middle
        assert mf_set['MEDIUM'].membership(0.5) > 0.8
    
    def test_create_low_medium_high_trapezoidal(self):
        """Test creation of standard LOW/MEDIUM/HIGH trapezoidal functions"""
        mf_set = create_low_medium_high_trapezoidal(0.0, 1.0)
        
        assert 'LOW' in mf_set
        assert 'MEDIUM' in mf_set
        assert 'HIGH' in mf_set
        
        # Test that each function has proper flat top regions
        assert mf_set['LOW'].membership(0.1) == 1.0
        assert mf_set['MEDIUM'].membership(0.5) == 1.0
        assert mf_set['HIGH'].membership(0.9) == 1.0
    
    def test_validate_membership_function(self):
        """Test membership function validation"""
        # Valid triangular function
        valid_mf = TriangularMF(0.0, 0.5, 1.0)
        assert validate_membership_function(valid_mf) == True
        
        # Valid trapezoidal function
        valid_trap = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
        assert validate_membership_function(valid_trap) == True
        
        # Valid Gaussian function
        valid_gauss = GaussianMF(0.5, 0.2)
        assert validate_membership_function(valid_gauss) == True


class TestPerformance:
    """Performance tests for membership functions"""
    
    def test_triangular_performance(self):
        """Test that triangular membership computation is fast"""
        import time
        
        mf = TriangularMF(0.0, 0.5, 1.0)
        test_points = [i * 0.001 for i in range(1000)]  # 1000 test points
        
        start_time = time.time()
        for point in test_points:
            mf.membership(point)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_computation = total_time / len(test_points)
        
        # Should be much faster than 1ms per computation
        assert avg_time_per_computation < 0.001, f"Too slow: {avg_time_per_computation:.6f}s per computation"
    
    def test_trapezoidal_performance(self):
        """Test that trapezoidal membership computation is fast"""
        import time
        
        mf = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
        test_points = [i * 0.001 for i in range(1000)]  # 1000 test points
        
        start_time = time.time()
        for point in test_points:
            mf.membership(point)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_computation = total_time / len(test_points)
        
        # Should be much faster than 1ms per computation
        assert avg_time_per_computation < 0.001, f"Too slow: {avg_time_per_computation:.6f}s per computation"


if __name__ == "__main__":
    # Run basic functionality tests
    print("Testing TriangularMF...")
    mf_tri = TriangularMF(0.0, 0.5, 1.0)
    print(f"Triangular(0.25) = {mf_tri.membership(0.25)}")
    print(f"Triangular(0.5) = {mf_tri.membership(0.5)}")
    print(f"Triangular(0.75) = {mf_tri.membership(0.75)}")
    
    print("\nTesting TrapezoidalMF...")
    mf_trap = TrapezoidalMF(0.0, 0.3, 0.7, 1.0)
    print(f"Trapezoidal(0.15) = {mf_trap.membership(0.15)}")
    print(f"Trapezoidal(0.5) = {mf_trap.membership(0.5)}")
    print(f"Trapezoidal(0.85) = {mf_trap.membership(0.85)}")
    
    print("\nTesting GaussianMF...")
    mf_gauss = GaussianMF(0.5, 0.2)
    print(f"Gaussian(0.3) = {mf_gauss.membership(0.3)}")
    print(f"Gaussian(0.5) = {mf_gauss.membership(0.5)}")
    print(f"Gaussian(0.7) = {mf_gauss.membership(0.7)}")
    
    print("\nTesting utility functions...")
    tri_set = create_low_medium_high_triangular()
    print(f"LOW(0.0) = {tri_set['LOW'].membership(0.0)}")
    print(f"MEDIUM(0.5) = {tri_set['MEDIUM'].membership(0.5)}")
    print(f"HIGH(1.0) = {tri_set['HIGH'].membership(1.0)}")
    
    print("\nAll basic tests passed!")