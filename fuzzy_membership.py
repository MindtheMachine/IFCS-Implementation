"""
Fuzzy Membership Functions for Signal Strength Enhancement
Implements triangular and trapezoidal membership functions for fuzzy logic processing
"""

from typing import Union
import math


class TriangularMF:
    """Triangular membership function for fuzzy logic"""
    
    def __init__(self, a: float, b: float, c: float):
        """Initialize triangular membership function
        
        Args:
            a: Left point (start of triangle)
            b: Peak point (maximum membership = 1.0)
            c: Right point (end of triangle)
        """
        if not (a <= b <= c):
            raise ValueError(f"Invalid triangular MF parameters: a={a}, b={b}, c={c}. Must satisfy a <= b <= c")
        
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
    
    def membership(self, x: float) -> float:
        """Compute membership value for input x
        
        Args:
            x: Input value to compute membership for
            
        Returns:
            Membership value in [0,1] range
        """
        x = float(x)
        
        # Handle degenerate case (point triangle)
        if self.a == self.b == self.c:
            return 1.0 if abs(x - self.a) < 1e-10 else 0.0
        
        # Outside triangle bounds
        if x < self.a or x > self.c:
            return 0.0
        
        # At boundaries
        if abs(x - self.a) < 1e-10 or abs(x - self.c) < 1e-10:
            return 0.0
        
        # At peak
        if abs(x - self.b) < 1e-10:
            return 1.0
        
        # Left slope (ascending)
        if x < self.b:
            if abs(self.b - self.a) < 1e-10:  # Avoid division by zero
                return 1.0
            return (x - self.a) / (self.b - self.a)
        
        # Right slope (descending)
        else:
            if abs(self.c - self.b) < 1e-10:  # Avoid division by zero
                return 1.0
            return (self.c - x) / (self.c - self.b)
    
    def __repr__(self) -> str:
        return f"TriangularMF(a={self.a}, b={self.b}, c={self.c})"


class TrapezoidalMF:
    """Trapezoidal membership function for fuzzy logic"""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """Initialize trapezoidal membership function
        
        Args:
            a: Left bottom point (start of trapezoid)
            b: Left top point (start of flat top)
            c: Right top point (end of flat top)
            d: Right bottom point (end of trapezoid)
        """
        if not (a <= b <= c <= d):
            raise ValueError(f"Invalid trapezoidal MF parameters: a={a}, b={b}, c={c}, d={d}. Must satisfy a <= b <= c <= d")
        
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
    
    def membership(self, x: float) -> float:
        """Compute membership value for input x
        
        Args:
            x: Input value to compute membership for
            
        Returns:
            Membership value in [0,1] range
        """
        x = float(x)
        
        # Outside trapezoid bounds
        if x < self.a or x > self.d:
            return 0.0
        
        # At boundaries
        if abs(x - self.a) < 1e-10 or abs(x - self.d) < 1e-10:
            return 0.0
        
        # Flat top region (maximum membership)
        if self.b <= x <= self.c:
            return 1.0
        
        # Left slope (ascending)
        if x < self.b:
            if abs(self.b - self.a) < 1e-10:  # Avoid division by zero
                return 1.0
            return (x - self.a) / (self.b - self.a)
        
        # Right slope (descending)
        else:  # x > self.c
            if abs(self.d - self.c) < 1e-10:  # Avoid division by zero
                return 1.0
            return (self.d - x) / (self.d - self.c)
    
    def __repr__(self) -> str:
        return f"TrapezoidalMF(a={self.a}, b={self.b}, c={self.c}, d={self.d})"


class GaussianMF:
    """Gaussian membership function for fuzzy logic (optional, for smooth curves)"""
    
    def __init__(self, center: float, sigma: float):
        """Initialize Gaussian membership function
        
        Args:
            center: Center point (maximum membership = 1.0)
            sigma: Standard deviation (controls width)
        """
        if sigma <= 0:
            raise ValueError(f"Invalid Gaussian MF sigma: {sigma}. Must be > 0")
        
        self.center = float(center)
        self.sigma = float(sigma)
    
    def membership(self, x: float) -> float:
        """Compute membership value for input x
        
        Args:
            x: Input value to compute membership for
            
        Returns:
            Membership value in [0,1] range
        """
        x = float(x)
        
        # Gaussian formula: exp(-0.5 * ((x - center) / sigma)^2)
        exponent = -0.5 * ((x - self.center) / self.sigma) ** 2
        return math.exp(exponent)
    
    def __repr__(self) -> str:
        return f"GaussianMF(center={self.center}, sigma={self.sigma})"


# Utility functions for creating common membership function sets

def create_low_medium_high_triangular(min_val: float = 0.0, max_val: float = 1.0) -> dict:
    """Create standard LOW/MEDIUM/HIGH triangular membership functions
    
    Args:
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        
    Returns:
        Dictionary with 'LOW', 'MEDIUM', 'HIGH' triangular membership functions
    """
    mid_val = (min_val + max_val) / 2.0
    quarter = (max_val - min_val) / 4.0
    
    return {
        'LOW': TriangularMF(min_val, min_val + 1e-10, mid_val + quarter),
        'MEDIUM': TriangularMF(min_val + quarter, mid_val, max_val - quarter),
        'HIGH': TriangularMF(mid_val - quarter, max_val - 1e-10, max_val)
    }


def create_low_medium_high_trapezoidal(min_val: float = 0.0, max_val: float = 1.0) -> dict:
    """Create standard LOW/MEDIUM/HIGH trapezoidal membership functions
    
    Args:
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        
    Returns:
        Dictionary with 'LOW', 'MEDIUM', 'HIGH' trapezoidal membership functions
    """
    third = (max_val - min_val) / 3.0
    
    return {
        'LOW': TrapezoidalMF(min_val, min_val, min_val + third, min_val + 2*third),
        'MEDIUM': TrapezoidalMF(min_val + third/2, min_val + third, min_val + 2*third, max_val - third/2),
        'HIGH': TrapezoidalMF(min_val + 2*third, max_val - third, max_val, max_val)
    }


def validate_membership_function(mf: Union[TriangularMF, TrapezoidalMF, GaussianMF], 
                                test_points: list = None) -> bool:
    """Validate that a membership function produces values in [0,1] range
    
    Args:
        mf: Membership function to validate
        test_points: Optional list of test points, defaults to standard test set
        
    Returns:
        True if all membership values are in [0,1], False otherwise
    """
    if test_points is None:
        # Default test points covering typical range
        test_points = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    
    for point in test_points:
        membership_val = mf.membership(point)
        if not (0.0 <= membership_val <= 1.0):
            return False
    
    return True