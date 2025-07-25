#!/usr/bin/env python
"""
Basic functionality tests for Integral Shapley Values toolkit.

This module contains unit tests to verify the core functionality
of the integral Shapley implementation.
"""

import unittest
import numpy as np
import sys
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.integral_shapley import (
    compute_integral_shapley_value,
    monte_carlo_shapley_value,
    exact_shapley_value
)
from src.utils.utilities import utility_acc, utility_RKHS, utility_KL
from src.utils.model_utils import return_model


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the integral Shapley toolkit."""
    
    def setUp(self):
        """Set up test data and models."""
        # Create a simple synthetic dataset
        X, y = make_classification(
            n_samples=50, n_features=4, n_classes=2, 
            n_redundant=0, random_state=42
        )
        
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_valid = scaler.transform(self.x_valid)
        
        # Train final model
        self.final_model = return_model('LinearSVC')
        self.final_model.fit(self.x_train, self.y_train)
        self.clf = return_model('LinearSVC')
        
        self.target_point = 0
    
    def test_model_utils(self):
        """Test model utility functions."""
        # Test different model types
        models_to_test = ['LinearSVC', 'logistic', 'Tree']
        
        for model_name in models_to_test:
            with self.subTest(model=model_name):
                model = return_model(model_name)
                self.assertTrue(hasattr(model, 'fit'))
                self.assertTrue(hasattr(model, 'predict'))
    
    def test_utility_functions(self):
        """Test utility functions."""
        utilities = [utility_acc, utility_RKHS, utility_KL]
        
        # Create a simple subset
        subset_indices = [0, 1, 2]
        X_sub = self.x_train[subset_indices]
        y_sub = self.y_train[subset_indices]
        
        for utility_func in utilities:
            with self.subTest(utility=utility_func.__name__):
                try:
                    result = utility_func(
                        X_sub, y_sub, self.x_valid, self.y_valid, 
                        self.clf, self.final_model
                    )
                    
                    # Check that result is a valid number
                    self.assertIsInstance(result, (int, float, np.number))
                    self.assertFalse(np.isnan(result))
                    
                except Exception as e:
                    self.fail(f"Utility function {utility_func.__name__} failed: {e}")
    
    def test_integral_methods(self):
        """Test different integral Shapley methods."""
        methods = ['trapezoid', 'gaussian', 'smart_adaptive']
        
        for method in methods:
            with self.subTest(method=method):
                try:
                    # Use small parameters for speed
                    if method == 'trapezoid':
                        kwargs = {'num_t_samples': 5, 'num_MC': 10}
                    elif method == 'gaussian':
                        kwargs = {'num_nodes': 4, 'num_MC': 10}
                    else:  # smart_adaptive
                        kwargs = {'tolerance': 1e-2, 'num_MC': 10}
                    
                    result = compute_integral_shapley_value(
                        self.x_train, self.y_train, self.x_valid, self.y_valid,
                        self.target_point, self.clf, self.final_model,
                        utility_acc, method=method, **kwargs
                    )
                    
                    # Check that result is a valid number
                    self.assertIsInstance(result, (int, float, np.number))
                    self.assertFalse(np.isnan(result))
                    
                except Exception as e:
                    self.fail(f"Method {method} failed: {e}")
    
    def test_monte_carlo_shapley(self):
        """Test traditional Monte Carlo Shapley computation."""
        try:
            result = monte_carlo_shapley_value(
                self.target_point, self.x_train, self.y_train, 
                self.x_valid, self.y_valid, self.clf, self.final_model,
                utility_acc, num_samples=100
            )
            
            self.assertIsInstance(result, (int, float, np.number))
            self.assertFalse(np.isnan(result))
            
        except Exception as e:
            self.fail(f"Monte Carlo Shapley failed: {e}")
    
    def test_exact_shapley_small(self):
        """Test exact Shapley computation on very small dataset."""
        # Create tiny dataset for exact computation
        X_tiny = self.x_train[:8]  # Only 8 points
        y_tiny = self.y_train[:8]
        
        try:
            result = exact_shapley_value(
                0, X_tiny, y_tiny, self.x_valid, self.y_valid,
                self.clf, self.final_model, utility_acc
            )
            
            self.assertIsInstance(result, (int, float, np.number))
            self.assertFalse(np.isnan(result))
            
        except Exception as e:
            self.fail(f"Exact Shapley failed: {e}")
    
    def test_consistency_across_methods(self):
        """Test that different methods give reasonably consistent results."""
        # Small dataset for faster computation
        X_small = self.x_train[:12]
        y_small = self.y_train[:12]
        
        target = 0
        results = {}
        
        # Test different methods with small parameters
        methods_params = {
            'trapezoid': {'num_t_samples': 10, 'num_MC': 20},
            'gaussian': {'num_nodes': 8, 'num_MC': 20},
            'monte_carlo': {'num_samples': 200}
        }
        
        for method, params in methods_params.items():
            try:
                if method == 'monte_carlo':
                    result = monte_carlo_shapley_value(
                        target, X_small, y_small, self.x_valid, self.y_valid,
                        self.clf, self.final_model, utility_acc, **params
                    )
                else:
                    result = compute_integral_shapley_value(
                        X_small, y_small, self.x_valid, self.y_valid, target,
                        self.clf, self.final_model, utility_acc, 
                        method=method, **params
                    )
                
                results[method] = result
                
            except Exception as e:
                self.fail(f"Method {method} failed in consistency test: {e}")
        
        # Check that results are not too different (within reasonable bounds)
        if len(results) >= 2:
            values = list(results.values())
            relative_std = np.std(values) / (np.mean(np.abs(values)) + 1e-10)
            
            # Allow for reasonable variation due to sampling
            self.assertLess(relative_std, 2.0, 
                           f"Methods give very different results: {results}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single data point
        X_single = self.x_train[:1]
        y_single = self.y_train[:1]
        
        # This should handle gracefully (might return 0 or the full utility)
        try:
            result = compute_integral_shapley_value(
                X_single, y_single, self.x_valid, self.y_valid, 0,
                self.clf, self.final_model, utility_acc,
                method='trapezoid', num_t_samples=5, num_MC=10
            )
            # Should not crash
            self.assertIsInstance(result, (int, float, np.number))
            
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, (ValueError, IndexError))
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid method
        with self.assertRaises(ValueError):
            compute_integral_shapley_value(
                self.x_train, self.y_train, self.x_valid, self.y_valid,
                self.target_point, self.clf, self.final_model,
                utility_acc, method='invalid_method'
            )
        
        # Test invalid target point index
        with self.assertRaises((IndexError, ValueError)):
            compute_integral_shapley_value(
                self.x_train, self.y_train, self.x_valid, self.y_valid,
                len(self.x_train) + 10, self.clf, self.final_model,
                utility_acc, method='trapezoid', num_t_samples=5, num_MC=10
            )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions specifically."""
    
    def setUp(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=30, n_features=3, n_classes=2, 
            n_redundant=0, random_state=42
        )
        
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_valid = scaler.transform(self.x_valid)
        
        self.final_model = return_model('LinearSVC')
        self.final_model.fit(self.x_train, self.y_train)
        self.clf = return_model('LinearSVC')
    
    def test_accuracy_utility_bounds(self):
        """Test that accuracy utility returns values in [0,1]."""
        subset_indices = [0, 1, 2]
        X_sub = self.x_train[subset_indices]
        y_sub = self.y_train[subset_indices]
        
        result = utility_acc(
            X_sub, y_sub, self.x_valid, self.y_valid,
            self.clf, self.final_model
        )
        
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_utility_with_single_class(self):
        """Test utility functions with single-class subsets."""
        # Create subset with only one class
        class_0_indices = np.where(self.y_train == 0)[0][:3]
        X_sub = self.x_train[class_0_indices]
        y_sub = self.y_train[class_0_indices]
        
        # Should handle single-class case gracefully
        result = utility_acc(
            X_sub, y_sub, self.x_valid, self.y_valid,
            self.clf, self.final_model
        )
        
        self.assertIsInstance(result, (int, float, np.number))
        self.assertFalse(np.isnan(result))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\nAll tests passed successfully! ✓")
    else:
        print(f"\nSome tests failed. ✗")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()