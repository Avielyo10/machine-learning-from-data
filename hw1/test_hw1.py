import unittest
from hw1 import preprocess, apply_bias_trick, compute_cost, gradient_descent
import numpy as np
import pandas as pd


class TestPreprocessFunction(unittest.TestCase):

    def test_mean_normalization(self):
        # Test case for normal numerical ranges
        X = df["sqft_living"].values
        y = df["price"].values
        X_norm, y_norm = preprocess(X, y)
        self.assertTrue(np.allclose(X_norm.mean(), 0), "X mean is not normalized")
        self.assertTrue(np.allclose(y_norm.mean(), 0), "y mean is not normalized")
        # check if the range is normalized
        self.assertTrue(
            np.allclose(X_norm.max() - X_norm.min(), 1), "X range is not normalized"
        )
        self.assertTrue(
            np.allclose(y_norm.max() - y_norm.min(), 1), "y range is not normalized"
        )

    def test_negative_values(self):
        # Test case for inputs with negative values
        X = df["sqft_living"].values
        y = df["price"].values
        X_norm, y_norm = preprocess(X, y)
        self.assertTrue(
            np.allclose(X_norm.mean(), 0),
            "X mean with negative values is not normalized",
        )
        self.assertTrue(
            np.allclose(y_norm.mean(), 0),
            "y mean with negative values is not normalized",
        )
        # check if the range is normalized
        self.assertTrue(
            np.allclose(X_norm.max() - X_norm.min(), 1),
            "X range with negative values is not normalized",
        )
        self.assertTrue(
            np.allclose(y_norm.max() - y_norm.min(), 1),
            "y range with negative values is not normalized",
        )


class TestApplyBiasTrick(unittest.TestCase):

    def test_single_dimensional_array(self):
        # Test case for a single-dimensional array
        X = np.array([1, 2, 3])
        X_bias = apply_bias_trick(X.reshape(-1, 1))
        # [[1. 1.]
        # [1. 2.]
        # [1. 3.]]
        # Check if the first column consists entirely of ones
        self.assertTrue(np.all(X_bias[:, 0] == 1), "First column is not all ones")
        # Check if the shape is correct
        self.assertEqual(X_bias.shape, (3, 2), "Shape of the output is incorrect")
        # Check if the rest of the data remains unchanged
        self.assertTrue(np.all(X_bias[:, 1] == X), "Rest of the data is modified")

    def test_multi_dimensional_array(self):
        # Test case for a multi-dimensional array
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_bias = apply_bias_trick(X)
        # [[1. 1. 2.]
        # [1. 3. 4.]
        # [1. 5. 6.]]
        # Check if the first column consists entirely of ones
        self.assertTrue(
            np.all(X_bias[:, 0] == 1),
            "First column is not all ones in a multi-dimensional array",
        )
        # Check if the shape is correct
        self.assertEqual(
            X_bias.shape,
            (3, 3),
            "Shape of the output is incorrect in a multi-dimensional array",
        )
        # Check if the rest of the data remains unchanged
        self.assertTrue(
            np.all(X_bias[:, 1:] == X),
            "Rest of the data is modified in a multi-dimensional array",
        )


class TestComputeCost(unittest.TestCase):

    def test_known_cost(self):
        X = np.array([[1], [2]])
        y = np.array([5, 6])
        theta = np.array([0.1, 0.2])
        X_bias = apply_bias_trick(X)
        expected_cost = 13.085
        cost = compute_cost(X_bias, y, theta)
        self.assertEqual(cost, expected_cost, "Computed cost is incorrect")


class TestGradientDescent(unittest.TestCase):

    def test_theta_update(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        theta = np.array([0.5, 0.5])
        alpha = 0.01
        num_iters = 10
        X_bias = apply_bias_trick(X)
        final_theta, _ = gradient_descent(X_bias, y, theta, alpha, num_iters)
        self.assertNotEqual(final_theta[0], theta[0], "Theta was not updated")
        self.assertNotEqual(final_theta[1], theta[1], "Theta was not updated")

    def test_cost_reduction(self):
        X = np.array([[1], [2]])
        y = np.array([1, 2])
        theta = np.array([0, 0])
        alpha = 0.1
        num_iters = 100
        X_bias = apply_bias_trick(X)
        _, J_history = gradient_descent(X_bias, y, theta, alpha, num_iters)
        self.assertTrue(J_history[-1] < J_history[0], "Cost did not reduce")


if __name__ == "__main__":
    df = pd.read_csv("hw1/data.csv")
    unittest.main()
