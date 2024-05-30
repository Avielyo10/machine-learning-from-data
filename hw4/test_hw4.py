import unittest
import numpy as np

from scipy.stats import norm
from hw4 import norm_pdf


class TestNormPDF(unittest.TestCase):
    def test_norm_pdf(self):
        # Test case for norm_pdf function
        data = np.array([1, 2, 3, 4, 5])
        mu = 3
        sigma = 1
        pdf = norm_pdf(data, mu, sigma)

        # Expected output
        expected_pdf = norm.pdf(data, mu, sigma)

        # Check if the output is correct
        self.assertTrue(np.allclose(pdf, expected_pdf), "Output is incorrect")
        print(f"pdf: {pdf}")
        print(f"expected_pdf: {expected_pdf}")

    def test_norm_pdf_with_different_mu_sigma(self):
        # Test case for norm_pdf function with different mu and sigma
        data = np.array([1, 2, 3, 4, 5])
        mu = 2
        sigma = 2
        pdf = norm_pdf(data, mu, sigma)

        # Expected output
        expected_pdf = norm.pdf(data, mu, sigma)

        # Check if the output is correct
        self.assertTrue(np.allclose(pdf, expected_pdf), "Output is incorrect")
        print(f"pdf: {pdf}")
        print(f"expected_pdf: {expected_pdf}")

if __name__ == "__main__":
    unittest.main()
