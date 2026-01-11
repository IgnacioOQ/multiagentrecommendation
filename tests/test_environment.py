import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import ExogenousRewardEnvironment

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = ExogenousRewardEnvironment(n_recommendations=20, n_contexts=50)

    def test_initialization(self):
        self.assertEqual(self.env.n_recommendations, 20)
        self.assertEqual(self.env.n_contexts, 50)
        self.assertEqual(self.env.state_space.shape, (20, 50))

    def test_get_state_value(self):
        # By default initialized to 0s then gaussian smoothing applied or just 0s if not called
        # The __init__ calls `do_gaussian_smoothing` commented out?
        # Checking the code: yes, it's commented out in __init__.
        # But `simulations.py` calls it.
        # Let's call it here to populate values.
        self.env.do_gaussian_smoothing()

        val = self.env.get_state_value(10, 10)
        self.assertIsNotNone(val)
        self.assertIsInstance(val, (float, np.float64, np.float32))

        # Test out of bounds
        self.assertIsNone(self.env.get_state_value(-1, 0))
        self.assertIsNone(self.env.get_state_value(0, 100))

    def test_step_context(self):
        start_context = self.env.current_context
        self.env.step_context()
        next_context = self.env.current_context
        self.assertNotEqual(start_context, next_context) # Probabilistic, but highly likely
        self.assertTrue(0 <= next_context < 50)
        self.assertIn(next_context, self.env.context_history)

    def test_shift_environment(self):
        self.env.do_gaussian_smoothing()
        initial_state = self.env.state_space.copy()

        self.env.shift_environment_right()
        shifted_state = self.env.state_space

        # Column 1 in shifted should be Column 0 in initial
        np.testing.assert_array_equal(shifted_state[:, 1], initial_state[:, 0])
        # Column 0 in shifted should be last Column in initial
        np.testing.assert_array_equal(shifted_state[:, 0], initial_state[:, -1])

if __name__ == '__main__':
    unittest.main()
