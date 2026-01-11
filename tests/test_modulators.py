import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_modulators import ReceptorModulator, HomeostaticModulator, TD_DHR, TD_DHR_D

class TestModulators(unittest.TestCase):

    def test_receptor_modulator(self):
        # High sensitivity initially
        modulator = ReceptorModulator(max_sensitivity=1.0, alpha=0.1, desensitization_threshold=5.0)

        # Reward below threshold -> resensitization (remains max)
        rew = modulator.modify_reward(4.0)
        modulator.step(4.0)
        self.assertEqual(modulator.sensitivity, 1.0)
        self.assertEqual(rew, 4.0)

        # Reward above threshold -> desensitization
        # sensitivity = 1.0 - 0.1 * 10 = 0.0? No wait, logic:
        # sensitivity -= alpha * reward
        # 1.0 - 0.1 * 10 = 0.0. Min is usually 0.1
        modulator.step(10.0)
        # modify_reward uses CURRENT sensitivity before step update usually,
        # but here we called step manually.
        # Let's check sensitivity state.
        self.assertLess(modulator.sensitivity, 1.0)

    def test_homeostatic_modulator(self):
        # Simple test of logic flow
        mod = HomeostaticModulator(setpoint=0, lag=0, n_bins=5)
        # Should act like an agent
        # modify_reward returns modulated value

        # Test basic modulation application (random init usually means 0 modulation or random)
        # But wait, it's a Q-learner.

        exo = 10.0
        mod_rew = mod.modify_reward(exo, step=0)
        # mod_rew = exo - modulation.
        # Check that it returns a number
        self.assertIsInstance(mod_rew, (float, np.float64, int))

    def test_td_dhr(self):
        # Test TD_DHR initialization and basic step
        dhr = TD_DHR(setpoint=0, history_length=3)

        # 3 steps to fill history partially
        dhr.modify_reward(10.0, step=1)
        dhr.modify_reward(10.0, step=2)
        res = dhr.modify_reward(10.0, step=3)

        self.assertIsInstance(res, (float, np.float64))
        self.assertEqual(len(dhr.history), 3)

    def test_td_dhr_d(self):
        # Test Dynamic setpoint
        dhr = TD_DHR_D(setpoint=0, top_threshold=10, adjustment_factor=0.1)

        # Consistent high reward should lower setpoint
        dhr.modify_reward(20.0, step=1)
        # 20 > 10, dist=10, decrease = 1.0. New setpoint = -1.0

        self.assertAlmostEqual(dhr.current_setpoint, -1.0)
        self.assertEqual(dhr.setpoint, -1.0)

if __name__ == '__main__':
    unittest.main()
