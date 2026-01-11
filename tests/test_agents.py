import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents import BaseQLearningAgent, RecommenderAgent, RecommendedAgent

class TestAgents(unittest.TestCase):
    def test_base_agent_initialization(self):
        agent = BaseQLearningAgent(n_actions=5, strategy="egreedy")
        self.assertEqual(agent.n_actions, 5)
        self.assertEqual(agent.q_table, {})
        self.assertEqual(agent.time, 0)

    def test_base_agent_choose_action(self):
        agent = BaseQLearningAgent(n_actions=2, exploration_rate=0.0) # No exploration
        key = "context1"
        agent._ensure_key(key)
        agent.q_table[key] = np.array([10.0, 5.0]) # Action 0 is better

        action = agent.choose_action(key)
        self.assertEqual(action, 0)

        agent.q_table[key] = np.array([2.0, 8.0]) # Action 1 is better
        action = agent.choose_action(key)
        self.assertEqual(action, 1)

    def test_base_agent_update(self):
        agent = BaseQLearningAgent(n_actions=2, learning_rate=0.1)
        key = "context1"
        agent._ensure_key(key)
        # Initial Q: [0, 0]

        # Action 0, Reward 10
        # New Q = 0 + 0.1 * (10 - 0) = 1.0
        agent.update(key, action=0, reward=10)
        self.assertAlmostEqual(agent.q_table[key][0], 1.0)

        # Action 0, Reward 10 again
        # New Q = 1.0 + 0.1 * (10 - 1.0) = 1.0 + 0.9 = 1.9
        agent.update(key, action=0, reward=10)
        self.assertAlmostEqual(agent.q_table[key][0], 1.9)

    def test_recommender_agent(self):
        agent = RecommenderAgent(num_recommendations=10)
        self.assertEqual(agent.n_actions, 10)

        # Test act/update wrappers
        context = 5
        action = agent.act(context)
        self.assertTrue(0 <= action < 10)

        agent.update_reward(context, action, 5.0)
        self.assertTrue(context in agent.q_table)

    def test_recommended_agent(self):
        agent = RecommendedAgent()
        self.assertEqual(agent.n_actions, 2) # Accept/Reject

        context = 1
        rec = 5
        accepted = agent.act(context, rec)
        self.assertIn(accepted, [True, False])

        # Test update
        agent.update_reward(context, rec, accepted=True, reward=10.0)
        key = (context, rec)
        self.assertTrue(key in agent.q_table)
        # Action 0 is Accept
        self.assertNotEqual(agent.q_table[key][0], 0.0)

if __name__ == '__main__':
    unittest.main()
