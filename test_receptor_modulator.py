
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reward_modulators import ReceptorModulator
from tqdm import trange

def test_receptor_modulator_behavior(alpha=0.001, beta=0.01, desensitization_threshold=50):
    """
    Tests the behavior of the ReceptorModulator by simulating its response to different reward levels.
    """
    modulator = ReceptorModulator(alpha=alpha, beta=beta, desensitization_threshold=desensitization_threshold)

    # Phase 1: High rewards to trigger desensitization
    high_rewards = np.linspace(60, 100, 200)

    # Phase 2: Low rewards to trigger recovery
    low_rewards = np.linspace(40, 0, 300)

    # Phase 3: Fluctuating rewards
    fluctuating_rewards = 50 + 40 * np.sin(np.linspace(0, 4 * np.pi, 500))

    rewards = np.concatenate([high_rewards, low_rewards, fluctuating_rewards])

    sensitivity_history = []

    for reward in rewards:
        modulator.step(reward)
        sensitivity_history.append(modulator.sensitivity)

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot rewards over time
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Reward", color="blue")
    plt.axhline(desensitization_threshold, color='red', linestyle='--', label=f'Desensitization Threshold ({desensitization_threshold})')
    plt.title("Reward Sequence")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # Plot sensitivity over time
    plt.subplot(2, 1, 2)
    plt.plot(sensitivity_history, label="Sensitivity", color="green")
    plt.title("Receptor Sensitivity Over Time")
    plt.xlabel("Step")
    plt.ylabel("Sensitivity")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("receptor_modulator_test.png")
    plt.show()

if __name__ == "__main__":
    test_receptor_modulator_behavior()
