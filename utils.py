from imports import *

def plot_full_results(environment_state_space, average_reward_map, average_recommender_map):
    """
    Plots the environment state space, agent learned rewards, and recommender agent average rewards side by side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Three subplots side by side

    # --- Agent learned rewards ---
    im1 = axs[0].imshow(average_reward_map, cmap="viridis", origin="lower", aspect="auto")
    axs[0].set_title("Client Agent Average Rewards")
    axs[0].set_xlabel("Context")
    axs[0].set_ylabel("Recommendation")
    plt.colorbar(im1, ax=axs[0])

    # --- Recommender agent rewards ---
    recommender_min = np.nanmin(average_recommender_map)
    recommender_max = np.nanmax(average_recommender_map)
    im2 = axs[1].imshow(average_recommender_map, cmap="coolwarm", origin="lower", aspect="auto",
                        vmin=recommender_min, vmax=recommender_max)
    axs[1].set_title("Recommender Agent Average Rewards")
    axs[1].set_xlabel("Context")
    axs[1].set_ylabel("Recommendation")
    plt.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()
    
    
def plot_reward_statistics(sim_results, rolling_window=1000):
        """
        Plots rolling mean, rolling variance, and cumulative average reward for both agents.

        Parameters:
            sim_results: dict output from run_recommender_simulation()
            rolling_window: int, size of the rolling window used (for title display)
        """
        steps = range(len(sim_results["recommender_rewards"]))

        fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

        # --- Rolling Mean ---
        axes[0].plot(sim_results["recommender_rolling_mean"], label="Recommender Rolling Mean")
        # axes[0].plot(sim_results["recommended_rolling_mean"], label="Recommended Rolling Mean")
        axes[0].set_title(f"Rolling Mean (Window={rolling_window})")
        axes[0].set_ylabel("Rolling Mean")
        axes[0].legend()

        # --- Rolling Variance ---
        axes[1].plot(sim_results["recommender_rolling_var"], label="Recommender Rolling Variance")
        # axes[1].plot(sim_results["recommended_rolling_var"], label="Recommended Rolling Variance")
        axes[1].set_title(f"Rolling Variance (Window={rolling_window})")
        axes[1].set_ylabel("Rolling Variance")
        axes[1].legend()

        # --- Cumulative Average ---
        axes[2].plot(sim_results["recommender_avg_rewards"], label="Recommender Cumulative Avg")
        # axes[2].plot(sim_results["recommended_avg_rewards"], label="Recommended Cumulative Avg")
        axes[2].set_title("Cumulative Average Reward")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Cumulative Average")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
