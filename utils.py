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