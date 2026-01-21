from src.imports import *

# Publication-quality style settings
PAPER_STYLE = {
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
}

def apply_paper_style():
    """Apply publication-quality style settings to matplotlib."""
    plt.rcParams.update(PAPER_STYLE)


def plot_full_results(environment_state_space, average_reward_map, average_recommender_map):
    """
    Plots the environment state space, agent learned rewards, and recommender agent average rewards side by side.
    """
    apply_paper_style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # --- Client Agent learned rewards ---
    im1 = axs[0].imshow(average_reward_map, cmap="viridis", origin="lower", aspect="auto")
    axs[0].set_title("User Agent Average Rewards")
    axs[0].set_xlabel("Context")
    axs[0].set_ylabel("Recommendation")
    plt.colorbar(im1, ax=axs[0])

    # --- Recommender Agent average rewards ---
    recommender_min = np.nanmin(average_recommender_map)
    recommender_max = np.nanmax(average_recommender_map)
    im2 = axs[1].imshow(
        average_recommender_map,
        cmap="RdYlGn",
        origin="lower",
        aspect="auto",
        vmin=recommender_min,
        vmax=recommender_max
    )
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
    apply_paper_style()
    steps = range(len(sim_results["recommender_rewards"]))

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    # --- Rolling Mean ---
    axes[0].plot(sim_results["recommender_rolling_mean"], label="Recommender Rolling Mean", linewidth=1.5)
    axes[0].set_title(f"Rolling Mean (Window={rolling_window})")
    axes[0].set_ylabel("Rolling Mean")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Rolling Variance ---
    axes[1].plot(sim_results["recommender_rolling_var"], label="Recommender Rolling Variance", linewidth=1.5)
    axes[1].set_title(f"Rolling Variance (Window={rolling_window})")
    axes[1].set_ylabel("Rolling Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Cumulative Average ---
    axes[2].plot(sim_results["recommender_avg_rewards"], label="Recommender Cumulative Avg", linewidth=1.5)
    axes[2].set_title("Cumulative Average Reward")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Cumulative Average")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sensitivity(modulator, variable="sensitivity"):
    """Plot receptor sensitivity or novelty over time."""
    apply_paper_style()
    if not modulator.modulation_history:
        print("No modulation history recorded.")
        return

    history = pd.DataFrame(modulator.modulation_history)
    plt.figure(figsize=(10, 4))
    plt.plot(history["step"], history[variable], label=variable, linewidth=1.5)
    plt.title(f"Receptor {variable} Over Time")
    plt.xlabel("Step")
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# NEW PUBLICATION-QUALITY VISUALIZATION FUNCTIONS
# ============================================================================

def plot_initial_vs_final_qvalues(
    initial_user_qvalues,
    final_user_qvalues,
    environment_state_space,
    n_contexts,
    n_recommendations,
    figsize=(14, 10),
    save_path=None
):
    """
    Create publication-quality comparison of initial vs final Q-value landscapes.
    
    Shows the learning progression of the user agent by comparing Q-values before
    and after training, alongside the true environment rewards.
    
    Parameters:
        initial_user_qvalues: 2D array (n_recommendations, n_contexts) of initial Q-values
        final_user_qvalues: 2D array (n_recommendations, n_contexts) of final Q-values
        environment_state_space: 2D array of true environment rewards
        n_contexts: number of contexts
        n_recommendations: number of recommendations
        figsize: tuple for figure size
        save_path: optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    apply_paper_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Compute shared color limits for Q-values
    all_qvalues = np.concatenate([initial_user_qvalues.flatten(), final_user_qvalues.flatten()])
    q_vmin, q_vmax = np.nanpercentile(all_qvalues, [2, 98])
    
    # (a) True Environment Rewards
    im0 = axes[0, 0].imshow(
        environment_state_space, 
        cmap="viridis", 
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations]
    )
    axes[0, 0].set_title("(a) True Environment Rewards", fontweight='bold')
    axes[0, 0].set_xlabel("Context")
    axes[0, 0].set_ylabel("Recommendation")
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    cbar0.set_label("Reward")
    
    # (b) Initial User Q-Values
    im1 = axes[0, 1].imshow(
        initial_user_qvalues, 
        cmap="plasma", 
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations],
        vmin=q_vmin,
        vmax=q_vmax
    )
    axes[0, 1].set_title("(b) Initial User Q-Values", fontweight='bold')
    axes[0, 1].set_xlabel("Context")
    axes[0, 1].set_ylabel("Recommendation")
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    cbar1.set_label("Q-Value")
    
    # (c) Final User Q-Values
    im2 = axes[1, 0].imshow(
        final_user_qvalues, 
        cmap="plasma", 
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations],
        vmin=q_vmin,
        vmax=q_vmax
    )
    axes[1, 0].set_title("(c) Final User Q-Values", fontweight='bold')
    axes[1, 0].set_xlabel("Context")
    axes[1, 0].set_ylabel("Recommendation")
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    cbar2.set_label("Q-Value")
    
    # (d) Learning Change (Final - Initial)
    difference = final_user_qvalues - initial_user_qvalues
    diff_abs_max = np.nanmax(np.abs(difference))
    im3 = axes[1, 1].imshow(
        difference, 
        cmap="RdBu_r",  # Red for increase, Blue for decrease
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations],
        vmin=-diff_abs_max,
        vmax=diff_abs_max
    )
    axes[1, 1].set_title("(d) Learning Change (Final − Initial)", fontweight='bold')
    axes[1, 1].set_xlabel("Context")
    axes[1, 1].set_ylabel("Recommendation")
    cbar3 = plt.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    cbar3.set_label("ΔQ-Value")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_reward_distribution_analysis(
    sim_results,
    n_contexts,
    n_recommendations,
    rolling_window=5000,
    figsize=(14, 10),
    save_path=None
):
    """
    Create publication-quality reward distribution analysis.
    
    Shows the distribution of rewards received by the user agent, including
    a histogram, spatial distribution heatmap, and cumulative reward over time.
    
    Parameters:
        sim_results: dict output from run_recommender_simulation()
        n_contexts: number of contexts
        n_recommendations: number of recommendations
        rolling_window: window size for rolling statistics
        figsize: tuple for figure size
        save_path: optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    apply_paper_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    recommended_rewards = np.array(sim_results["recommended_rewards"])
    recommender_rewards = np.array(sim_results["recommender_rewards"])
    average_reward_map = sim_results["average_reward_map"]
    
    # (a) Histogram of User Rewards
    ax0 = axes[0, 0]
    # Filter out zeros (rejected recommendations) for cleaner histogram
    non_zero_rewards = recommended_rewards[recommended_rewards != 0]
    ax0.hist(non_zero_rewards, bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
    ax0.axvline(np.mean(non_zero_rewards), color='#E94F37', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(non_zero_rewards):.2f}')
    ax0.axvline(np.median(non_zero_rewards), color='#F6AE2D', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(non_zero_rewards):.2f}')
    ax0.set_title("(a) Distribution of User Rewards (Accepted Only)", fontweight='bold')
    ax0.set_xlabel("Reward Value")
    ax0.set_ylabel("Frequency")
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # (b) Spatial Reward Heatmap
    ax1 = axes[0, 1]
    im1 = ax1.imshow(
        average_reward_map, 
        cmap="viridis", 
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations]
    )
    ax1.set_title("(b) Average Reward by State (Context, Recommendation)", fontweight='bold')
    ax1.set_xlabel("Context")
    ax1.set_ylabel("Recommendation")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Average Reward")
    
    # (c) Cumulative Reward Over Time
    ax2 = axes[1, 0]
    cumulative_user = np.cumsum(recommended_rewards)
    steps = np.arange(len(recommended_rewards))
    ax2.plot(steps, cumulative_user, color='#2E86AB', linewidth=1.5, label='User Cumulative Reward')
    ax2.fill_between(steps, 0, cumulative_user, alpha=0.3, color='#2E86AB')
    ax2.set_title("(c) Cumulative User Reward Over Time", fontweight='bold')
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Cumulative Reward")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # (d) Rolling Average Reward
    ax3 = axes[1, 1]
    rolling_mean = pd.Series(recommended_rewards).rolling(rolling_window, min_periods=1).mean()
    rolling_std = pd.Series(recommended_rewards).rolling(rolling_window, min_periods=1).std()
    
    ax3.plot(steps, rolling_mean, color='#E94F37', linewidth=1.5, label=f'Rolling Mean (window={rolling_window})')
    ax3.fill_between(steps, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                     alpha=0.2, color='#E94F37', label='±1 Std Dev')
    ax3.set_title("(d) Rolling Average User Reward", fontweight='bold')
    ax3.set_xlabel("Simulation Step")
    ax3.set_ylabel("Average Reward")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_accept_reject_analysis(
    sim_results,
    n_contexts,
    n_recommendations,
    rolling_window=5000,
    figsize=(14, 10),
    save_path=None
):
    """
    Create publication-quality accept/reject behavior analysis.
    
    Demonstrates how the user agent learns to accept beneficial recommendations
    and reject suboptimal ones through the recommender system.
    
    Parameters:
        sim_results: dict output from run_recommender_simulation()
        n_contexts: number of contexts
        n_recommendations: number of recommendations
        rolling_window: window size for rolling statistics
        figsize: tuple for figure size
        save_path: optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    apply_paper_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    accept_history = sim_results["accept_history"]
    
    # Extract data from accept_history: (step, context, recommendation, accept)
    steps = np.array([entry[0] for entry in accept_history])
    contexts = np.array([entry[1] for entry in accept_history])
    recommendations = np.array([entry[2] for entry in accept_history])
    accepts = np.array([entry[3] for entry in accept_history])
    
    # (a) Overall Accept Rate Over Time
    ax0 = axes[0, 0]
    rolling_accept_rate = pd.Series(accepts.astype(float)).rolling(rolling_window, min_periods=1).mean()
    ax0.plot(steps, rolling_accept_rate, color='#2E86AB', linewidth=1.5)
    ax0.axhline(np.mean(accepts), color='#E94F37', linestyle='--', 
                linewidth=1.5, label=f'Overall Rate: {np.mean(accepts):.3f}')
    ax0.set_title("(a) Acceptance Rate Over Time", fontweight='bold')
    ax0.set_xlabel("Simulation Step")
    ax0.set_ylabel("Acceptance Rate")
    ax0.set_ylim([0, 1])
    ax0.legend(loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # (b) Accept Rate Heatmap by (Context, Recommendation)
    ax1 = axes[0, 1]
    accept_counts = np.zeros((n_recommendations, n_contexts))
    total_counts = np.zeros((n_recommendations, n_contexts))
    
    for ctx, rec, acc in zip(contexts, recommendations, accepts):
        accept_counts[rec, ctx] += acc
        total_counts[rec, ctx] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        accept_rate_map = np.true_divide(accept_counts, total_counts)
        accept_rate_map[total_counts == 0] = np.nan
    
    im1 = ax1.imshow(
        accept_rate_map, 
        cmap="RdYlGn",  # Red = low accept, Green = high accept
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations],
        vmin=0,
        vmax=1
    )
    ax1.set_title("(b) Acceptance Rate by State", fontweight='bold')
    ax1.set_xlabel("Context")
    ax1.set_ylabel("Recommendation")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Acceptance Rate")
    
    # (c) Reward Comparison: Accepted vs Rejected
    ax2 = axes[1, 0]
    recommended_rewards = np.array(sim_results["recommended_rewards"])
    
    accepted_rewards = recommended_rewards[accepts == 1]
    rejected_rewards = recommended_rewards[accepts == 0]
    
    # Box plot comparison
    bp = ax2.boxplot([accepted_rewards, rejected_rewards], 
                     labels=['Accepted', 'Rejected'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#E94F37')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax2.set_title("(c) Reward Distribution: Accepted vs Rejected", fontweight='bold')
    ax2.set_ylabel("Reward Value")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean annotations
    ax2.annotate(f'Mean: {np.mean(accepted_rewards):.2f}', 
                 xy=(1, np.mean(accepted_rewards)), xytext=(1.3, np.mean(accepted_rewards)),
                 fontsize=10, ha='left')
    ax2.annotate(f'Mean: {np.mean(rejected_rewards):.2f}', 
                 xy=(2, np.mean(rejected_rewards)), xytext=(2.3, np.mean(rejected_rewards)),
                 fontsize=10, ha='left')
    
    # (d) Recommendation Frequency Heatmap (where recommender suggests)
    ax3 = axes[1, 1]
    recommendation_counts = np.zeros((n_recommendations, n_contexts))
    
    for ctx, rec in zip(contexts, recommendations):
        recommendation_counts[rec, ctx] += 1
    
    im3 = ax3.imshow(
        recommendation_counts, 
        cmap="Blues",
        origin="lower", 
        aspect="auto",
        extent=[0, n_contexts, 0, n_recommendations]
    )
    ax3.set_title("(d) Recommendation Frequency by State", fontweight='bold')
    ax3.set_xlabel("Context")
    ax3.set_ylabel("Recommendation")
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("Frequency")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def plot_learning_summary(
    environment_state_space,
    initial_user_qvalues,
    final_user_qvalues,
    sim_results,
    n_contexts,
    n_recommendations,
    figsize=(16, 12),
    save_path=None
):
    """
    Create a comprehensive single-figure learning summary.
    
    Combines the key insights from all other visualizations into one
    publication-ready figure that tells the complete story of how the
    recommender system helps the user agent learn.
    
    Parameters:
        environment_state_space: 2D array of true environment rewards
        initial_user_qvalues: 2D array of initial Q-values
        final_user_qvalues: 2D array of final Q-values
        sim_results: dict output from run_recommender_simulation()
        n_contexts: number of contexts
        n_recommendations: number of recommendations
        figsize: tuple for figure size
        save_path: optional path to save the figure
    
    Returns:
        fig: matplotlib figure object
    """
    apply_paper_style()
    
    fig = plt.figure(figsize=figsize)
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Extract data
    accept_history = sim_results["accept_history"]
    accepts = np.array([entry[3] for entry in accept_history])
    steps = np.array([entry[0] for entry in accept_history])
    contexts = np.array([entry[1] for entry in accept_history])
    recommendations = np.array([entry[2] for entry in accept_history])
    recommended_rewards = np.array(sim_results["recommended_rewards"])
    
    # Shared colormap limits for Q-values
    all_qvalues = np.concatenate([initial_user_qvalues.flatten(), final_user_qvalues.flatten()])
    q_vmin, q_vmax = np.nanpercentile(all_qvalues, [2, 98])
    
    # (a) True Environment
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(environment_state_space, cmap="viridis", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations])
    ax0.set_title("(a) True Environment", fontweight='bold', fontsize=11)
    ax0.set_xlabel("Context", fontsize=9)
    ax0.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im0, ax=ax0, shrink=0.7)
    
    # (b) Initial User Q-Values
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(initial_user_qvalues, cmap="plasma", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations], vmin=q_vmin, vmax=q_vmax)
    ax1.set_title("(b) Initial User Q-Values", fontweight='bold', fontsize=11)
    ax1.set_xlabel("Context", fontsize=9)
    ax1.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im1, ax=ax1, shrink=0.7)
    
    # (c) Final User Q-Values
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(final_user_qvalues, cmap="plasma", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations], vmin=q_vmin, vmax=q_vmax)
    ax2.set_title("(c) Final User Q-Values", fontweight='bold', fontsize=11)
    ax2.set_xlabel("Context", fontsize=9)
    ax2.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im2, ax=ax2, shrink=0.7)
    
    # (d) Learning Change
    ax3 = fig.add_subplot(gs[1, 0])
    difference = final_user_qvalues - initial_user_qvalues
    diff_abs_max = np.nanmax(np.abs(difference))
    im3 = ax3.imshow(difference, cmap="RdBu_r", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations],
                     vmin=-diff_abs_max, vmax=diff_abs_max)
    ax3.set_title("(d) Q-Value Change", fontweight='bold', fontsize=11)
    ax3.set_xlabel("Context", fontsize=9)
    ax3.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im3, ax=ax3, shrink=0.7)
    
    # (e) Acceptance Rate Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    accept_counts = np.zeros((n_recommendations, n_contexts))
    total_counts = np.zeros((n_recommendations, n_contexts))
    for ctx, rec, acc in zip(contexts, recommendations, accepts):
        accept_counts[rec, ctx] += acc
        total_counts[rec, ctx] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        accept_rate_map = np.true_divide(accept_counts, total_counts)
        accept_rate_map[total_counts == 0] = np.nan
    im4 = ax4.imshow(accept_rate_map, cmap="RdYlGn", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations], vmin=0, vmax=1)
    ax4.set_title("(e) Acceptance Rate by State", fontweight='bold', fontsize=11)
    ax4.set_xlabel("Context", fontsize=9)
    ax4.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im4, ax=ax4, shrink=0.7)
    
    # (f) Recommendation Frequency
    ax5 = fig.add_subplot(gs[1, 2])
    recommendation_counts = np.zeros((n_recommendations, n_contexts))
    for ctx, rec in zip(contexts, recommendations):
        recommendation_counts[rec, ctx] += 1
    im5 = ax5.imshow(recommendation_counts, cmap="Blues", origin="lower", aspect="auto",
                     extent=[0, n_contexts, 0, n_recommendations])
    ax5.set_title("(f) Recommendation Frequency", fontweight='bold', fontsize=11)
    ax5.set_xlabel("Context", fontsize=9)
    ax5.set_ylabel("Recommendation", fontsize=9)
    plt.colorbar(im5, ax=ax5, shrink=0.7)
    
    # (g) Acceptance Rate Over Time (spans 2 columns)
    ax6 = fig.add_subplot(gs[2, 0:2])
    rolling_window = 5000
    rolling_accept_rate = pd.Series(accepts.astype(float)).rolling(rolling_window, min_periods=1).mean()
    ax6.plot(steps, rolling_accept_rate, color='#2E86AB', linewidth=1.2)
    ax6.axhline(np.mean(accepts), color='#E94F37', linestyle='--', linewidth=1.2, 
                label=f'Overall: {np.mean(accepts):.3f}')
    ax6.set_title("(g) Acceptance Rate Over Time", fontweight='bold', fontsize=11)
    ax6.set_xlabel("Simulation Step", fontsize=9)
    ax6.set_ylabel("Acceptance Rate", fontsize=9)
    ax6.set_ylim([0, 1])
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # (h) Cumulative Reward
    ax7 = fig.add_subplot(gs[2, 2])
    cumulative_reward = np.cumsum(recommended_rewards)
    ax7.plot(steps, cumulative_reward, color='#2E86AB', linewidth=1.2)
    ax7.fill_between(steps, 0, cumulative_reward, alpha=0.3, color='#2E86AB')
    ax7.set_title("(h) Cumulative User Reward", fontweight='bold', fontsize=11)
    ax7.set_xlabel("Simulation Step", fontsize=9)
    ax7.set_ylabel("Cumulative Reward", fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def extract_user_qvalues_from_agent(recommended_agent, n_contexts, n_recommendations):
    """
    Extract user Q-values for 'accept' action from the recommended agent's Q-table.
    
    Parameters:
        recommended_agent: the RecommendedAgent instance
        n_contexts: number of contexts
        n_recommendations: number of recommendations
    
    Returns:
        qvalues: 2D numpy array (n_recommendations, n_contexts) of Q-values for accept
    """
    qvalues = np.zeros((n_recommendations, n_contexts))
    
    for context in range(n_contexts):
        for recommendation in range(n_recommendations):
            key = (context, recommendation)
            if key in recommended_agent.q_table:
                # Action 0 is 'accept'
                qvalues[recommendation, context] = recommended_agent.q_table[key][0]
            else:
                qvalues[recommendation, context] = 0.0
    
    return qvalues

