from imports import *
from reward_modulators import *
from agents import *
from environment import ExogenousRewardEnvironment

def run_recommender_simulation(
    recommender_agent_class,
    recommended_agent_class,
    environment_class,
    modulator_class,
    n_recommendations=20,
    n_contexts=50,
    n_steps: int = 1000000,
    exploration_rate=0.05,
    exploration_decay=0.99,
    rolling_window: int = 1000,
    random_seed: int = None,
    initialize_recommender = True,
    initialize_recommended = True,
    stationarity = True,
    landscape_type='default',
    strategy='egreedy',
    modulated = False
) -> dict:
    """
    Runs the simulation between a recommender and a recommended agent using ExogenousRewardEnvironment.

    Parameters:
        recommender_agent_class: RecommenderAgent class (not instance)
        recommended_agent_class: RecommendedAgent class (not instance)
        environment_class: ExogenousRewardEnvironment class (not instance)
        n_steps: number of simulation steps
        rolling_window: size of window for rolling stats (mean, variance)
        random_seed: optional, for reproducibility

    Returns:
        histories: dict with rewards, averages, and rolling statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Initialize environment and agents
    environment = environment_class(n_recommendations=n_recommendations, n_contexts=n_contexts)
    # environment.do_rows_gaussian_smoothing()
    if landscape_type == 'default':
        environment.do_gaussian_smoothing()
    elif landscape_type == 'rows':
        # Use row-wise Gaussian smoothing
        environment.do_rows_gaussian_smoothing()
    # environment.do_rows_gaussian_smoothing()
    recommender_agent = recommender_agent_class(
        num_recommendations=n_recommendations,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay
    )
    recommended_agent = recommended_agent_class(exploration_rate=exploration_rate,exploration_decay=exploration_decay,strategy=strategy)
    modulator = modulator_class() if modulated else None

    recommender_rewards = []
    recommender_contexts = []
    recommended_rewards = []
    recommender_avg_rewards = []
    recommended_avg_rewards = []
    accept_history = []
    original_modulated_differences = []  # <--- NEW


    recommender_action_map = np.zeros((environment.n_recommendations, environment.n_contexts))
    recommender_action_counts = np.zeros((environment.n_recommendations, environment.n_contexts))
    reward_map = np.zeros((environment.n_recommendations, environment.n_contexts))
    reward_counts = np.zeros((environment.n_recommendations, environment.n_contexts))

    total_rec_reward = 0
    total_user_reward = 0

    environment.visualize_landscape()

    if initialize_recommender:
        n_actions = environment.n_recommendations
        n_contexts = environment.n_contexts

        recommender_agent.q_table = {}
        recommender_agent.action_counts = {}

        # Parameters
        noise_std = 0.3          # Moderate noise
        target_min, target_max = -1.0, 1.0  # Match actual reward scale of recommender

        # Rescale environment state space to [-1, 1]
        env_min = np.min(environment.state_space)
        env_max = np.max(environment.state_space)

        if env_max > env_min:
            scale = (target_max - target_min) / (env_max - env_min)
            normalized_state_space = target_min + scale * (environment.state_space - env_min)
        else:
            normalized_state_space = np.full_like(environment.state_space, (target_min + target_max) / 2)

        for context in range(n_contexts):
            q_values = np.zeros(n_actions)
            recommender_agent.action_counts[context] = np.zeros(n_actions)

            for recommendation in range(n_actions):
                base_value = normalized_state_space[recommendation, context]
                noisy_value = base_value + np.random.normal(0.0, noise_std)
                q_values[recommendation] = np.clip(noisy_value, target_min, target_max)

            recommender_agent.q_table[context] = q_values

        recommender_agent.visualize_q_landscape(
            context_list=range(n_contexts),
            title='Recommender Agent Initial Estimates'
        )



    if initialize_recommended:
        local_rec, local_ctx = environment.local_max_pos

        # Shared parameters
        noise_std = 7.0
        strength = 70.0
        lower_bound = -50
        upper_bound = 100

        if landscape_type == 'default':
            # --- 2D Gaussian bump around (local_ctx, local_rec) ---
            context_spread = environment.n_contexts / 10
            rec_spread = environment.n_recommendations / 10

            ctx_midpoint = environment.n_contexts // 2
            rec_midpoint = environment.n_recommendations // 2

            for context in range(ctx_midpoint):  # Bottom half only
                for recommendation in range(rec_midpoint):  # Left half only
                    key = (context, recommendation)

                    # Gaussian distance from local max
                    dist_sq = ((context - local_ctx) ** 2) / (2 * context_spread ** 2) + \
                            ((recommendation - local_rec) ** 2) / (2 * rec_spread ** 2)

                    weight = np.exp(-dist_sq)
                    bias = weight * strength

                    # Initialize Q-values and action counts
                    recommended_agent.q_table[key] = np.zeros(2)
                    recommended_agent.action_counts[key] = np.zeros(2)

                    # Biased prior for 'accept' (action=0), neutral/noisy for 'reject' (action=1)
                    accept_prior = bias + np.random.normal(0, noise_std)
                    recommended_agent.q_table[key][0] = np.clip(accept_prior, lower_bound, upper_bound)
                    recommended_agent.q_table[key][1] = np.random.normal(0, 1.0)

        elif landscape_type == 'rows':
            rec_spread = environment.n_recommendations / 10  # Std deviation in recommendation space
            midpoint = environment.n_recommendations // 2

            for context in range(environment.n_contexts):
                for recommendation in range(midpoint):  # Only initialize lower half
                    key = (context, recommendation)

                    # Gaussian bump bias
                    dist_sq = ((recommendation - local_rec) ** 2) / (2 * rec_spread ** 2)
                    weight = np.exp(-dist_sq)
                    bias = weight * strength

                    # Initialize Q-values and action counts
                    recommended_agent.q_table[key] = np.zeros(2)
                    recommended_agent.action_counts[key] = np.zeros(2)

                    # Biased prior for 'accept' (action=0), neutral/noisy for 'reject' (action=1)
                    accept_prior = bias + np.random.normal(0, noise_std)
                    recommended_agent.q_table[key][0] = np.clip(accept_prior, lower_bound, upper_bound)
                    recommended_agent.q_table[key][1] = np.random.normal(0, 1.0)

        # Visualization
        recommended_agent.visualize_accept_q_landscape(
            context_list=range(environment.n_contexts),
            recommendation_list=range(environment.n_recommendations),
            title="Recommended Agent's Initial Q Landscape"
        )

    for step in trange(n_steps, desc="Running Simulation"):
        context = environment.current_context
        recommendation = recommender_agent.act(context)
        accept = recommended_agent.act(context, recommendation)
        accept_history.append((step, context, recommendation, accept))

        # if accept:
        #     agent_reward = environment.state_space[recommendation, context]
        #     if modulated:
        #         modulated_reward = modulator_class.modify_reward(agent_reward)
        #     recommender_reward = 1
        # else:
        #     agent_reward = 0
        #     if modulated:
        #         modulated_reward = modulator_class.modify_reward(agent_reward)
        #     recommender_reward = -1

        if accept:
            agent_reward = environment.state_space[recommendation, context]
            if modulated:
                # modulated_reward = modulator.modify_reward(agent_reward, step)
                modulated_reward = modulator.modify_reward(agent_reward, step,context, recommendation)
                modulator.step(agent_reward)
            recommender_reward = 1
        else:
            agent_reward = 0
            if modulated:
                # modulated_reward = modulator.modify_reward(agent_reward, step=step)
                modulated_reward = modulator.modify_reward(agent_reward, step,context, recommendation)
                modulator.step(agent_reward)
            recommender_reward = -1

        # Update agents
        reward_to_use = modulated_reward if modulated else agent_reward
        recommended_agent.update_reward(context, recommendation, accept, reward_to_use)
        recommended_rewards.append(reward_to_use)
        original_modulated_differences.append(agent_reward - reward_to_use)
            
        recommender_agent.update_reward(context, recommendation, recommender_reward)
        recommender_rewards.append(recommender_reward)
        recommender_contexts.append(context)

        total_user_reward += agent_reward
        total_rec_reward += recommender_reward

        recommended_avg_rewards.append(total_user_reward / (step + 1))
        recommender_avg_rewards.append(total_rec_reward / (step + 1))

        reward_map[recommendation, context] += agent_reward
        reward_counts[recommendation, context] += 1
        recommender_action_map[recommendation, context] += recommender_reward
        recommender_action_counts[recommendation, context] += 1

        environment.step_context()
        if not stationarity:
        #   if step % 100 == 0:
            environment.shift_environment_right()

    # Compute rolling statistics
    recommender_rewards_series = pd.Series(recommender_rewards)
    recommended_rewards_series = pd.Series(recommended_rewards)

    recommender_rolling_mean = recommender_rewards_series.rolling(rolling_window).mean().tolist()
    recommended_rolling_mean = recommended_rewards_series.rolling(rolling_window).mean().tolist()

    recommender_rolling_var = recommender_rewards_series.rolling(rolling_window).var().tolist()
    recommended_rolling_var = recommended_rewards_series.rolling(rolling_window).var().tolist()

    with np.errstate(divide='ignore', invalid='ignore'):
        average_reward_map = np.true_divide(reward_map, reward_counts)
        average_reward_map[reward_counts == 0] = np.nan
        average_recommender_map = np.true_divide(recommender_action_map, recommender_action_counts)
        average_recommender_map[recommender_action_counts == 0] = np.nan

    recommender_agent.visualize_q_landscape(range(n_contexts),title='Recommender Agent Learned Rewards')
    recommended_agent.visualize_accept_q_landscape(range(n_contexts), range(n_recommendations), title='User Learned Rewards')

    return {
        "recommender_rewards": recommender_rewards,
        "recommended_rewards": recommended_rewards,
        "recommender_avg_rewards": recommender_avg_rewards,
        "recommended_avg_rewards": recommended_avg_rewards,
        "recommender_rolling_mean": recommender_rolling_mean,
        "recommended_rolling_mean": recommended_rolling_mean,
        "recommender_rolling_var": recommender_rolling_var,
        "recommended_rolling_var": recommended_rolling_var,
        "context_history": environment.get_context_history(),
        "average_reward_map": average_reward_map,
        "environment_state_space": environment.state_space,
        "context_history": recommender_contexts,
        "average_recommender_map": average_recommender_map,
        "accept_history": accept_history,
        "original_modulated_differences": original_modulated_differences,
        "modulator": modulator  # <--- Add this
    }

