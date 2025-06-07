from imports import *
from reward_modulators import *


# def run_recommender_simulation(
#     recommender_agent_class,
#     recommended_agent_class,
#     environment_class,
#     n_recommendations=20,
#     n_contexts=50,
#     n_steps: int = 1000000,
#     exploration_rate=0.05,
#     exploration_decay=0.99,
#     random_seed: int = None,
#     initialize_recommender = True,
#     initialize_recommended = True,
#     stationarity = True,
#     landscape_type='default',
#     type='egreedy',
# ) -> dict:
#     """
#     Runs the simulation between a recommender and a recommended agent using ExogenousRewardEnvironment.

#     Parameters:
#         recommender_agent_class: RecommenderAgent class (not instance)
#         recommended_agent_class: RecommendedAgent class (not instance)
#         environment_class: ExogenousRewardEnvironment class (not instance)
#         n_steps: number of simulation steps
#         random_seed: optional, for reproducibility

#     Returns:
#         histories: dict with recommender and recommended rewards over time, and reward map by (recommendation, context)
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)
#         random.seed(random_seed)

#     # Initialize environment and agents
#     environment = environment_class(n_recommendations=n_recommendations, n_contexts=n_contexts)
#     if landscape_type == 'default':
#         environment.do_gaussian_smoothing()
#     elif landscape_type == 'rows':
#         # Use row-wise Gaussian smoothing
#         environment.do_rows_gaussian_smoothing()
#     # environment.do_rows_gaussian_smoothing()
#     recommender_agent = recommender_agent_class(num_recommendations=n_recommendations,exploration_rate=exploration_rate,exploration_decay=exploration_decay)
#     recommended_agent = recommended_agent_class(exploration_rate=exploration_rate,exploration_decay=exploration_decay,type=type)

#     recommender_rewards = []
#     recommender_action_map = np.zeros((environment.n_recommendations, environment.n_contexts))
#     recommender_action_counts = np.zeros((environment.n_recommendations, environment.n_contexts))
#     recommended_rewards = []
#     reward_map = np.zeros((environment.n_recommendations, environment.n_contexts))
#     reward_counts = np.zeros((environment.n_recommendations, environment.n_contexts))

#     if initialize_recommender:
#       n_actions = environment.n_recommendations
#       n_contexts = environment.n_contexts  # list or iterable of context values

#       recommender_agent.q_table = {}
#       recommender_agent.action_counts = {} # Initialize action_counts here

#       # Configuration
#       noise_std = 20.0          # Adjust this for more/less noisy priors
#       lower_bound = -50
#       upper_bound = 100

#       for context in range(n_contexts):
#           q_values = np.zeros(n_actions)
#           recommender_agent.action_counts[context] = np.zeros(n_actions) # Initialize action counts for this context
#           for recommendation in range(n_actions):
#               base_value = environment.state_space[recommendation, context]
#               noise = np.random.normal(loc=0.0, scale=noise_std)
#               noisy_value = base_value + noise
#               q_values[recommendation] = np.clip(noisy_value, lower_bound, upper_bound)
#           recommender_agent.q_table[context] = q_values
#       recommender_agent.visualize_q_landscape(range(n_contexts),title="Recommender Agent's Initial Q-value Landscape")

#     if initialize_recommended:
#         # --- Initialize recommended agent with prior around local max only ---
#         local_rec, local_ctx = environment.local_max_pos

#         # Parameters for localized Gaussian bias
#         noise_std = 5.0
#         strength = 70.0
#         lower_bound = -50
#         upper_bound = 100
#         context_spread = environment.n_contexts / 10
#         rec_spread = environment.n_recommendations / 10

#         for context in range(environment.n_contexts):
#             for recommendation in range(environment.n_recommendations):
#                 key = (context, recommendation)

#                 # Gaussian distance from local max
#                 dist_sq = ((context - local_ctx) ** 2) / (2 * context_spread ** 2) + \
#                         ((recommendation - local_rec) ** 2) / (2 * rec_spread ** 2)

#                 weight = np.exp(-dist_sq)
#                 bias = weight * strength

#                 recommended_agent.q_table[key] = np.zeros(2)
#                 recommended_agent.action_counts[key] = np.zeros(2)

#                 # Apply bias to accept (action=0) only
#                 accept_prior = bias + np.random.normal(0, noise_std)
#                 recommended_agent.q_table[key][0] = np.clip(accept_prior, lower_bound, upper_bound)
#                 # Leave reject (action=1) neutral
#                 recommended_agent.q_table[key][1] = np.random.normal(0, 1.0)

#         recommended_agent.visualize_accept_q_landscape(
#             context_list=range(environment.n_contexts),
#             recommendation_list=range(environment.n_recommendations),
#             title="Recommended Agent's Initial Q Landscape"
#         )


#     environment.visualize_landscape()

#     for step in trange(n_steps, desc="Running Simulation"):
#         # --- 1. Environment provides current context ---
#         context = environment.current_context

#         # --- 2. Recommender makes recommendation ---
#         recommendation = recommender_agent.act(context)

#         # --- 3. Recommended agent decides to accept/reject ---
#         accept = recommended_agent.act(context, recommendation)

#         # --- 4. Compute rewards ---
#         if accept:
#             agent_reward = environment.state_space[recommendation, context]
#             recommender_reward = 1
#         else:
#             agent_reward = 0
#             recommender_reward = -1

#         # --- 5. Update agents ---
#         recommended_agent.update(context, recommendation, accept, agent_reward)
#         recommender_agent.update(context, recommendation, recommender_reward)

#         # --- 6. Store rewards ---
#         recommended_rewards.append(agent_reward)
#         recommender_rewards.append(recommender_reward)

#         # --- 7. Update reward map ---
#         reward_map[recommendation, context] += agent_reward
#         reward_counts[recommendation, context] += 1

#         # --- 8. Update recommender reward map ---
#         recommender_action_map[recommendation, context] += recommender_reward
#         recommender_action_counts[recommendation, context] += 1

#         # --- 8. Step environment to next context ---
#         environment.step_context()
#         if not stationarity:
#           environment.shift_environment_right()

#     # Calculate average rewards per (recommendation, context)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         average_reward_map = np.true_divide(reward_map, reward_counts)
#         average_reward_map[reward_counts == 0] = np.nan  # set empty cells to NaN

#         # Calculate average recommender rewards per (recommendation, context)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         average_recommender_map = np.true_divide(recommender_action_map, recommender_action_counts)
#         average_recommender_map[recommender_action_counts == 0] = np.nan

#     recommender_agent.visualize_q_landscape(range(n_contexts),title='Recommender Agent Learned Rewards')
#     recommended_agent.visualize_accept_q_landscape(range(n_contexts), range(n_recommendations), title='User Learned Rewards')

#     return {
#         "recommender_rewards": recommender_rewards,
#         "recommended_rewards": recommended_rewards,
#         "context_history": environment.get_context_history(),
#         "average_reward_map": average_reward_map,
#         "environment_state_space": environment.state_space,
#         "average_recommender_map": average_recommender_map
#     }


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
    type='egreedy',
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
    recommended_agent = recommended_agent_class(exploration_rate=exploration_rate,exploration_decay=exploration_decay,type=type)
    modulator_class = modulator_class()

    recommender_rewards = []
    recommender_contexts = []
    recommended_rewards = []
    recommender_avg_rewards = []
    recommended_avg_rewards = []
    accept_history = []
    original_recommended_differences = []  # <--- NEW


    recommender_action_map = np.zeros((environment.n_recommendations, environment.n_contexts))
    recommender_action_counts = np.zeros((environment.n_recommendations, environment.n_contexts))
    reward_map = np.zeros((environment.n_recommendations, environment.n_contexts))
    reward_counts = np.zeros((environment.n_recommendations, environment.n_contexts))

    total_rec_reward = 0
    total_user_reward = 0

    if initialize_recommender:
      n_actions = environment.n_recommendations
      n_contexts = environment.n_contexts  # list or iterable of context values

      recommender_agent.q_table = {}
      recommender_agent.action_counts = {} # Initialize action_counts here

      # Configuration
      noise_std = 20.0          # Adjust this for more/less noisy priors
      lower_bound = -50
      upper_bound = 100

      for context in range(n_contexts):
          q_values = np.zeros(n_actions)
          recommender_agent.action_counts[context] = np.zeros(n_actions) # Initialize action counts for this context
          for recommendation in range(n_actions):
              base_value = environment.state_space[recommendation, context]
              noise = np.random.normal(loc=0.0, scale=noise_std)
              noisy_value = base_value + noise
              q_values[recommendation] = np.clip(noisy_value, lower_bound, upper_bound)
          recommender_agent.q_table[context] = q_values
      recommender_agent.visualize_q_landscape(range(n_contexts))

    if initialize_recommended:
            # --- Initialize recommended agent with prior around local max only ---
            local_rec, local_ctx = environment.local_max_pos

            # Parameters for localized Gaussian bias
            noise_std = 5.0
            strength = 70.0
            lower_bound = -50
            upper_bound = 100
            context_spread = environment.n_contexts / 10
            rec_spread = environment.n_recommendations / 10

            for context in range(environment.n_contexts):
                for recommendation in range(environment.n_recommendations):
                    key = (context, recommendation)

                    # Gaussian distance from local max
                    dist_sq = ((context - local_ctx) ** 2) / (2 * context_spread ** 2) + \
                            ((recommendation - local_rec) ** 2) / (2 * rec_spread ** 2)

                    weight = np.exp(-dist_sq)
                    bias = weight * strength

                    recommended_agent.q_table[key] = np.zeros(2)
                    recommended_agent.action_counts[key] = np.zeros(2)

                    # Apply bias to accept (action=0) only
                    accept_prior = bias + np.random.normal(0, noise_std)
                    recommended_agent.q_table[key][0] = np.clip(accept_prior, lower_bound, upper_bound)
                    # Leave reject (action=1) neutral
                    recommended_agent.q_table[key][1] = np.random.normal(0, 1.0)

            recommended_agent.visualize_accept_q_landscape(
                context_list=range(environment.n_contexts),
                recommendation_list=range(environment.n_recommendations),
                title="Recommended Agent's Initial Q Landscape"
            )
        
    environment.visualize_landscape()

    for step in trange(n_steps, desc="Running Simulation"):
        context = environment.current_context
        recommendation = recommender_agent.act(context)
        accept = recommended_agent.act(context, recommendation)
        accept_history.append((step, context, recommendation, accept))

        if accept:
            agent_reward = environment.state_space[recommendation, context]
            if modulated:
                modulated_reward = modulator_class.modify_reward(agent_reward)
            recommender_reward = 1
        else:
            agent_reward = 0
            if modulated:
                modulated_reward = modulator_class.modify_reward(agent_reward)
            recommender_reward = -1
        
        if modulated:
            recommended_agent.update(context, recommendation, accept, modulated_reward)
            original_recommended_differences.append(agent_reward - modulated_reward)
            recommended_rewards.append(modulated_reward)
        else:
            recommended_agent.update(context, recommendation, accept, agent_reward)
            original_recommended_differences.append(agent_reward)
            recommended_rewards.append(agent_reward)
            
        recommender_agent.update(context, recommendation, recommender_reward)
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
        modulator_class.step()
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
        "original_recommended_differences": original_recommended_differences
    }

