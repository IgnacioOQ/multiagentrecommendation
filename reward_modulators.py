from imports import *

class MoodSwings:
    def __init__(self, n_moods=50):
        """
        Non-stationary mood model where the probabilities of mood changes drift over time.
        """
        self.n_moods = n_moods
        self.mood = 0
        self.mood_history = [self.mood]
        self.moves = np.arange(-10, 10)#[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        self.time = 0

        # Start with symmetric probabilities
        self.move_probs = np.ones(len(self.moves)) / len(self.moves)

    def step(self):
        self.time += 1

        # --- Non-stationarity: Slowly bias move_probs over time ---
        drift_factor = np.sin(self.time / 50)  # Smooth periodic drift
        weights = np.array(self.moves) * drift_factor
        self.move_probs = np.exp(weights)  # Exponential biasing
        self.move_probs /= self.move_probs.sum()  # Normalize to sum to 1

        move = np.random.choice(self.moves, p=self.move_probs)
        self.mood += move

        # Wrap around
        if self.mood > self.n_moods:
            self.mood = -self.n_moods + (self.mood - self.n_moods - 1)
        elif self.mood < -self.n_moods:
            self.mood = self.n_moods - (abs(self.mood) - self.n_moods - 1)

        self.mood_history.append(self.mood)

    def get_mood(self):
        return self.mood

    def reset(self):
        self.mood = 0
        self.mood_history = [self.mood]
        self.time = 0

    def modify_reward(self, reward):
        return reward + self.mood
    
    
class HomeostaticModulator:
    def __init__(self,
                exploration_rate=1,
                exploration_decay=0.999,
                min_exploration_rate=0.001,
                type='egreedy',
                setpoint=0):
        """
        Agent decides to accept/decline a recommendation given context and recommendation.
        """
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.moves = np.arange(-20, 20)
        self.n_actions = len(self.moves)  # 0 = accept, 1 = decline
        self.q_table = {}  # key: (exogenous_reward)
        self.action_counts = {}
        self.time = 0  # for UCB
        self.type = type
        self.setpoint = setpoint

    def act(self, exogenous_reward):
        """
        Decide to accept (True) or decline (False) the recommendation.
        """
        key = exogenous_reward
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
            self.action_counts[key] = np.zeros(self.n_actions)

        self.time += 1

        if self.type == 'egreedy':
            action = self.egreedy_choice(key)
        if self.type == 'ucb':
            action = self.ucb_choice(key)
        if self.type == 'softmax':
            action = self.softmax_choice(key)

        self.action_counts[key][action] += 1
        # action = self.moves[action]
        return action  # Number between -20 and 20

    def egreedy_choice(self, key):
        """
        Epsilon-greedy selection.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[key])

    def ucb_choice(self, key):
        """
        Upper Confidence Bound (UCB) selection.
        """
        total_counts = np.sum(self.action_counts[key]) + 1e-5
        ucb_values = self.q_table[key] + \
                    self.exploration_rate * np.sqrt(np.log(self.time + 1) / (self.action_counts[key] + 1e-5))
        return np.argmax(ucb_values)

    def softmax_choice(self, context, tau=1.0):
        """
        Softmax-based action selection using scipy's softmax.
    
        Args:
            context (int): The current state or context.
            tau (float): Temperature parameter to adjust exploration level.
    
        Returns:
            int: Selected action index.
        """
        q_values = self.q_table[context]
        # Apply temperature
        scaled_qs = q_values / tau
        probabilities = softmax(scaled_qs)
        return np.random.choice(self.n_actions, p=probabilities)

    def modify_reward(self, exogenous_reward):
        action = self.act(exogenous_reward)
        modulated_reward = self.update(exogenous_reward, action)
        return modulated_reward

    def update(self, exogenous_reward, action):
        """
        Update Q-value after observing reward.
        """
        key = exogenous_reward
        n = self.action_counts[key][action]
        q = self.q_table[key][action]
        modulated_reward = exogenous_reward - self.moves[action]  # Action is the index in moves
        reward = -abs(modulated_reward - self.setpoint)
        self.q_table[key][action] = q + (reward - q) / n

        # Decay exploration for epsilon-greedy
        if self.type == 'egreedy':
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * self.exploration_decay)
                
        return modulated_reward
                
# ============================================
# PID Controller
# ============================================

class PIDController:
    """
    Classic PID controller.

    Parameters:
    - kp: Proportional gain → reacts to present error.
    - ki: Integral gain → reacts to accumulated past error.
    - kd: Derivative gain → reacts to rate of change (future error prediction).
    """
    def __init__(self, kp=1.5, ki=0.2, kd=0.1, target_level=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_level = target_level
        self.integral = 0.0
        self.prev_error = 0.0
        self.level_trajectories = []

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_action(self, current_level):
        """
        Applies PID (Proportional-Integral-Derivative) control formula
        to compute the control action based on the current homeostasis level.

        Parameters:
        - current_level (float): The current homeostasis level of the environment.

        Returns:
        - action (float): The control adjustment to be applied.

        --- PID Control Breakdown ---

        1. Proportional Term (P):
           - Reacts to the **current error**.
           - Larger error → stronger response.
           - Computed as: kp * error.

        2. Integral Term (I):
           - Reacts to the **accumulated past error** (integral over time).
           - Helps eliminate steady-state errors.
           - Computed as: ki * sum of past errors.

        3. Derivative Term (D):
           - Reacts to the **rate of change of the error** (prediction of future error).
           - Helps dampen oscillations.
           - Computed as: kd * change in error.
        """
        # Step 1: Calculate current error (difference between target and actual level)
        error = self.target_level - current_level

        # Step 2: Update integral (accumulated sum of errors over time)
        self.integral += error

        # Step 3: Calculate derivative (rate of error change)
        derivative = error - self.prev_error

        # Step 4: Apply PID formula:
        action = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Step 5: Store current error for next derivative calculation
        self.prev_error = error

        return action

    def run(self, env, episodes=300, max_steps=50):
        """
        Run PID controller, saving levels & forces at key episodes.
        """
        print("\nRunning PID Controller...")
        save_episodes = [0, episodes // 4, episodes // 2, (3 * episodes) // 4, episodes - 1]
        history = []

        for episode in tqdm(range(episodes)):
            level = np.random.uniform(-10, 10)
            self.reset()
            total_reward = 0
            episode_levels = [level]
            episode_forces = []

            for step in range(max_steps):
                action = self.compute_action(level)
                force = env.get_external_force(step, episode)
                level = env.transition(level, action, step, episode)
                total_reward += env.reward(level)

                episode_levels.append(level)
                episode_forces.append(force)

            history.append(total_reward)

            if episode in save_episodes:
                self.level_trajectories.append((episode, episode_levels, episode_forces))

        return history

    def plot_levels(self):
        """
        Plot multiple PID level trajectories over key episodes (no external force overlay).
        """
        plt.figure(figsize=(12, 5))
        for ep_idx, trajectory, _ in self.level_trajectories:
            plt.plot(trajectory, label=f'Episode {ep_idx}')
        plt.axhline(self.target_level, color='black', linestyle='--', label='Target Level')
        plt.xlabel('Step')
        plt.ylabel('Homeostasis Level')
        plt.title('PID Controller Level Trajectories')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_external_forces(self):
        """
        Plot external forces applied during key episodes.
        """
        plt.figure(figsize=(12, 5))
        for ep_idx, _, forces in self.level_trajectories:
            plt.plot(forces, label=f'Episode {ep_idx}')
        plt.xlabel('Step')
        plt.ylabel('External Force')
        plt.title('External Forces - PID Controller')
        plt.legend()
        plt.grid(True)
        plt.show()