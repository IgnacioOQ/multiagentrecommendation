from imports import *
from agents import BaseQLearningAgent

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

    def step(self,*_):
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

    def modify_reward(self, reward,*_):
        return reward + self.mood
    
class ReceptorModulator:
    def __init__(self, alpha=0.0001, beta=0.001, min_sensitivity=0.1, max_sensitivity=1.0):
        """
        Simulates receptor downregulation — a key biological process where the brain becomes
        less sensitive to stimuli after repeated exposure (e.g., tolerance to dopamine-triggering rewards).

        Parameters:
            alpha (float): 
                Controls how fast sensitivity decays in response to reward (desensitization).
                Higher alpha = faster downregulation.
            beta (float): 
                Controls how fast sensitivity recovers when reward is absent (resensitization).
            min_sensitivity (float): 
                Floor for how insensitive the system can become — prevents total shutdown.
            max_sensitivity (float): 
                Baseline sensitivity before downregulation (typically 1.0).
        """
        self.sensitivity = max_sensitivity       # Initial sensitivity starts fully responsive
        self.alpha = alpha                       # Desensitization rate (reward-driven)
        self.beta = beta                         # Recovery rate (homeostatic rebound)
        self.min_sensitivity = min_sensitivity   # Lower limit on receptor responsiveness
        self.max_sensitivity = max_sensitivity   # Upper limit (baseline setpoint)
        self.cumulative_reward = 0.0             # (Optional/unused in this version)
        self.modulation_history = []             # Log of each reward modulation for plotting or diagnostics
        self.setpoint_reward = 0  # Target reward level for homeostasis

    def modify_reward(self, reward, step=None, *_):
        """
        Modulates the input reward using current sensitivity level.
        
        This simulates the effect of receptor downregulation: 
        even if an external stimulus is strong, internal response weakens over time.

        Parameters:
            reward (float): Raw reward from the environment.
            step (int, optional): Current timestep (used for logging).
            *_: placeholder to allow compatibility with other modulators that accept more args.

        Returns:
            effective_reward (float): Scaled reward after applying internal sensitivity.
        """
        effective_reward = self.sensitivity * reward  # Perceived reward is scaled by internal sensitivity

        # Store data for post-simulation analysis (e.g., plotting sensitivity decay over time)
        self.modulation_history.append({
            "step": step,
            "original": reward,
            "sensitivity": self.sensitivity,
            "effective": effective_reward
        })

        return effective_reward

    def step(self, reward=None, *_):
        """
        Updates internal receptor sensitivity based on current reward exposure.

        Biological interpretation:
        - If the agent receives positive reward, it becomes less sensitive over time (desensitization).
        - If the agent receives no reward, sensitivity recovers toward baseline (resensitization).

        Parameters:
            reward (float or None): Reward used to determine downregulation (optional).
            *_: placeholder to accept and ignore extra arguments for compatibility.

        Effects:
            Updates self.sensitivity by increasing or decreasing it,
            then clips the result to the allowed range [min_sensitivity, max_sensitivity].
        """
        if abs(reward - self.setpoint_reward) < 10:
            # Close enough → recover
            self.sensitivity += self.beta * (self.max_sensitivity - self.sensitivity)
        else:
            # Desensitize
            # self.sensitivity -= self.alpha * abs(reward - self.setpoint_reward)
            self.sensitivity -= self.alpha * abs(abs(reward) - self.setpoint_reward)


        # Ensure sensitivity stays within bounds
        self.sensitivity = np.clip(self.sensitivity, self.min_sensitivity, self.max_sensitivity)


        
class NoveltyModulator:
    def __init__(self, eta=1.0):
        """
        A reward modulator that adds a novelty bonus to raw environmental rewards.

        Biological motivation:
            Dopamine systems in the brain show elevated firing in response to novel or infrequent stimuli.
            This modulator captures that behavior by rewarding novelty via inverse visitation frequency.

        Parameters:
            eta (float): Controls the strength of the novelty bonus.
                        Higher values place more weight on novelty; eta=0 disables novelty entirely.
        """
        self.visit_counts = defaultdict(int)  # Tracks how many times each (context, recommendation) pair was seen
        self.eta = eta                        # Novelty sensitivity parameter
        self.modulation_history = []          # Records all modulation steps for plotting or debugging

    def modify_reward(self, raw_reward, step, context, recommendation):
        """
        Computes a novelty-adjusted reward by adding a bonus for less-frequent (context, recommendation) pairs.

        Parameters:
            context (int): The current context of the environment (e.g., user state).
            recommendation (int): The item being recommended.
            raw_reward (float): The reward from the environment (e.g., utility, engagement).
            step (int, optional): Current timestep (for logging).

        Returns:
            modulated_reward (float): Final reward used for learning, adjusted for novelty.
        """
        key = (context, recommendation)

        # Increment visit count for this (context, recommendation) pair
        self.visit_counts[key] += 1

        # Compute novelty: inverse square root of visit count (decays over time)
        # Novelty = 1 on first visit, ~0.7 after 2 visits, ~0.3 after 10 visits, etc.
        novelty = 1.0 / np.sqrt(self.visit_counts[key])

        # Add novelty bonus to original reward
        modulated_reward = raw_reward + self.eta * novelty

        # Log this step for visualization or analysis
        self.modulation_history.append({
            "step": step,
            "context": context,
            "recommendation": recommendation,
            "novelty": novelty,
            "original": raw_reward,
            "modulated": modulated_reward
        })

        return modulated_reward

    def step(self, *_):
        """
        Optional step method for interface compatibility with other modulators (e.g. receptor modulator).
        Does nothing here because novelty is state-based, not time-dependent.
        """
        pass



class HomeostaticModulator(BaseQLearningAgent):
    def __init__(self, exploration_rate=1.0, exploration_decay=0.999,
                 min_exploration_rate=0.001, strategy='egreedy', setpoint=0, lag=0):
        self.moves = np.arange(-20, 20)
        self.setpoint = setpoint
        self.modulation_history = []
        self.lag = lag
        self.pending_modulations = deque(maxlen=lag + 1)  # FIFO queue
        super().__init__(
            n_actions=len(self.moves),
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate,
            strategy=strategy
        )

    def act(self, exogenous_reward):
        key = round(exogenous_reward, 2)
        action_idx = self.choose_action(key)
        return self.moves[action_idx]

    def modify_reward(self, exogenous_reward, step=None,*_):
        """
        Delays the application of modulation by `lag` steps.
        Learns now, but modulates later.
        """
        # Compute modulation for current exogenous_reward
        modulation = self.act(exogenous_reward)
        self.update_modulation(exogenous_reward, modulation)  # learning still happens now

        # Store for future application
        self.pending_modulations.append((step, exogenous_reward, modulation))

        # If enough time has passed, apply oldest modulation
        if step is not None and step >= self.lag:
            apply_step, old_exo, old_mod = self.pending_modulations.popleft()
            modulated_reward = old_exo - old_mod
        else:
            apply_step, old_exo, old_mod = step, exogenous_reward, 0
            modulated_reward = exogenous_reward

        # Record history
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "applied_step": apply_step,
            "modulation": old_mod,
            "modulated_reward": modulated_reward,
            "setpoint": self.setpoint
        })

        return modulated_reward

    def update_modulation(self, exogenous_reward, modulation):
        key = round(exogenous_reward, 2)
        action_idx = np.where(self.moves == modulation)[0][0]
        modulated_reward = exogenous_reward - modulation
        reward = -abs(modulated_reward - self.setpoint)
        self.update(key, action_idx, reward)
        return modulated_reward

    def plot_modulation_trajectory(self, max_points=1000):
        if not self.modulation_history:
            print("No modulation history to plot.")
            return

        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward", color="blue")
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (lagged)", color="green")
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation", color="purple")
        plt.xlabel("Step")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def step(self,*_):
        pass

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