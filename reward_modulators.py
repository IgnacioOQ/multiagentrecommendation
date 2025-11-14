from imports import *
from agents import BaseQLearningAgent
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple # Make sure deque and namedtuple are imported

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
    def __init__(self, alpha=0.0001, beta=0.001, min_sensitivity=0.1, max_sensitivity=1.0, desensitization_threshold=10):
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
            desensitization_threshold (float):
                The reward threshold above which the system desensitizes, and below which it recovers.
        """
        self.sensitivity = max_sensitivity       # Initial sensitivity starts fully responsive
        self.alpha = alpha                       # Desensitization rate (reward-driven)
        self.beta = beta                         # Recovery rate (homeostatic rebound)
        self.min_sensitivity = min_sensitivity   # Lower limit on receptor responsiveness
        self.max_sensitivity = max_sensitivity   # Upper limit (baseline setpoint)
        self.desensitization_threshold = desensitization_threshold
        self.modulation_history = []             # Log of each reward modulation for plotting or diagnostics

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
        - If the agent receives a reward above the threshold, it becomes less sensitive (desensitization).
        - If the agent receives a reward below the threshold, sensitivity recovers toward baseline (resensitization).

        Parameters:
            reward (float or None): Reward used to determine downregulation (optional).
            *_: placeholder to accept and ignore extra arguments for compatibility.

        Effects:
            Updates self.sensitivity by increasing or decreasing it,
            then clips the result to the allowed range [min_sensitivity, max_sensitivity].
        """
        if abs(reward) > self.desensitization_threshold:
            # High reward -> Desensitize
            self.sensitivity -= self.alpha * abs(reward)
        else:
            # Low or no reward -> Recover
            self.sensitivity += self.beta * (self.max_sensitivity - self.sensitivity)

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

        for episode in trange(episodes):
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

class HomeostaticModulator(BaseQLearningAgent):
    def __init__(self, setpoint=0, lag=0, n_bins=20, reward_range=(-100, 100), **kwargs):
        self.moves = np.arange(-50, 50)
        self.setpoint = setpoint
        self.modulation_history = []
        self.lag = lag
        self.pending_modulations = deque(maxlen=lag + 1)  # FIFO queue
        
        # --- FIX: Discretization ---
        self.n_bins = n_bins
        self.reward_range = reward_range
        # Create bin edges. e.g., 20 bins for range -100 to 100
        self.bins = np.linspace(self.reward_range[0], self.reward_range[1], self.n_bins + 1)
        # --- End Fix ---

        super().__init__(n_actions=len(self.moves), **kwargs)

    # --- FIX: New method to get discrete state key ---
    def _get_state_key(self, exogenous_reward):
        """Discretizes the continuous reward into a bin index."""
        # Clip reward to be within the defined range
        clipped_reward = np.clip(exogenous_reward, self.reward_range[0], self.reward_range[1])
        # Find which bin the reward falls into
        # np.digitize returns index i such that bins[i-1] <= x < bins[i]
        # We subtract 1 to get a 0-based index.
        # We also handle the edge case where reward == reward_range[1]
        bin_index = np.digitize(clipped_reward, self.bins) - 1
        return max(0, min(bin_index, self.n_bins - 1)) # Ensure it's within [0, n_bins-1]
    # --- End Fix ---

    def act(self, exogenous_reward, step=0): # Added step
        # --- FIX: Use discretized key ---
        key = self._get_state_key(exogenous_reward)
        # --- End Fix ---
        action_idx = self.choose_action(key, step=step) # Pass step
        return self.moves[action_idx]

    def modify_reward(self, exogenous_reward, step=None,*_):
        """
        Delays the application of modulation by `lag` steps.
        Learns now, but modulates later.
        """
        # Compute modulation for current exogenous_reward
        modulation = self.act(exogenous_reward, step=step) # Pass step
        self.update_modulation(exogenous_reward, modulation, step=step)  # learning still happens now

        # Store for future application
        # --- FIX: Store ONLY the modulation, not the full tuple ---
        self.pending_modulations.append(modulation)
        # --- End Fix ---

        # If enough time has passed, apply oldest modulation
        if step is not None and step >= self.lag:
            # --- FIX: Pop the modulation calculated at T-lag ---
            old_mod = self.pending_modulations.popleft()
            # --- FIX: Apply OLD modulation to CURRENT reward ---
            # This is the correct behavior for a lagged system
            modulated_reward = exogenous_reward - old_mod
            apply_step = step - self.lag
            # --- End Fix ---
        else:
            apply_step, old_mod = step, 0
            modulated_reward = exogenous_reward # No modulation applied yet

        # Record history
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "applied_step": apply_step, # Step when modulation was calculated
            "modulation": old_mod, # The modulation value applied
            "modulated_reward": modulated_reward, # The final output
            "setpoint": self.setpoint
        })

        return modulated_reward

    def update_modulation(self, exogenous_reward, modulation, step=None):
        # --- FIX: Use discretized key ---
        key = self._get_state_key(exogenous_reward)
        # --- End Fix ---
        
        action_idx = np.where(self.moves == modulation)[0][0]
        
        # Internal reward is based on the immediate (non-lagged) effect
        current_modulated_reward = exogenous_reward - modulation
        reward = -abs(current_modulated_reward - self.setpoint)
        
        self.update(key, action_idx, reward, step=step) # Pass step
        return current_modulated_reward # Return what was used for learning

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
        # This is R(T) - M(T-lag)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (R[T] - M[T-lag])", color="green")
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        # This is M(T-lag)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (M[T-lag])", color="purple")
        plt.xlabel("Step")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def step(self,*_):
        pass

# --- "Time Difference" Modulator with Memory (Fixed) ---
class TD_HomeostaticModulator(BaseQLearningAgent):
    """
    Implements a homeostatic controller with "episodic memory" using a Q-table.
    FIX: State is a *discretized* history of the last `history_length` exogenous rewards.
    """

    def __init__(self, setpoint=0, lag=0, history_length=3, 
                 n_bins=10, reward_range=(-100, 100), **kwargs):
        self.moves = np.arange(-50, 50)
        self.setpoint = setpoint
        self.modulation_history = []
        self.lag = lag
        self.history_length = history_length
        self.pending_modulations = deque(maxlen=lag + 1)
        
        # --- FIX: Discretization ---
        self.n_bins = n_bins
        self.reward_range = reward_range
        self.bins = np.linspace(self.reward_range[0], self.reward_range[1], self.n_bins + 1)
        self.state_history = deque(maxlen=self.history_length)
        # Initialize state history with the bin for the setpoint
        setpoint_bin = self._discretize_reward(self.setpoint)
        for _ in range(self.history_length):
            self.state_history.append(setpoint_bin)
        # --- End Fix ---

        super().__init__(n_actions=len(self.moves), **kwargs)

    # --- FIX: New discretization helper ---
    def _discretize_reward(self, exogenous_reward):
        """Discretizes a single continuous reward into a bin index."""
        clipped_reward = np.clip(exogenous_reward, self.reward_range[0], self.reward_range[1])
        bin_index = np.digitize(clipped_reward, self.bins) - 1
        return max(0, min(bin_index, self.n_bins - 1)) # Ensure it's within [0, n_bins-1]
    # --- End Fix ---

    def _get_state_key(self):
        # The state key is now a tuple of bin indices
        return tuple(self.state_history)

    def act(self, state_key, step=0): # Added step
        action_idx = self.choose_action(state_key, step=step) # Pass step
        return self.moves[action_idx]

    def modify_reward(self, exogenous_reward, step=None, *_):
        # --- FIX: Discretize the new reward before adding to history ---
        discretized_reward = self._discretize_reward(exogenous_reward)
        self.state_history.append(discretized_reward)
        # --- End Fix ---
        
        key = self._get_state_key()
        modulation = self.act(key, step=step) # Pass step
        self.update_modulation(key, exogenous_reward, modulation, step=step) # Pass step
        
        # --- FIX: Store ONLY the modulation ---
        self.pending_modulations.append(modulation)
        # --- End Fix ---

        if step is not None and step >= self.lag:
            # --- FIX: Pop old modulation, apply to current reward ---
            old_mod = self.pending_modulations.popleft()
            modulated_reward = exogenous_reward - old_mod
            apply_step = step - self.lag
            # --- End Fix ---
            
            # This is R(T) - M(T-lag)
            current_modulated = exogenous_reward - old_mod 
            # This is R(T-lag) - M(T-lag) -> what the agent *tried* to do
            # We can't calculate this easily anymore without storing old_exo,
            # so we'll simplify the plot.
            
        else:
            apply_step, old_mod = step, 0
            modulated_reward = exogenous_reward
            current_modulated = exogenous_reward

        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "applied_step": apply_step,
            "modulation": old_mod,
            "modulated_reward": modulated_reward, # This is R(T) - M(T-lag)
            "current_modulated_reward": current_modulated, # Same as above now
            "setpoint": self.setpoint
        })
        return modulated_reward

    def update_modulation(self, state_key, exogenous_reward, modulation, step=None):
        action_idx = np.where(self.moves == modulation)[0][0]
        # Internal reward is based on the immediate (non-lagged) effect
        current_modulated_reward = exogenous_reward - modulation
        reward = -abs(current_modulated_reward - self.setpoint)
        self.update(state_key, action_idx, reward, step=step) # Pass step
        return current_modulated_reward

    def plot_modulation_trajectory(self, max_points=1000):
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (R[T])", color="blue", alpha=0.7)
        # --- FIX: Simplified plot label ---
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (R[T] - M[T-lag])", color="green", linestyle='-')
        # --- End Fix ---
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title("Homeostatic Control Trajectory (TD-Tabular)")
        
        plt.subplot(2, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (M[T-lag])", color="purple")
        plt.xlabel("Step")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def step(self, *_):
        pass



# --- Replay Memory for DQN ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Base TD (Q-table) DHR Controller ---
class TD_DHR:
    def __init__(self, setpoint, lag=0, history_length=5):
        self.setpoint = setpoint
        self.lag = lag
        self.history_length = history_length
        self.history = deque([0] * history_length, maxlen=history_length)
        self.modulation_history = []
        
        # --- EXPLANATION: n_actions ---
        # The agent's "action" is *which index* (from 0 to history_length-1)
        # to pick from the `self.history` deque.
        # It is learning: "In this state, which of my 5 past rewards
        # is the best one to use as a modulation signal?"
        self.agent = BaseQLearningAgent(
            n_actions=self.history_length, # So, n_actions IS history_length
            learning_rate=0.1,
            gamma=0.9,
            exploration_decay=0.9999,
            min_exploration_rate=0.01
        )
        self.step = 0

    def _get_state(self):
        """ The "physiological" state is the mean of the recent reward history. """
        if not self.history:
            return 0
        return np.mean(self.history)

    def modify_reward(self, exogenous_reward, step):
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()
        state_key = int(H_T) # Discretize state for Q-table

        # 2. Homeostatic controller chooses an action (which past R to use for modulation)
        # The action is an *index* into the history, e.g., action = 2
        action = self.agent.choose_action(state_key, step)
        
        # 3. Get modulation signal (M[T-lag])
        # We apply the action `k` by using it as an index on our history deque.
        # e.g., if action = 2, M_T_minus_lag = self.history[2] (the 3rd oldest reward)
        M_T_minus_lag = self.history[action] 

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        next_state_key = int(H_T_plus_1)
        
        # 7. Calculate intrinsic reward for the *controller*
        # The agent is rewarded based on how close its *output*
        # (the modulated_reward) is to the setpoint.
        intrinsic_reward = -abs(modulated_reward - self.setpoint)


        # 8. Update the homeostatic agent's Q-table
        # --- EXPLANATION: Online Learning ---
        # This is a classic "online" TD update. The agent learns from the
        # (S, A, R, S') tuple *immediately* and then discards it.
        # It does NOT use a replay memory.
        self.agent.update(state_key, action, intrinsic_reward, next_state_key)

        # 9. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.agent.exploration_rate
        })
        
        return modulated_reward

    def update_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint
        
    def plot_modulation_trajectory(self, max_points=1000):
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1) # Changed to 3 plots
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (R[T])", color="blue", alpha=0.7)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (R'[T])", color="green", linestyle='-')
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title(f"Homeostatic Control Trajectory ({type(self).__name__})")

        plt.subplot(3, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (M[T-lag])", color="purple")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()

        # plt.subplot(3, 1, 3) # New plot for Epsilon
        # if 'epsilon' in df.columns:
        #     plt.plot(df["step"], df["epsilon"], label="Epsilon", color="orange")
        # plt.ylabel("Exploration Rate")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        plt.tight_layout()
        plt.show()

# --- DQN (Deep Q-Network) DHR Controller ---

class DQN(nn.Module):
    """Simple Feed-Forward Network"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        # --- EXPLANATION: Output Layer ---
        # The output layer has `action_size` nodes.
        # Each node corresponds to the Q-value for one of the
        # discrete "index" actions (e.g., Q(S, action=0), Q(S, action=1), ...)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_DHR(TD_DHR):
    def __init__(self, setpoint, lag=0, history_length=5, 
                 gamma=0.9, batch_size=128, replay_memory=10000, 
                 target_update=10, lr=1e-3):
        
        # Don't call super().__init__() for 'agent', we replace it
        self.setpoint = setpoint
        self.lag = lag
        self.history_length = history_length
        self.history = deque([0] * history_length, maxlen=history_length)
        self.modulation_history = []
        self.step = 0
        
        self.state_size = 1  # State is just the mean H_T
        # --- EXPLANATION: n_actions ---
        # Same as TD_DHR: The action is *which index* (0 to history_length-1)
        # to pick from the `self.history` deque.
        self.action_size = history_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.exploration_rate = 1.0
        self.exploration_decay = 0.9999
        self.min_exploration_rate = 0.01

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory)

    def choose_action(self, H_T, step):
        """ Choose action using DQN policy net and e-greedy """
        
        if step > 0: 
            self.exploration_rate = max(self.min_exploration_rate, 
                                        self.exploration_rate * self.exploration_decay)
        epsilon = self.exploration_rate if step > 0 else 0.0 

        if np.random.rand() < epsilon:
            # Exploration: Pick a random *index*
            return torch.tensor([[random.randrange(self.action_size)]], 
                                device=self.device, dtype=torch.long)
        else:
            # Exploitation:
            with torch.no_grad():
                state_tensor = torch.tensor([H_T], device=self.device, dtype=torch.float32).unsqueeze(0)
                # 1. policy_net(state_tensor) -> [Q(S,a=0), Q(S,a=1), ..., Q(S,a=4)]
                # 2. .max(1)[1] -> gets the *index* (the action) of the highest Q-value
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)

    def modify_reward(self, exogenous_reward, step):
        """ This method overrides the parent TD_DHR.modify_reward """
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()

        # 2. Homeostatic controller chooses an action (tensor)
        # The action_tensor is a 2D tensor, e.g., tensor([[2]])
        action_tensor = self.choose_action(H_T, step)
        action = action_tensor.item() # Convert to scalar index, e.g., 2
        
        # 3. Get modulation signal (M[T-lag])
        # We apply the action `k` by using it as an index on our history deque.
        # e.g., if action = 2, M_T_minus_lag = self.history[2]
        M_T_minus_lag = self.history[action] 

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        
        # 7. Calculate intrinsic reward for the *controller*
        intrinsic_reward = -abs(modulated_reward - self.setpoint)
        reward_tensor = torch.tensor([intrinsic_reward], device=self.device)
        
        # 8. Store the transition in memory
        # --- EXPLANATION: Replay Memory ---
        # Unlike TD_DHR, we do NOT learn yet. We store the full
        # (S, A, R, S') tuple (as tensors) in the ReplayMemory.
        # The actual learning happens in self.optimize_model() on batches.
        state_tensor = torch.tensor([H_T], device=self.device, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor([H_T_plus_1], device=self.device, dtype=torch.float32).unsqueeze(0)
        
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # 9. Perform one step of the optimization (on the policy network)
        if step > 0: # Only optimize during actual steps, not pre-training
            self.optimize_model()

        # 10. Update target network
        if step > 0 and step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 11. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.exploration_rate if step > 0 else 0
        })
        
        return modulated_reward

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory yet
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(S_t, A_t)
        # .gather(1, action_batch) picks the Q-value for the action we *took*
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(S_{t+1}) for all next states.
        # Use target_net for stability
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # .max(1)[0] gets the *value* of the best action in the next state
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values (The "ground truth" from Bellman)
        # R + gamma * V(S_{t+1})
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

# ---
# --- DYNAMIC SETPOINT (ALLOSTASIS) CONTROLLERS ---
# ---

class TD_DHR_D(TD_DHR):
    """
    A TD_DHR controller with a Dynamic Setpoint (Allostasis).
    The setpoint itself moves if the exogenous reward consistently
    breaches a top or bottom threshold.
    """
    def __init__(self, setpoint, lag=0, history_length=5,
                 top_threshold=50, bottom_threshold=-70, adjustment_factor=0.05):
        
        # Call parent __init__ with the initial setpoint
        super().__init__(setpoint=setpoint, lag=lag, history_length=history_length)
        
        # Dynamic setpoint parameters
        self.current_setpoint = setpoint
        self.top_threshold = top_threshold
        self.bottom_threshold = bottom_threshold
        self.adjustment_factor = adjustment_factor # How fast the setpoint moves

    def _update_dynamic_setpoint(self, exogenous_reward):
            """
            Adjusts the setpoint based on breaches of the thresholds.
            This is a form of allostasis.
            """
            setpoint_changed = False
            top_distance = exogenous_reward - self.top_threshold
            if top_distance > 0:
                # Reward is too high, pull the setpoint down
                decrease_amount = top_distance * self.adjustment_factor
                self.current_setpoint -= decrease_amount
                setpoint_changed = True # Mark that a change happened
                
            bottom_distance = self.bottom_threshold - exogenous_reward
            if bottom_distance > 0:
                # Reward is too low, push the setpoint up
                increase_amount = bottom_distance * self.adjustment_factor
                self.current_setpoint += increase_amount
                setpoint_changed = True # Mark that a change happened
            
            # --- *** THE FIX *** ---
            if setpoint_changed:
                # The goal has moved! The agent's old policy is now stale.
                # We must force the agent to re-explore to find the new
                # optimal policy for the *new* setpoint.
                # We bump epsilon to a "re-explore" value (e.g., 0.5)
                # or its current value, whichever is higher.
                new_epsilon = 1 #max(self.agent.exploration_rate, 0.5) 
                self.agent.exploration_rate = new_epsilon
                
                # Update the parent's static setpoint for plotting consistency
                self.setpoint = self.current_setpoint
            # --- *** END FIX *** ---

    def modify_reward(self, exogenous_reward, step):
        """
        Overrides the parent method to include dynamic setpoint logic.
        """
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()
        state_key = int(H_T) # Discretize state for Q-table

        # 2. Homeostatic controller chooses an action
        action = self.agent.choose_action(state_key, step)
        
        # 3. Get modulation signal (M[T-lag])
        M_T_minus_lag = self.history[action] 

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        next_state_key = int(H_T_plus_1)
        
        # --- NEW DYNAMIC LOGIC ---
        # 7. Update the setpoint itself based on the exogenous reward
        self._update_dynamic_setpoint(exogenous_reward)
        # --- END NEW LOGIC ---

        # 8. Calculate intrinsic reward for the *controller*
        #    This now uses the *current* dynamic setpoint
        intrinsic_reward = -abs(modulated_reward - self.setpoint)

        # 9. Update the homeostatic agent's Q-table
        self.agent.update(state_key, action, intrinsic_reward, next_state_key)

        # 10. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.agent.exploration_rate,
            "current_setpoint": self.current_setpoint # Track the setpoint
        })
        
        return modulated_reward

    def plot_modulation_trajectory(self, max_points=1000):
        """
        Overrides the parent plot to show the dynamic setpoint.
        """
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (R[T])", color="blue", alpha=0.7)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (R'[T])", color="green", linestyle='-')
        
        # Plot the dynamic setpoint
        if "current_setpoint" in df.columns:
             plt.plot(df["step"], df["current_setpoint"], label="Dynamic Setpoint", color="red", linestyle="--")
        else:
             plt.axhline(self.setpoint, color="gray", linestyle="--", label="Static Setpoint")

        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title(f"Homeostatic Control Trajectory ({type(self).__name__})")

        plt.subplot(3, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (M[T-lag])", color="purple")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()

        # plt.subplot(3, 1, 3)
        # if 'epsilon' in df.columns:
        #     plt.plot(df["step"], df["epsilon"], label="Epsilon", color="orange")
        # plt.ylabel("Exploration Rate")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        plt.tight_layout()
        plt.show()


class DQN_DHR_D(DQN_DHR):
    """
    A DQN_DHR controller with a Dynamic Setpoint (Allostasis).
    """
    def __init__(self, setpoint, lag=0, history_length=5, 
                 gamma=0.9, batch_size=128, replay_memory=10000, 
                 target_update=10, lr=1e-3,
                 top_threshold=50, bottom_threshold=-70, adjustment_factor=0.05):
        
        # Call parent __init__
        super().__init__(setpoint=setpoint, lag=lag, history_length=history_length,
                         gamma=gamma, batch_size=batch_size, replay_memory=replay_memory,
                         target_update=target_update, lr=lr)
        
        # Dynamic setpoint parameters
        self.current_setpoint = setpoint
        self.top_threshold = top_threshold
        self.bottom_threshold = bottom_threshold
        self.adjustment_factor = adjustment_factor

    def _update_dynamic_setpoint(self, exogenous_reward):
            """
            Adjusts the setpoint based on breaches of the thresholds.
            This is a form of allostasis.
            """
            setpoint_changed = False
            top_distance = exogenous_reward - self.top_threshold
            if top_distance > 0:
                # Reward is too high, pull the setpoint down
                decrease_amount = top_distance * self.adjustment_factor
                self.current_setpoint -= decrease_amount
                setpoint_changed = True # Mark that a change happened
                
            bottom_distance = self.bottom_threshold - exogenous_reward
            if bottom_distance > 0:
                # Reward is too low, push the setpoint up
                increase_amount = bottom_distance * self.adjustment_factor
                self.current_setpoint += increase_amount
                setpoint_changed = True # Mark that a change happened
            
            # --- *** THE FIX *** ---
            if setpoint_changed:
                # The goal has moved! The agent's old policy is now stale.
                # We must force the agent to re-explore to find the new
                # optimal policy for the *new* setpoint.
                # We bump epsilon to a "re-explore" value (e.g., 0.5)
                # or its current value, whichever is higher.
                new_epsilon = 1 # max(self.exploration_rate, 0.5) 
                self.exploration_rate = new_epsilon
                
                # Update the parent's static setpoint for plotting consistency
                self.setpoint = self.current_setpoint
            # --- *** END FIX *** ---

    def modify_reward(self, exogenous_reward, step):
        """
        This method overrides the parent DQN_DHR.modify_reward
        """
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()

        # 2. Homeostatic controller chooses an action (tensor)
        action_tensor = self.choose_action(H_T, step)
        action = action_tensor.item() # Convert to scalar
        
        # 3. Get modulation signal (M[T-lag])
        M_T_minus_lag = self.history[action] 

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        
        # --- NEW DYNAMIC LOGIC ---
        # 7. Update the setpoint itself based on the exogenous reward
        self._update_dynamic_setpoint(exogenous_reward)
        # --- END NEW LOGIC ---

        # 8. Calculate intrinsic reward for the *controller*
        #    This now uses the *current* dynamic setpoint
        intrinsic_reward = -abs(modulated_reward - self.current_setpoint)
        reward_tensor = torch.tensor([intrinsic_reward], device=self.device)
        
        # 9. Store the transition in memory
        state_tensor = torch.tensor([H_T], device=self.device, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor([H_T_plus_1], device=self.device, dtype=torch.float32).unsqueeze(0)
        
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # 10. Perform one step of the optimization (on the policy network)
        if step > 0:
            self.optimize_model()

        # 11. Update target network
        if step > 0 and step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 12. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.exploration_rate if step > 0 else 0,
            "current_setpoint": self.current_setpoint # Track the setpoint
        })
        
        return modulated_reward

    # We also need to override the plot function for the DQN-D class
    def plot_modulation_trajectory(self, max_points=1000):
        """
        Overrides the parent plot to show the dynamic setpoint.
        (This is a copy of the TD_DHR_D plot method)
        """
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (R[T])", color="blue", alpha=0.7)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (R'[T])", color="green", linestyle='-')
        
        if "current_setpoint" in df.columns:
             plt.plot(df["step"], df["current_setpoint"], label="Dynamic Setpoint", color="red", linestyle="--")
        else:
             plt.axhline(self.setpoint, color="gray", linestyle="--", label="Static Setpoint")

        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title(f"Homeostatic Control Trajectory ({type(self).__name__})")

        plt.subplot(3, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (M[T-lag])", color="purple")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()

        # plt.subplot(3, 1, 3)
        # if 'epsilon' in df.columns:
        #     plt.plot(df["step"], df["epsilon"], label="Epsilon", color="orange")
        # plt.ylabel("Exploration Rate")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        plt.tight_layout()
        plt.show()
        
class DQN_DHR_E(DQN_DHR):
    """
    An "Expanded" DQN_DHR controller.
    Its action space is expanded to include two new actions:
    k:   min(history) - expansion_amount
    k+1: max(history) + expansion_amount
    """
    def __init__(self, setpoint, lag=0, history_length=5, 
                 gamma=0.9, batch_size=128, replay_memory=10000, 
                 target_update=10, lr=1e-3, expansion_amount=10.0):
        
        # Manually initialize all parameters from parent
        self.setpoint = setpoint
        self.lag = lag
        self.history_length = history_length
        self.history = deque([0] * history_length, maxlen=history_length)
        self.modulation_history = []
        self.step = 0
        
        self.state_size = 1  # State is just the mean H_T
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.exploration_rate = 1.0
        self.exploration_decay = 0.9999
        self.min_exploration_rate = 0.01
        
        self.expansion_amount = expansion_amount
        
        # --- KEY CHANGE ---
        self.action_size = self.history_length + 2
        # --- END KEY CHANGE ---

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Re-initialize networks with the new action_size
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_memory)

    def modify_reward(self, exogenous_reward, step):
        """
        This method overrides the parent DQN_DHR.modify_reward
        """
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()

        # 2. Homeostatic controller chooses an action (tensor)
        # The action_tensor is a 2D tensor, e.g., tensor([[6]])
        action_tensor = self.choose_action(H_T, step)
        action = action_tensor.item() # Convert to scalar index, e.g., 6
        
        # --- KEY CHANGE: Map action to modulation signal ---
        if 0 <= action < self.history_length:
            # Standard history action
            M_T_minus_lag = self.history[action]
        elif action == self.history_length:
            # New "smaller value" action
            M_T_minus_lag = np.min(self.history) - self.expansion_amount
        elif action == self.history_length + 1:
            # New "greater value" action
            M_T_minus_lag = np.max(self.history) + self.expansion_amount
        else:
            # Fallback (should not happen)
            M_T_minus_lag = self.history[0]
        # --- END KEY CHANGE ---

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        
        # 7. Calculate intrinsic reward for the *controller*
        intrinsic_reward = -abs(modulated_reward - self.setpoint)
        reward_tensor = torch.tensor([intrinsic_reward], device=self.device)
        
        # 8. Store the transition in memory
        state_tensor = torch.tensor([H_T], device=self.device, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor([H_T_plus_1], device=self.device, dtype=torch.float32).unsqueeze(0)
        
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # 9. Perform one step of the optimization (on the policy network)
        if step > 0: 
            self.optimize_model()

        # 10. Update target network
        if step > 0 and step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 11. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.exploration_rate if step > 0 else 0
        })
        
        return modulated_reward

class TD_DHR_E(TD_DHR):
    """
    An "Expanded" TD_DHR controller.
    Its action space is not just indexes into history (0..k-1),
    but is expanded to include two new actions:
    k:   min(history) - expansion_amount
    k+1: max(history) + expansion_amount
    
    This allows the agent to "invent" a modulation signal
    that is not in its recent history.
    """
    def __init__(self, setpoint, lag=0, history_length=5, expansion_amount=10.0):
        
        # Manually initialize key parts from parent
        self.setpoint = setpoint
        self.lag = lag
        self.history_length = history_length
        self.history = deque([0] * history_length, maxlen=history_length)
        self.modulation_history = []
        self.step = 0
        
        self.expansion_amount = expansion_amount
        
        # --- KEY CHANGE ---
        # The action space is now history_length + 2
        new_n_actions = self.history_length + 2
        # --- END KEY CHANGE ---

        self.agent = BaseQLearningAgent(
            n_actions=new_n_actions, # Pass new action space size
            learning_rate=0.1,
            gamma=0.9,
            exploration_decay=0.9999,
            min_exploration_rate=0.01
        )

    def modify_reward(self, exogenous_reward, step):
        """
        Overrides the parent method to include expanded action logic.
        """
        self.step += 1
        
        # 1. Get current physiological state (H_T)
        H_T = self._get_state()
        state_key = int(H_T) # Discretize state for Q-table

        # 2. Homeostatic controller chooses an action
        # Action is now an index from 0 to history_length + 1
        action = self.agent.choose_action(state_key, step)
        
        # --- KEY CHANGE: Map action to modulation signal ---
        if 0 <= action < self.history_length:
            # Standard history action
            M_T_minus_lag = self.history[action]
        elif action == self.history_length:
            # New "smaller value" action
            M_T_minus_lag = np.min(self.history) - self.expansion_amount
        elif action == self.history_length + 1:
            # New "greater value" action
            M_T_minus_lag = np.max(self.history) + self.expansion_amount
        else:
            # Fallback (should not happen)
            M_T_minus_lag = self.history[0]
        # --- END KEY CHANGE ---

        # 4. Calculate modulated reward (R'[T] = R[T] - M[T-lag])
        modulated_reward = exogenous_reward - M_T_minus_lag

        # 5. Update history with the *new* exogenous reward
        self.history.appendleft(exogenous_reward)

        # 6. Get the *next* physiological state (H[T+1])
        H_T_plus_1 = self._get_state()
        next_state_key = int(H_T_plus_1)
        
        # 7. Calculate intrinsic reward for the *controller*
        intrinsic_reward = -abs(modulated_reward - self.setpoint)

        # 8. Update the homeostatic agent's Q-table
        self.agent.update(state_key, action, intrinsic_reward, next_state_key)

        # 9. Store history for plotting
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "modulated_reward": modulated_reward,
            "modulation": M_T_minus_lag,
            "state_H_T": H_T,
            "next_state_H_T+1": H_T_plus_1,
            "intrinsic_reward": intrinsic_reward,
            "action": action,
            "epsilon": self.agent.exploration_rate
        })
        
        return modulated_reward
