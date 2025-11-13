from imports import *
from agents import BaseQLearningAgent
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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



class HomeostaticModulator(BaseQLearningAgent):
    def __init__(self, setpoint=0, lag=0, **kwargs):
        self.moves = np.arange(-50, 50)
        self.setpoint = setpoint
        self.modulation_history = []
        self.lag = lag
        self.pending_modulations = deque(maxlen=lag + 1)  # FIFO queue
        super().__init__(n_actions=len(self.moves), **kwargs)

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

# --- "Time Difference" Modulator with Memory (Unchanged) ---
class TD_HomeostaticModulator(BaseQLearningAgent):
    """
    Implements a homeostatic controller with "episodic memory" using a Q-table.
    Its state is a *history* of the last `history_length` exogenous rewards.
    """

    def __init__(self, setpoint=0, lag=0, history_length=3, **kwargs):
        self.moves = np.arange(-50, 50)
        self.setpoint = setpoint
        self.modulation_history = []
        self.lag = lag
        self.history_length = history_length
        self.pending_modulations = deque(maxlen=lag + 1)
        self.state_history = deque(maxlen=self.history_length)

        for _ in range(self.history_length):
            self.state_history.append(self.setpoint)

        super().__init__(n_actions=len(self.moves), **kwargs)

    def _get_state_key(self):
        return tuple(round(r, 1) for r in self.state_history)

    def act(self, state_key):
        action_idx = self.choose_action(state_key)
        return self.moves[action_idx]

    def modify_reward(self, exogenous_reward, step=None, *_):
        self.state_history.append(exogenous_reward)
        key = self._get_state_key()
        modulation = self.act(key)
        self.update_modulation(key, exogenous_reward, modulation)
        self.pending_modulations.append((step, exogenous_reward, modulation))

        if step is not None and step >= self.lag:
            apply_step, old_exo, old_mod = self.pending_modulations.popleft()
            modulated_reward = old_exo - old_mod
            current_modulated = exogenous_reward - old_mod
        else:
            apply_step, old_exo, old_mod = step, exogenous_reward, 0
            modulated_reward = exogenous_reward
            current_modulated = exogenous_reward

        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "applied_step": apply_step,
            "modulation": old_mod,
            "modulated_reward": modulated_reward,
            "current_modulated_reward": current_modulated,
            "setpoint": self.setpoint
        })
        return modulated_reward

    def update_modulation(self, state_key, exogenous_reward, modulation):
        action_idx = np.where(self.moves == modulation)[0][0]
        modulated_reward = exogenous_reward - modulation
        reward = -abs(modulated_reward - self.setpoint)
        self.update(state_key, action_idx, reward)
        return modulated_reward

    def plot_modulation_trajectory(self, max_points=1000):
        # (Plotting code omitted for brevity, but it's identical to the file)
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (at T)", color="blue", alpha=0.7)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (Official, from T-lag)", color="green", linestyle='-')
        plt.plot(df["step"], df["current_modulated_reward"], label="Modulated Reward (Current_Exo - Old_Mod)", color="red", linestyle=':', alpha=0.8)
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title("Homeostatic Control Trajectory")
        plt.subplot(2, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (from T-lag)", color="purple")
        plt.xlabel("Step")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def step(self, *_):
        pass


# --- NEW Deep Q-Network Components ---

class ReplayBuffer:
    """A simple FIFO experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """Samples a batch of transitions."""
        states, actions, rewards, next_states = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(states),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(-1),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
                torch.stack(next_states))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """The neural network that approximates the Q-function."""
    def __init__(self, input_dim, output_dim, network_type='gru', hidden_dim=64):
        super(QNetwork, self).__init__()
        self.network_type = network_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if network_type == 'gru':
            # GRU expects input shape (batch_size, seq_len, input_size)
            # Our state is (batch_size, history_length), so input_size=1
            self.gru = nn.GRU(1, hidden_dim, batch_first=True)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        elif network_type == 'mlp':
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, state):
        if self.network_type == 'gru':
            # Reshape state from (batch_size, seq_len) to (batch_size, seq_len, 1)
            state_gru = state.unsqueeze(-1)
            gru_out, hidden = self.gru(state_gru)
            # We use the final hidden state
            x = F.relu(self.fc1(hidden.squeeze(0)))
            q_values = self.fc2(x)
        else: # mlp
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
        return q_values

# --- NEW "Temporal Difference Deep Homeostatic Regulator" ---
class TD_DHR:
    """
    Implements a Deep Q-Network (DQN) homeostatic controller.

    This agent uses a neural network (MLP or GRU) to generalize
    from a history of states, overcoming the "curse of dimensionality"
    of the tabular TD_HomeostaticModulator.
    """
    def __init__(self, setpoint=0, lag=0, history_length=5,
                 network_type='gru', hidden_dim=64, learning_rate=1e-3,
                 replay_buffer_size=10000, batch_size=64, gamma=0.99,
                 tau=1e-3, epsilon_start=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01):

        self.moves = np.arange(-50, 50)
        self.n_actions = len(self.moves)
        self.setpoint = setpoint
        self.lag = lag
        self.history_length = history_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Use GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Networks
        input_dim = self.history_length
        self.policy_net = QNetwork(input_dim, self.n_actions, network_type, hidden_dim).to(self.device)
        self.target_net = QNetwork(input_dim, self.n_actions, network_type, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss() # Huber Loss, robust to outliers

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Initialize state and action histories
        self.pending_modulations = deque(maxlen=lag + 1)
        self.state_history = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.state_history.append(self.setpoint)

        self.modulation_history = []
        self.step_count = 0

    def _get_state_tensor(self, history_deque):
        """Converts a state history deque to a PyTorch FloatTensor."""
        return torch.tensor(history_deque, dtype=torch.float32).to(self.device)

    def choose_action(self, state_tensor):
        """Chooses an action using an epsilon-greedy policy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                # Add a batch dimension (B, L) -> (1, B, L)
                state_tensor = state_tensor.unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()

        return action_idx

    def _learn(self):
        """Trains the policy network using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences to learn

        # Sample a batch
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        # Move batch to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # --- Compute Q(s, a) ---
        # Get Q-values from the policy network for all actions
        q_values = self.policy_net(states)
        # Gather the Q-value for the specific action that was taken
        current_q_values = q_values.gather(1, actions)

        # --- Compute V(s') ---
        with torch.no_grad():
            # Get the *best* Q-value for the next state from the *target* network
            next_q_max = self.target_net(next_states).max(1)[0].unsqueeze(-1)
            # Compute the target Q-value: r + gamma * max(Q(s', a'))
            target_q_values = rewards + (self.gamma * next_q_max)

        # --- Compute Loss ---
        loss = self.loss_function(current_q_values, target_q_values)

        # --- Perform Gradient Descent ---
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_net(self):
        """Performs a soft update (Polyak averaging) of the target network."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def modify_reward(self, exogenous_reward, step=None, *_):
        """Main control loop for the DQN agent."""
        self.step_count = step if step is not None else self.step_count + 1

        # --- 0.5: Update State ---
        # Get state *before* adding new reward
        old_state_tensor = self._get_state_tensor(self.state_history)
        self.state_history.append(exogenous_reward)
        # Get state *after* adding new reward
        new_state_tensor = self._get_state_tensor(self.state_history)

        # --- 1: Choose Action & Get Hypothetical Reward ---
        action_idx = self.choose_action(new_state_tensor)
        modulation = self.moves[action_idx]

        # Calculate the *internal* reward for this hypothetical action
        hypothetical_reward = -abs(exogenous_reward - modulation - self.setpoint)

        # --- 1.5: Store Experience ---
        # Store the (s, a, r, s') transition in the replay buffer
        self.replay_buffer.push(old_state_tensor, action_idx, hypothetical_reward, new_state_tensor)

        # --- 1.7: Learn ---
        # Perform one learning step
        self._learn()

        # Periodically update the target network
        if self.step_count % 10 == 0: # Update target net every 10 steps
            self.update_target_net()

        # --- 2. STORE FOR LATER (Time T) ---
        # This lag logic is identical to the other agents
        self.pending_modulations.append((step, exogenous_reward, modulation))

        # --- 3. APPLY FROM PAST (Time T - lag) ---
        if step is not None and step >= self.lag:
            apply_step, old_exo, old_mod = self.pending_modulations.popleft()
            modulated_reward = old_exo - old_mod
            current_modulated = exogenous_reward - old_mod
        else:
            apply_step, old_exo, old_mod = step, exogenous_reward, 0
            modulated_reward = exogenous_reward
            current_modulated = exogenous_reward

        # --- 4. RECORD & RETURN ---
        self.modulation_history.append({
            "step": step,
            "exogenous_reward": exogenous_reward,
            "applied_step": apply_step,
            "modulation": old_mod,
            "modulated_reward": modulated_reward,
            "current_modulated_reward": current_modulated,
            "setpoint": self.setpoint,
            "epsilon": self.epsilon
        })
        return modulated_reward

    def plot_modulation_trajectory(self, max_points=1000):
        # (Plotting code is identical, just adding epsilon)
        if not self.modulation_history:
            print("No modulation history to plot.")
            return
        df = pd.DataFrame(self.modulation_history)
        if max_points and len(df) > max_points:
            df = df.iloc[-max_points:]

        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1) # Changed to 3 plots
        plt.plot(df["step"], df["exogenous_reward"], label="Exogenous Reward (at T)", color="blue", alpha=0.7)
        plt.plot(df["step"], df["modulated_reward"], label="Modulated Reward (Official, from T-lag)", color="green", linestyle='-')
        plt.plot(df["step"], df["current_modulated_reward"], label="Modulated Reward (Current_Exo - Old_Mod)", color="red", linestyle=':', alpha=0.8)
        plt.axhline(self.setpoint, color="gray", linestyle="--", label="Setpoint")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.title("Homeostatic Control Trajectory (DQN Agent)")

        plt.subplot(3, 1, 2)
        plt.plot(df["step"], df["modulation"], label="Applied Modulation (from T-lag)", color="purple")
        plt.ylabel("Modulation")
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3) # New plot for Epsilon
        plt.plot(df["step"], df["epsilon"], label="Epsilon (Exploration Rate)", color="orange")
        plt.xlabel("Step")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def step(self, *_):
        pass

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
