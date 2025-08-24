from imports import *


class BaseQLearningAgent:
    def __init__(
        self,
        n_actions,
        exploration_rate=1.0,
        exploration_decay=0.999,
        min_exploration_rate=0.001,
        strategy="egreedy",
    ):
        self.n_actions = n_actions
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.strategy = strategy
        self.q_table = {}  # key: context or (context, recommendation)
        self.action_counts = {}  # same key
        self.time = 0

    def _ensure_key(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
            self.action_counts[key] = np.zeros(self.n_actions)

    def choose_action(self, key):
        self._ensure_key(key)
        self.time += 1

        if self.strategy == "egreedy":
            return self._egreedy_choice(key)
        elif self.strategy == "ucb":
            return self._ucb_choice(key)
        elif self.strategy == "softmax":
            return self._softmax_choice(key)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _egreedy_choice(self, key):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[key])

    def _ucb_choice(self, key):
        counts = self.action_counts[key] + 1e-5
        return np.argmax(
            self.q_table[key]
            + self.exploration_rate * np.sqrt(np.log(self.time + 1) / counts)
        )

    def _softmax_choice(self, key, tau=1.0):
        scaled = self.q_table[key] / tau
        probs = softmax(scaled)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, key, action, reward):
        self._ensure_key(key)
        n = self.action_counts[key][action]
        q = self.q_table[key][action]

        if n == 0:
            self.q_table[key][action] = reward
        else:
            self.q_table[key][action] = q + (reward - q) / (n + 1)

        self.action_counts[key][action] += 1

        if self.strategy == "egreedy":
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )


class RecommenderAgent(BaseQLearningAgent):
    def __init__(self, num_recommendations, **kwargs):
        super().__init__(n_actions=num_recommendations, **kwargs)

    def act(self, context):
        return self.choose_action(context)

    def update_reward(self, context, recommendation, reward):
        # Treat context as the key
        self.update(context, recommendation, reward)

    def visualize_q_landscape(
        self, context_list, title="Recommender Agent's Q-value Landscape"
    ):
        n_contexts = len(context_list)
        q_matrix = np.zeros((self.n_actions, n_contexts))

        for col_idx, context in enumerate(context_list):
            q_matrix[:, col_idx] = self.q_table.get(
                context, np.full(self.n_actions, np.nan)
            )

        plt.figure(figsize=(6, 4))
        im = plt.imshow(
            q_matrix,
            cmap="plasma",
            origin="lower",
            aspect="auto",
            extent=[0, n_contexts, 0, self.n_actions],
        )
        plt.colorbar(im, label="Estimated Q-value")
        plt.title(title)
        plt.xlabel("Contexts (X-axis)")
        plt.ylabel("Recommendations (Y-axis)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.show()


class RecommendedAgent(BaseQLearningAgent):
    def __init__(self, **kwargs):
        super().__init__(n_actions=2, **kwargs)

    def act(self, context, recommendation):
        key = (context, recommendation)
        action = self.choose_action(key)
        return action == 0  # Accept if 0

    def update_reward(self, context, recommendation, accepted, reward):
        key = (context, recommendation)
        action = 0 if accepted else 1
        self.update(key, action, reward)

    def visualize_accept_q_landscape(
        self,
        context_list,
        recommendation_list,
        title="User Learned Q-values for Accept",
    ):
        n_contexts = len(context_list)
        n_recommendations = len(recommendation_list)
        q_matrix = np.full((n_recommendations, n_contexts), np.nan)

        for i, rec in enumerate(recommendation_list):
            for j, ctx in enumerate(context_list):
                key = (ctx, rec)
                if key in self.q_table:
                    q_matrix[i, j] = self.q_table[key][0]

        plt.figure(figsize=(6, 4))
        im = plt.imshow(
            q_matrix,
            cmap="viridis",
            origin="lower",
            aspect="auto",
            extent=[0, n_contexts, 0, n_recommendations],
        )
        plt.colorbar(im, label="Q-value for Accept (Action=0)")
        plt.title(title)
        plt.xlabel("Contexts (X-axis)")
        plt.ylabel("Recommendations (Y-axis)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.show()
