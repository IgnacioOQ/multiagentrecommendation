from imports import *

# Recommender Agent
class RecommenderAgent:
    def __init__(self,
                 num_recommendations,
                 exploration_rate=1,
                 exploration_decay=0.999,
                 min_exploration_rate=0.001,
                 type='egreedy'):
        """
        Agent selects a recommendation given context.
        """
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.n_actions = num_recommendations
        self.q_table = {}  # key: context
        self.action_counts = {}  # key: context
        self.time = 0  # for UCB
        self.type = type

    def act(self, context):
        """
        Decide which recommendation to make for the given context.
        """
        if context not in self.q_table:
            self.q_table[context] = np.zeros(self.n_actions)
            self.action_counts[context] = np.zeros(self.n_actions)

        self.time += 1

        if self.type == 'egreedy':
            action = self.egreedy_choice(context)
        if self.type == 'ucb':
            action = self.ucb_choice(context)
        if self.type == 'softmax':
            action = self.ucb_choice(context)

        self.action_counts[context][action] += 1
        return action

    def egreedy_choice(self, context):
        """
        Epsilon-greedy selection.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[context])

    def ucb_choice(self, context):
        """
        Upper Confidence Bound (UCB) selection.
        """
        total_counts = np.sum(self.action_counts[context]) + 1e-5
        ucb_values = self.q_table[context] + \
                     self.exploration_rate * np.sqrt(np.log(self.time + 1) / (self.action_counts[context] + 1e-5))
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
        
    def update(self, context, recommendation, reward):
        """
        Update Q-value after observing reward.
        """
        if context not in self.q_table:
            self.q_table[context] = np.zeros(self.n_actions)
            self.action_counts[context] = np.zeros(self.n_actions)

        n = self.action_counts[context][recommendation]
        q = self.q_table[context][recommendation]
        self.q_table[context][recommendation] = q + (reward - q) / n

        # Decay exploration for epsilon-greedy
        if not self.type == 'ucb':
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * self.exploration_decay)

    def visualize_q_landscape(self, context_list, title = "Recommender Agent's Q-value Landscape"):
            """
            Visualize the agent's Q-value approximation as a 2D heatmap.

            Parameters:
            - context_list: ordered list of context values (to index q_table columns)
            """
            n_contexts = len(context_list)
            q_matrix = np.zeros((self.n_actions, n_contexts))

            for col_idx, context in enumerate(context_list):
                if context in self.q_table:
                    q_matrix[:, col_idx] = self.q_table[context]
                else:
                    q_matrix[:, col_idx] = np.nan  # or leave as zeros

            plt.figure(figsize=(6, 4))
            im = plt.imshow(q_matrix, cmap="plasma", origin="lower", aspect="auto",
                            extent=[0, n_contexts, 0, self.n_actions])
            plt.colorbar(im, label="Estimated Q-value")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            plt.title(title)
            plt.xlabel("Contexts (X-axis)")
            plt.ylabel("Recommendations (Y-axis)")
            plt.show()


class RecommendedAgent:
    def __init__(self,
                 exploration_rate=1,
                 exploration_decay=0.999,
                 min_exploration_rate=0.001,
                 type='egreedy'):
        """
        Agent decides to accept/decline a recommendation given context and recommendation.
        """
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.n_actions = 2  # 0 = accept, 1 = decline
        self.q_table = {}  # key: (context, recommendation)
        self.action_counts = {}
        self.time = 0  # for UCB
        self.type = type

    def act(self, context, recommendation):
        """
        Decide to accept (True) or decline (False) the recommendation.
        """
        key = (context, recommendation)
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
        return action == 0  # True = accept, False = decline

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


    def update(self, context, recommendation, accepted, reward):
        """
        Update Q-value after observing reward.
        """
        key = (context, recommendation)
        action = 0 if accepted else 1
        n = self.action_counts[key][action]
        q = self.q_table[key][action]
        self.q_table[key][action] = q + (reward - q) / n

        # Decay exploration for epsilon-greedy
        if not self.type == 'ucb':
            self.exploration_rate = max(self.min_exploration_rate,
                                        self.exploration_rate * self.exploration_decay)

    def visualize_accept_q_landscape(self, context_list, recommendation_list, title="User Learned Q-values for Accept"):
        """
        Visualize the Q-values for action 0 (Accept) across contexts and recommendations.

        Parameters:
        - context_list: ordered list of context values (x-axis)
        - recommendation_list: ordered list of recommendation values (y-axis)
        """
        n_contexts = len(context_list)
        n_recommendations = len(recommendation_list)
        q_matrix = np.full((n_recommendations, n_contexts), np.nan)

        for i, rec in enumerate(recommendation_list):
            for j, ctx in enumerate(context_list):
                key = (ctx, rec)
                if key in self.q_table:
                    q_matrix[i, j] = self.q_table[key][0]  # Q-value for action 0 (accept)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(q_matrix, cmap="viridis", origin="lower", aspect="auto",
                        extent=[0, n_contexts, 0, n_recommendations])
        plt.colorbar(im, label="Q-value for Accept (Action=0)")
        plt.title(title)
        plt.xlabel("Contexts (X-axis)")
        plt.ylabel("Recommendations (Y-axis)")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.show()
