from imports import *

class ExogenousRewardEnvironment:
    def __init__(self, n_recommendations=20, n_contexts=50):
        self.n_recommendations = n_recommendations  # Y-axis
        self.n_contexts = n_contexts  # X-axis
        self.state_space = np.zeros((self.n_recommendations, self.n_contexts))
        center_y = self.n_recommendations // 2
        center_x = self.n_contexts // 2
        # self.global_max_pos = (15, 40)  # (y, x)
        # self.local_max_pos = (5, 5)  # (y, x)
        self.global_max_pos = (center_y + 5, center_x + 10)  # slightly offset from center
        self.local_max_pos = (center_y - 5, center_x - 10)   # slightly offset the other way
        # self.do_gaussian_smoothing()

        # --- New additions ---
        self.current_context = np.random.randint(0, self.n_contexts)  # Random start
        self.context_history = [self.current_context]  # Initialize history

    def gaussian_peak(self, x, y, peak_x, peak_y, strength=5, spread_x=None, spread_y=None):
        spread_x = self.n_contexts / 10 if spread_x is None else spread_x
        spread_y = self.n_recommendations / 10 if spread_y is None else spread_y
        return strength * np.exp(-(((x - peak_x) ** 2) / (2 * spread_x ** 2) +
                                   ((y - peak_y) ** 2) / (2 * spread_y ** 2)))

    def do_gaussian_smoothing(self):
        for i in range(self.n_recommendations):
            for j in range(self.n_contexts):
                self.state_space[i, j] = max(
                    self.state_space[i, j],
                    self.gaussian_peak(j, i, self.global_max_pos[1], self.global_max_pos[0], strength=100),
                    self.gaussian_peak(j, i, self.local_max_pos[1], self.local_max_pos[0], strength=60)
                )
        # Normalize the state space to a specific range
        lower_bound = -50  # for example
        upper_bound = 100

        min_val = np.min(self.state_space)
        max_val = np.max(self.state_space)

        if max_val > min_val:
            scale = (upper_bound - lower_bound) / (max_val - min_val)
            self.state_space = lower_bound + scale * (self.state_space - min_val)
        else:
            self.state_space.fill((upper_bound + lower_bound) / 2)  # set to midpoint if constant


    def do_rows_gaussian_smoothing(self):
      global_y = self.global_max_pos[0]  # Use only row coordinate
      local_y = self.local_max_pos[0]

      for j in range(self.n_contexts):  # For each context (column)
          for i in range(self.n_recommendations):  # For each recommendation (row)
              global_peak = self.gaussian_peak(j, i, j, global_y, strength=100, spread_x=1, spread_y=2)
              local_peak = self.gaussian_peak(j, i, j, local_y, strength=60, spread_x=1, spread_y=2)
              self.state_space[i, j] = max(self.state_space[i, j], global_peak, local_peak)


    def get_state_value(self, x_idx, y_idx):
        if 0 <= x_idx < self.n_contexts and 0 <= y_idx < self.n_recommendations:
            return self.state_space[y_idx, x_idx]
        else:
            return None

    def visualize_landscape(self, current_y=None,mark_context=False):
        """
        Visualize the 2D state landscape and mark global/local maxima,
        as well as the current context with an optional current_y.

        Args:
            current_y (int, optional): If provided, marks (current_context, current_y) as current position.
        """
        plt.figure(figsize=(6, 4))
        plt.imshow(self.state_space, cmap="viridis", origin="lower", aspect="auto",
                extent=[0, self.n_contexts, 0, self.n_recommendations])
        plt.colorbar(label="State Value")
        
        # Mark maxima
        plt.scatter(self.global_max_pos[1] + 0.5, self.global_max_pos[0] + 0.5,
            color="red", marker="o", s=100, label="Global Max")
        plt.scatter(self.local_max_pos[1] + 0.5, self.local_max_pos[0] + 0.5,
            color="orange", marker="o", s=100, label="Local Max")

        # Mark current context with vertical line
        if mark_context:
            plt.axvline(x=self.current_context, color="green", linestyle="--", linewidth=2, label="Current Context")
                # Annotate current context number on x-axis
            plt.text(self.current_context + 0.5, -2, f"Context {self.current_context}",
             color="green", ha="center", va="top", fontsize=10, rotation=30)
        
        # Optionally show full (context, recommendation) location
        if current_y is not None:
            plt.scatter(self.current_context, current_y, color="lime", marker="x", s=100, label="Current Position")

        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.title("Environmental Rewards Landscape")
        plt.xlabel("Contexts (X-axis)")
        plt.ylabel("Recommendations (Y-axis)")
        plt.legend()
        plt.show()

    # --- New method: context transition ---
    def step_context(self):
        """
        Randomly move left (-1) or right (+1). Wrap around if reaching edge.
        Store the new context in history.
        """
        move = np.random.choice([-5,-4,-3,-2,-1, 1,2,3,4,5])
        self.current_context = (self.current_context + move) % self.n_contexts
        self.context_history.append(self.current_context)

    def get_context_history(self):
        """Return the full list of visited contexts."""
        return self.context_history

    def shift_environment_right(self):
      """
      Shift the reward landscape one context (column) to the right.
      The last column wraps around to the first.
      """
      self.state_space = np.roll(self.state_space, shift=1, axis=1)

      # Also update the position of the known maxima
      self.global_max_pos = (self.global_max_pos[0], (self.global_max_pos[1] + 1) % self.n_contexts)
      self.local_max_pos = (self.local_max_pos[0], (self.local_max_pos[1] + 1) % self.n_contexts)
