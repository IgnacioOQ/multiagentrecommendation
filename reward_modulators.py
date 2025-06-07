from imports import *

class MoodSwings:
    def __init__(self, n_moods=50):
        """
        Non-stationary mood model where the probabilities of mood changes drift over time.
        """
        self.n_moods = n_moods
        self.mood = 0
        self.mood_history = [self.mood]
        self.moves = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
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