"""
Data adapters for connecting real datasets to the simulation framework.

Provides adapters that transform MovieLens and Amazon Beauty data
into formats compatible with the ExogenousRewardEnvironment and
bandit experiments.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Type
import os

import numpy as np
import pandas as pd


class BaseDataAdapter(ABC):
    """Base class for data source adapters."""

    @abstractmethod
    def load_data(self) -> Any:
        """Load and preprocess the raw data."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded data."""
        pass


class MovieLensEnvironmentAdapter(BaseDataAdapter):
    """
    Adapter that converts MovieLens data into a reward landscape
    for ExogenousRewardEnvironment.

    Strategy:
    - Movies -> Recommendations (Y-axis): Select top-N popular movies
    - Users -> Contexts (X-axis): Cluster users by genre preferences
    - Ratings -> Rewards: Average rating per (user_cluster, movie) cell

    Design Decisions:
    1. Top-N movies by rating count ensures sufficient data per cell
    2. User clustering via genre preferences creates meaningful contexts
    3. Missing cells filled with global mean rating
    4. Rewards normalized to simulation range [-50, 100]
    """

    def __init__(
        self,
        data_dir: str = "data",
        n_recommendations: int = 20,
        n_contexts: int = 50,
        clustering_method: str = "genre"
    ):
        self.data_dir = data_dir
        self.n_recommendations = n_recommendations
        self.n_contexts = n_contexts
        self.clustering_method = clustering_method

        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.movie_map: Dict[int, int] = {}  # movieId -> recommendation index
        self.reverse_movie_map: Dict[int, int] = {}  # index -> movieId
        self.user_map: Dict[int, int] = {}   # userId -> context index
        self.reward_landscape: Optional[np.ndarray] = None
        self._global_max_pos: Optional[Tuple[int, int]] = None
        self._local_max_pos: Optional[Tuple[int, int]] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ratings and movies from interim data."""
        ratings_path = os.path.join(self.data_dir, "interim", "ratings.csv")
        movies_path = os.path.join(self.data_dir, "interim", "movies.csv")

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Ratings file not found at {ratings_path}. "
                "Run 'python -m src.data.process' first."
            )

        self.ratings_df = pd.read_csv(ratings_path)
        self.movies_df = pd.read_csv(movies_path)

        return self.ratings_df, self.movies_df

    def _select_top_movies(self) -> pd.Index:
        """
        Select top-N movies by popularity (rating count).

        Returns:
            Index of movieIds (length = n_recommendations)
        """
        movie_counts = self.ratings_df["movieId"].value_counts()
        top_movies = movie_counts.head(self.n_recommendations).index
        self.movie_map = {mid: idx for idx, mid in enumerate(top_movies)}
        self.reverse_movie_map = {idx: mid for mid, idx in self.movie_map.items()}
        return top_movies

    def _cluster_users_by_genre(self) -> pd.Series:
        """
        Cluster users into n_contexts groups based on genre preferences.

        Algorithm:
        1. Build user-genre matrix from ratings + movie genres
        2. Apply k-means clustering
        3. Map each user to a cluster (context) index

        Returns:
            Series mapping userId -> context_index
        """
        from sklearn.cluster import KMeans

        # Explode genres (handle pipe-separated format)
        movies_exploded = self.movies_df.copy()
        movies_exploded["genres"] = movies_exploded["genres"].str.split("|")
        movies_exploded = movies_exploded.explode("genres")

        # Merge with ratings
        merged = self.ratings_df.merge(
            movies_exploded[["movieId", "genres"]],
            on="movieId"
        )

        # Compute user-genre rating matrix
        user_genre = merged.groupby(
            ["userId", "genres"]
        )["rating"].mean().unstack(fill_value=0)

        # Handle case with few users
        n_users = len(user_genre)
        if n_users < self.n_contexts:
            # Not enough users, use modulo hashing
            user_cluster = pd.Series(
                {uid: uid % self.n_contexts for uid in self.ratings_df["userId"].unique()}
            )
        else:
            # K-means clustering on genre preferences
            kmeans = KMeans(
                n_clusters=self.n_contexts,
                random_state=42,
                n_init=10
            )
            clusters = kmeans.fit_predict(user_genre.values)
            user_cluster = pd.Series(clusters, index=user_genre.index)

        self.user_map = user_cluster.to_dict()
        return user_cluster

    def build_reward_landscape(self) -> np.ndarray:
        """
        Build the 2D reward landscape from MovieLens ratings.

        Returns:
            2D numpy array of shape (n_recommendations, n_contexts)
            where cell [rec, ctx] = average rating for that movie-user_cluster pair
        """
        if self.ratings_df is None:
            self.load_data()

        # Select movies and cluster users
        top_movies = self._select_top_movies()
        self._cluster_users_by_genre()

        # Filter to selected movies
        filtered = self.ratings_df[
            self.ratings_df["movieId"].isin(top_movies)
        ].copy()
        filtered["movie_idx"] = filtered["movieId"].map(self.movie_map)
        filtered["user_cluster"] = filtered["userId"].map(self.user_map)
        filtered = filtered.dropna(subset=["movie_idx", "user_cluster"])
        filtered["movie_idx"] = filtered["movie_idx"].astype(int)
        filtered["user_cluster"] = filtered["user_cluster"].astype(int)

        # Aggregate: mean rating per (movie_idx, user_cluster)
        pivot = filtered.groupby(
            ["movie_idx", "user_cluster"]
        )["rating"].mean().unstack(fill_value=np.nan)

        # Initialize landscape
        self.reward_landscape = np.zeros((self.n_recommendations, self.n_contexts))

        # Fill in available values
        global_mean = self.ratings_df["rating"].mean()
        for rec_idx in range(self.n_recommendations):
            for ctx_idx in range(self.n_contexts):
                if rec_idx in pivot.index and ctx_idx in pivot.columns:
                    val = pivot.loc[rec_idx, ctx_idx]
                    self.reward_landscape[rec_idx, ctx_idx] = (
                        val if not np.isnan(val) else global_mean
                    )
                else:
                    self.reward_landscape[rec_idx, ctx_idx] = global_mean

        # Normalize to simulation range [-50, 100]
        min_val = self.reward_landscape.min()
        max_val = self.reward_landscape.max()
        if max_val > min_val:
            self.reward_landscape = (
                -50 + 150 * (self.reward_landscape - min_val) / (max_val - min_val)
            )
        else:
            # All same value, set to middle of range
            self.reward_landscape = np.full_like(self.reward_landscape, 25.0)

        # Find peaks
        self._find_peaks()

        return self.reward_landscape

    def _find_peaks(self) -> None:
        """Find global and local maximum positions in the landscape."""
        if self.reward_landscape is None:
            return

        # Global max
        flat_idx = np.argmax(self.reward_landscape)
        self._global_max_pos = np.unravel_index(
            flat_idx, self.reward_landscape.shape
        )

        # Local max: mask area around global max and find next max
        masked = self.reward_landscape.copy()
        r, c = self._global_max_pos
        r_min = max(0, r - 3)
        r_max = min(self.n_recommendations, r + 4)
        c_min = max(0, c - 3)
        c_max = min(self.n_contexts, c + 4)
        masked[r_min:r_max, c_min:c_max] = -np.inf

        local_flat = np.argmax(masked)
        self._local_max_pos = np.unravel_index(local_flat, masked.shape)

    def create_environment_class(self) -> Type:
        """
        Create an environment class initialized with MovieLens data.

        Returns a class (not instance) that can be passed to run_recommender_simulation.
        The class captures the landscape data via closure.

        Returns:
            Environment class with MovieLens reward landscape
        """
        from src.environment import ExogenousRewardEnvironment

        if self.reward_landscape is None:
            self.build_reward_landscape()

        # Capture landscape and peaks in closure
        landscape = self.reward_landscape.copy()
        global_max = self._global_max_pos
        local_max = self._local_max_pos

        class MovieLensEnvironment(ExogenousRewardEnvironment):
            """Environment initialized with MovieLens ratings data."""

            def __init__(self, n_recommendations, n_contexts):
                super().__init__(n_recommendations, n_contexts)
                # Override with real data
                self.state_space = landscape.copy()
                self.global_max_pos = global_max
                self.local_max_pos = local_max

            def do_gaussian_smoothing(self):
                """Override to skip synthetic smoothing - we have real data."""
                pass

            def do_rows_gaussian_smoothing(self):
                """Override to skip synthetic smoothing - we have real data."""
                pass

        return MovieLensEnvironment

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded data."""
        info = {
            "data_source": "movielens",
            "n_ratings": len(self.ratings_df) if self.ratings_df is not None else 0,
            "n_movies": len(self.movies_df) if self.movies_df is not None else 0,
            "n_users": (
                self.ratings_df["userId"].nunique()
                if self.ratings_df is not None else 0
            ),
            "n_recommendations": self.n_recommendations,
            "n_contexts": self.n_contexts,
            "clustering_method": self.clustering_method,
            "movie_map": self.movie_map,
        }

        if self._global_max_pos is not None:
            info["global_max_pos"] = self._global_max_pos
            info["local_max_pos"] = self._local_max_pos

        return info

    def get_movie_title(self, rec_idx: int) -> str:
        """Get movie title for a recommendation index."""
        if self.movies_df is None or rec_idx not in self.reverse_movie_map:
            return f"Movie {rec_idx}"

        movie_id = self.reverse_movie_map[rec_idx]
        movie_row = self.movies_df[self.movies_df["movieId"] == movie_id]
        if len(movie_row) > 0:
            return movie_row.iloc[0]["title"]
        return f"Movie {movie_id}"


class AmazonBeautyBanditAdapter(BaseDataAdapter):
    """
    Adapter for Amazon Beauty data with LinUCB contextual bandit.

    Strategy:
    - Products -> Arms: Select top-N products by review count
    - Review text -> Context: TF-IDF features (100 dims by default)
    - Rating -> Reward: Binary (rating >= threshold -> 1, else 0)

    This adapter is used with BanditExperimentRunner, not grid simulation.
    """

    def __init__(
        self,
        data_dir: str = "data",
        n_arms: int = 50,
        context_dim: int = 100,
        reward_threshold: float = 4.0
    ):
        self.data_dir = data_dir
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.reward_threshold = reward_threshold

        self.df: Optional[pd.DataFrame] = None
        self.tfidf = None
        self.item_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}
        self.contexts: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None

    def load_data(self) -> pd.DataFrame:
        """Load Amazon Beauty data from interim."""
        data_path = os.path.join(self.data_dir, "interim", "amazon_beauty.json")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Amazon Beauty file not found at {data_path}. "
                "Run 'python -m src.data.process' first."
            )

        self.df = pd.read_json(data_path, orient="records", lines=True)
        return self.df

    def prepare_bandit_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for bandit training/evaluation.

        Returns:
            contexts: (n_samples, context_dim) TF-IDF features
            actions: (n_samples,) product indices (logged actions)
            rewards: (n_samples,) binary rewards
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.df is None:
            self.load_data()

        # Select top products by review count
        top_products = self.df["asin"].value_counts().head(self.n_arms).index.tolist()
        df_filtered = self.df[self.df["asin"].isin(top_products)].copy()

        # Create item mapping
        self.item_map = {asin: idx for idx, asin in enumerate(top_products)}
        self.reverse_item_map = {idx: asin for asin, idx in self.item_map.items()}

        # TF-IDF on review text
        self.tfidf = TfidfVectorizer(
            max_features=self.context_dim,
            stop_words="english"
        )
        self.contexts = self.tfidf.fit_transform(
            df_filtered["reviewText"].fillna("")
        ).toarray()

        # Actions and rewards
        self.actions = df_filtered["asin"].map(self.item_map).values.astype(int)
        self.rewards = (
            df_filtered["overall"] >= self.reward_threshold
        ).astype(int).values

        return self.contexts, self.actions, self.rewards

    def load_pretrained_model(self) -> Tuple[Any, Any, Dict]:
        """
        Load pretrained LinUCB model, TF-IDF vectorizer, and item map.

        Returns:
            agent: LinUCBAgent loaded from models/bandit_policy.npz
            tfidf: TfidfVectorizer from models/tfidf_vectorizer.pkl
            item_map: Dict from models/item_map.pkl
        """
        import joblib
        from src.agents import LinUCBAgent

        models_dir = "models"

        # Load saved model
        agent = LinUCBAgent(
            n_arms=self.n_arms,
            context_dim=self.context_dim,
            alpha=0.1,
            regularization=1.0
        )
        agent.load(os.path.join(models_dir, "bandit_policy.npz"))

        # Load TF-IDF and item map
        tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        item_map = joblib.load(os.path.join(models_dir, "item_map.pkl"))

        return agent, tfidf, item_map

    def get_context_features(self, review_text: str) -> np.ndarray:
        """Convert review text to context features."""
        if self.tfidf is None:
            raise ValueError(
                "TF-IDF not fitted. Call prepare_bandit_data() first."
            )
        return self.tfidf.transform([review_text]).toarray()[0]

    def get_reward(self, rating: float) -> int:
        """Convert rating to binary reward."""
        return int(rating >= self.reward_threshold)

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded data."""
        return {
            "data_source": "amazon_beauty",
            "n_reviews": len(self.df) if self.df is not None else 0,
            "n_products": (
                self.df["asin"].nunique() if self.df is not None else 0
            ),
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "reward_threshold": self.reward_threshold,
            "item_map": self.item_map,
        }

    def create_bandit_environment(self) -> Dict[str, Any]:
        """
        Create a bandit-style environment for LinUCB experiments.

        Returns dict with:
        - agent: Fresh LinUCBAgent
        - contexts: array of context vectors
        - logged_actions: array of historical actions
        - rewards: array of rewards
        - n_samples: number of samples
        - item_map: ASIN to arm index mapping
        """
        from src.agents import LinUCBAgent

        contexts, actions, rewards = self.prepare_bandit_data()

        agent = LinUCBAgent(
            n_arms=self.n_arms,
            context_dim=self.context_dim,
            alpha=0.1,
            regularization=1.0
        )

        return {
            "agent": agent,
            "contexts": contexts,
            "logged_actions": actions,
            "rewards": rewards,
            "n_samples": len(contexts),
            "item_map": self.item_map,
            "reverse_item_map": self.reverse_item_map,
        }
