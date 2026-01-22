import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.agents.bandit import LinUCBAgent


def replay_evaluation(
    agent: LinUCBAgent,
    contexts: np.ndarray,
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    random_state: int = 42,
) -> tuple[list[float], int]:
    """Evaluate bandit policy using rejection sampling (replay method).

    This method evaluates a bandit policy on historical log data by only
    using samples where the agent's chosen action matches the logged action.

    Args:
        agent: LinUCB agent to evaluate.
        contexts: Context vectors of shape (n_samples, context_dim).
        logged_actions: Actions taken in the historical log.
        rewards: Rewards received for the logged actions.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cumulative_mean_rewards, n_matched) where cumulative_mean_rewards
        is a list of running mean rewards and n_matched is the number of matched samples.
    """
    np.random.seed(random_state)

    n_samples = len(contexts)
    cumulative_reward = 0.0
    n_matched = 0
    mean_rewards = []

    for i in tqdm(range(n_samples), desc="Replay evaluation"):
        context = contexts[i]
        logged_action = logged_actions[i]
        reward = rewards[i]

        # Agent selects action based on current policy
        selected_action = agent.select_action(context)

        # Rejection sampling: only use if actions match
        if selected_action == logged_action:
            n_matched += 1
            cumulative_reward += reward

            # Update agent with this observation
            agent.store(context, selected_action, reward, context, False)
            agent.update()

            mean_rewards.append(cumulative_reward / n_matched)

    return mean_rewards, n_matched


def train_bandit_model(data_dir: str = "data") -> dict:
    """Train a Contextual Bandit model (LinUCB) on Amazon Beauty data.

    Args:
        data_dir: Base directory for data files.

    Returns:
        Dictionary containing training metrics.
    """
    data_path = os.path.join(data_dir, "interim", "amazon_beauty.json")

    if not os.path.exists(data_path):
        from src.data.process import process_amazon
        process_amazon(save_dir=data_dir)

    print("Loading Amazon data...")
    df = pd.read_json(data_path, orient='records', lines=True)

    # Limit to top items to make it feasible for LinUCB (matrix inversion)
    top_items = df['asin'].value_counts().head(50).index.tolist()
    df_filtered = df[df['asin'].isin(top_items)].copy()

    if len(df_filtered) < 100:
        print("Warning: Not enough data after filtering.")

    n_arms = len(top_items)
    print(f"Using {len(df_filtered)} interactions with {n_arms} unique items.")

    # Create contexts using TF-IDF
    context_dim = 100
    tfidf = TfidfVectorizer(max_features=context_dim, stop_words='english')
    contexts = tfidf.fit_transform(df_filtered['reviewText'].fillna('')).toarray()

    # Actions (items) mapped to integers
    item_map = {item: i for i, item in enumerate(top_items)}
    actions = df_filtered['asin'].map(item_map).values

    # Binary rewards: rating >= 4.0 -> 1, else 0
    rewards = (df_filtered['overall'] >= 4.0).astype(int).values

    # Initialize custom LinUCB agent
    print("Training LinUCB...")
    agent = LinUCBAgent(
        n_arms=n_arms,
        context_dim=context_dim,
        alpha=0.1,
        regularization=1.0,
        name="LinUCB_AmazonBeauty",
    )

    # Evaluate using rejection sampling (replay method)
    print("Evaluating with Rejection Sampling (Replay)...")
    mean_rewards, n_matched = replay_evaluation(
        agent, contexts, actions, rewards, random_state=42
    )

    if n_matched > 0:
        final_mean_reward = mean_rewards[-1] if mean_rewards else 0.0
        print(f"Mean Reward (Replay): {final_mean_reward:.4f}")
        print(f"Matched samples: {n_matched}/{len(contexts)} ({100*n_matched/len(contexts):.1f}%)")
    else:
        print("Warning: No matched samples in replay evaluation.")
        final_mean_reward = 0.0

    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "bandit_policy.npz")
    agent.save(model_path)

    # Also save the TF-IDF vectorizer and item mapping for inference
    import joblib
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    item_map_path = os.path.join(models_dir, "item_map.pkl")
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(item_map, item_map_path)

    print(f"Bandit policy saved to {model_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")
    print(f"Item mapping saved to {item_map_path}")

    return {
        "mean_reward": final_mean_reward,
        "n_matched": n_matched,
        "n_total": len(contexts),
        "n_arms": n_arms,
        "context_dim": context_dim,
    }

if __name__ == "__main__":
    train_bandit_model()
