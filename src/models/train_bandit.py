import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from contextualbandits.online import LinUCB
from contextualbandits.evaluation import evaluateRejectionSampling
import joblib

def train_bandit_model(data_dir="data"):
    """
    Trains a Contextual Bandit model (LinUCB) on Amazon Beauty data.
    """
    data_path = os.path.join(data_dir, "interim", "amazon_beauty.json")

    if not os.path.exists(data_path):
        # Try processing if not exists? Or just raise error.
        # Let's check raw and process if needed.
        # But for now assume process ran.
        # If not found, try to run process_amazon?
        # Let's rely on makefile order usually, but here raise error.
        from src.data.process import process_amazon
        process_amazon(save_dir=data_dir)

    print("Loading Amazon data...")
    df = pd.read_json(data_path, orient='records', lines=True)

    # We need to simulate a bandit scenario.
    # Context: Review text (TF-IDF)
    # Action: Product (asin) - Too many actions?
    # Usually for bandit simulation on this dataset, we might filter top categories or items.
    # Given the description "text-based context", TF-IDF is correct.

    # Simplified setup:
    # We treat every review as a step.
    # Context: User or Review Text? AGENTS.md says "TF-IDF vectors generated from review text."
    # Wait, if context comes from review text, and we predict item? Or we predict rating?
    # Contextual Bandits usually: Given context -> Select Action -> Get Reward.
    # Here Action is likely the Item. Reward is the Rating.
    # But unlimited items is hard.

    # Let's follow a standard approach for this dataset if not specified.
    # Or just implementation of LinUCB class usage.

    # Limit to top items to make it feasible for LinUCB (matrix inversion).
    top_items = df['asin'].value_counts().head(50).index.tolist()
    df_filtered = df[df['asin'].isin(top_items)].copy()

    if len(df_filtered) < 100:
        print("Not enough data after filtering. Using all data but limiting actions randomly?")
        # Just use what we have.

    print(f"Using {len(df_filtered)} interactions with {len(top_items)} unique items.")

    # Create contexts
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    contexts = tfidf.fit_transform(df_filtered['reviewText'].fillna('')).toarray()

    # Actions (items) mapped to integers
    item_map = {item: i for i, item in enumerate(top_items)}
    actions = df_filtered['asin'].map(item_map).values

    # Rewards: Binary? Or Rating?
    # LinUCB often works with binary rewards or continuous.
    # Let's use Rating directly or threshold.
    # Threshold 4.0 -> 1, else 0.
    rewards = (df_filtered['overall'] >= 4.0).astype(int).values

    # Train LinUCB
    print("Training LinUCB...")
    n_choices = len(top_items)
    model = LinUCB(nchoices=n_choices, alpha=0.1, random_state=42)

    # Simulation: 'Replay' method (Rejection Sampling)
    # Since we are using historical log data.
    print("Evaluating with Rejection Sampling (Replay)...")

    # evaluateRejectionSampling expects:
    # model, X, a, r, online=True/False?
    # Actually it runs the simulation.

    # Using the library's evaluation method
    # It returns a list of mean rewards over time
    mean_rewards = evaluateRejectionSampling(model, contexts, actions, rewards, online=True)

    print(f"Mean Reward (Replay): {np.mean(mean_rewards):.4f}")

    # Fit the model on all data for final artifact
    model.fit(contexts, actions, rewards)

    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "bandit_policy.pkl")
    joblib.dump(model, model_path)

    print(f"Bandit policy saved to {model_path}")

if __name__ == "__main__":
    train_bandit_model()
