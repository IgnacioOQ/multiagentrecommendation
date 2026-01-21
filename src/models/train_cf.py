import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import joblib

def train_cf_model(data_dir="data"):
    """
    Trains a Collaborative Filtering model (SVD) on MovieLens data.
    """
    ratings_path = os.path.join(data_dir, "interim", "ratings.csv")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}. Run data processing first.")

    print("Loading ratings data...")
    df = pd.read_csv(ratings_path)

    # Surprise Reader
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    print("Training SVD model...")
    algo = SVD(random_state=42)

    # Cross validation
    print("Running cross-validation...")
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Full training
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "svd_model.pkl")
    joblib.dump(algo, model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_cf_model()
