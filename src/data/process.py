import os
import pandas as pd
from src.data.download import MovieLensPipeline, AmazonBeautyPipeline

def process_movielens(save_dir="data"):
    """
    Runs the MovieLens pipeline and saves the processed ratings to csv.
    """
    pipeline = MovieLensPipeline(save_dir=save_dir)
    ratings, movies = pipeline.load_data()

    interim_dir = os.path.join(save_dir, "interim")
    os.makedirs(interim_dir, exist_ok=True)

    output_path = os.path.join(interim_dir, "ratings.csv")
    ratings.to_csv(output_path, index=False)

    # Also save movies for reference if needed
    movies.to_csv(os.path.join(interim_dir, "movies.csv"), index=False)

    print(f"Processed MovieLens data saved to {output_path}")
    return output_path

def process_amazon(save_dir="data"):
    """
    Runs the Amazon Beauty pipeline and saves the processed data to json/csv.
    """
    pipeline = AmazonBeautyPipeline(save_dir=save_dir)
    df = pipeline.load_data()

    interim_dir = os.path.join(save_dir, "interim")
    os.makedirs(interim_dir, exist_ok=True)

    output_path = os.path.join(interim_dir, "amazon_beauty.json")
    # Save as json records
    df.to_json(output_path, orient='records', lines=True)

    print(f"Processed Amazon data saved to {output_path}")
    return output_path

if __name__ == "__main__":
    process_movielens()
    process_amazon()
