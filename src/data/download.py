import os
import requests
import zipfile
import gzip
import io
import pandas as pd
import json

class MovieLensPipeline:
    URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        self.raw_dir = os.path.join(save_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def load_data(self):
        """
        Downloads (if not present) and loads the MovieLens dataset.
        Returns two dataframes: ratings and movies.
        """
        zip_path = os.path.join(self.raw_dir, "ml-latest-small.zip")

        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens data from {self.URL}...")
            response = requests.get(self.URL)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)

        # We can read directly from zip without extracting all files if we want,
        # or extract them. Let's read directly to be cleaner or extract to raw.
        # The test mock creates a zip structure.

        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('ml-latest-small/ratings.csv') as f:
                ratings = pd.read_csv(f)
            with zf.open('ml-latest-small/movies.csv') as f:
                movies = pd.read_csv(f)

        return ratings, movies

class AmazonBeautyPipeline:
    URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"

    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        self.raw_dir = os.path.join(save_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def load_data(self):
        """
        Downloads (if not present) and loads the Amazon Beauty dataset.
        Returns a dataframe.
        """
        gz_path = os.path.join(self.raw_dir, "reviews_Beauty_5.json.gz")

        if not os.path.exists(gz_path):
            print(f"Downloading Amazon Beauty data from {self.URL}...")
            response = requests.get(self.URL)
            response.raise_for_status()
            with open(gz_path, "wb") as f:
                f.write(response.content)

        data = []
        with gzip.open(gz_path, 'rb') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        df = pd.DataFrame(data)
        return df
