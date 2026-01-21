import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.data.download import MovieLensPipeline, AmazonBeautyPipeline

class TestPipelines(unittest.TestCase):

    @patch('src.data.download.requests.get')
    def test_movielens_download_mock(self, mock_get):
        # Mock the response content to be a valid zip file with expected CSVs
        import zipfile
        import io

        # Create a dummy zip file in memory
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('ml-latest-small/ratings.csv', "userId,movieId,rating,timestamp\n1,1,4.0,964982703")
            zf.writestr('ml-latest-small/movies.csv', "movieId,title,genres\n1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy")

        mock_response = MagicMock()
        mock_response.content = mem_zip.getvalue()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        pipeline = MovieLensPipeline(save_dir="tests/test_data")
        ratings, movies = pipeline.load_data()

        self.assertIsNotNone(ratings)
        self.assertIsNotNone(movies)
        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings.iloc[0]['rating'], 4.0)

    @patch('src.data.download.requests.get')
    def test_amazon_download_mock(self, mock_get):
        # Mock response for gzipped json lines
        import gzip

        dummy_json = '{"reviewerID": "A1", "asin": "123", "overall": 5.0, "reviewText": "Good", "unixReviewTime": 111}\n'
        compressed_data = gzip.compress(dummy_json.encode('utf-8'))

        mock_response = MagicMock()
        mock_response.content = compressed_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock open to avoid writing to disk in download method if we didn't mock requests right,
        # but since we mock requests.get, it returns content.
        # However, the code writes to disk. We should mock built-in open for the write part to avoid side effects?
        # For simplicity, we let it write to a test dir.

        pipeline = AmazonBeautyPipeline(save_dir="tests/test_data")
        # Ensure we don't have a cached file interfering
        cached_file = "tests/test_data/raw/reviews_Beauty_5.json.gz"
        if os.path.exists(cached_file):
            os.remove(cached_file)

        df = pipeline.load_data()

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['reviewerID'], 'A1')

import os
if __name__ == '__main__':
    if not os.path.exists("tests/test_data/raw"):
        os.makedirs("tests/test_data/raw")
    unittest.main()
