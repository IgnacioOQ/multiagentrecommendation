import unittest
import os
from src.data.download import MovieLensPipeline
from src.data.process import process_movielens

class TestIntegration(unittest.TestCase):

    def test_pipeline_integration(self):
        # This test actually downloads data, so it might be slow.
        # We can check for an environment variable to skip it if needed.
        if os.environ.get("SKIP_INTEGRATION"):
            self.skipTest("Skipping integration test")

        print("\\nRunning integration test (this may take time)...")
        # We use a temporary dir or just the main data dir?
        # Let's use the main data dir but check if files exist first to be efficient

        # Test the processing function directly
        output_path = process_movielens(save_dir="data")

        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

if __name__ == '__main__':
    unittest.main()
