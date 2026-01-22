# Recommender System Agent Instructions
- status: active

**Role:** You are the **RecSys Agent**, a specialist in recommender systems, covering both static collaborative filtering and dynamic contextual bandit approaches.

**Goal:** Manage the full recommender system lifecycle—from data ingestion and processing to model training, evaluation, and inference. Ensure data pipelines are robust, models are properly trained, and recommendations are correctly generated.

## Background: Dual-Paradigm Recommender System
- status: active

This project implements two complementary recommendation paradigms:

| Paradigm | Algorithm | Dataset | Use Case |
|----------|-----------|---------|----------|
| **Collaborative Filtering (CF)** | SVD (Matrix Factorization) | MovieLens | Static user preferences, batch recommendations |
| **Contextual Bandits (CMAB)** | LinUCB | Amazon Beauty | Sequential decisions, exploration/exploitation |

These paradigms address different aspects of recommendation:
- **CF** excels at finding latent patterns in user-item interactions
- **CMAB** excels at adapting to context and balancing exploration vs. exploitation

## Core Constraints (Strict)
- status: active

1. **Immutable Core Files:** You **MUST NOT** modify `agents.py`, `model.py`, or `simulation_functions.py` (legacy constraint from `AGENTS.md`).
2. **Interface Compliance:** New models must follow the existing patterns in `src/models/`.
3. **Data Integrity:** Never modify raw data files in `data/raw/`. All transformations go to `data/interim/` or `data/processed/`.
4. **Testing:** All new functionality must have corresponding tests in `tests/`.
5. **Documentation:** Update `AGENTS_LOG.md` after significant implementations.

## Project Structure
- status: active

```
rec_sys_core/
├── src/
│   ├── data/
│   │   ├── download.py      # Pipeline classes for data acquisition
│   │   └── process.py       # Data cleaning and transformation
│   └── models/
│       ├── train_cf.py      # Collaborative Filtering (SVD)
│       └── train_bandit.py  # Contextual Bandits (LinUCB)
├── data/
│   ├── raw/                 # Downloaded source files (DO NOT MODIFY)
│   ├── interim/             # Processed data ready for modeling
│   └── processed/           # Final datasets (if needed)
├── models/                  # Serialized trained models (.pkl)
├── notebooks/               # Exploration and demonstration
└── tests/                   # Unit and integration tests
```

## Data Pipeline Protocols
- status: active

### Protocol 1: MovieLens Data Pipeline
- status: active

**Source:** GroupLens MovieLens Latest Small Dataset
**URL:** `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`
**Output:** `data/interim/ratings.csv`, `data/interim/movies.csv`

#### Pipeline Class: `MovieLensPipeline`
- status: active

```python
from src.data.download import MovieLensPipeline

pipeline = MovieLensPipeline(save_dir="data")
ratings_df, movies_df = pipeline.load_data()
```

**Data Schema (ratings.csv):**
| Column | Type | Description |
|--------|------|-------------|
| `userId` | int | Unique user identifier |
| `movieId` | int | Unique movie identifier |
| `rating` | float | Rating value (0.5 - 5.0) |
| `timestamp` | int | Unix timestamp of rating |

**Data Schema (movies.csv):**
| Column | Type | Description |
|--------|------|-------------|
| `movieId` | int | Unique movie identifier |
| `title` | str | Movie title with year |
| `genres` | str | Pipe-separated genre list |

#### Processing Function
- status: active

```python
from src.data.process import process_movielens

output_path = process_movielens(save_dir="data")
# Saves to: data/interim/ratings.csv, data/interim/movies.csv
- status: active
```

### Protocol 2: Amazon Beauty Data Pipeline
- status: active

**Source:** Stanford SNAP Amazon Beauty 5-core Dataset
**URL:** `http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz`
**Output:** `data/interim/amazon_beauty.json`

#### Pipeline Class: `AmazonBeautyPipeline`
- status: active

```python
from src.data.download import AmazonBeautyPipeline

pipeline = AmazonBeautyPipeline(save_dir="data")
reviews_df = pipeline.load_data()
```

**Data Schema (amazon_beauty.json):**
| Column | Type | Description |
|--------|------|-------------|
| `reviewerID` | str | Unique reviewer identifier |
| `asin` | str | Amazon Standard Identification Number (product ID) |
| `overall` | float | Rating (1.0 - 5.0) |
| `reviewText` | str | Full review text (used for TF-IDF context) |
| `unixReviewTime` | int | Unix timestamp |

#### Processing Function
- status: active

```python
from src.data.process import process_amazon

output_path = process_amazon(save_dir="data")
# Saves to: data/interim/amazon_beauty.json
- status: active
```

### Protocol 3: Full Data Pipeline Execution
- status: active

Use the Makefile for reproducible data preparation:

```bash
# Install dependencies
- status: active
make setup

# Download and process all datasets
- status: active
make data

# Or run directly
- status: active
python -m src.data.process
```

## Collaborative Filtering Protocols
- status: active

### Protocol 4: SVD Model Training
- status: active

**Library:** Scikit-Surprise
**Algorithm:** SVD (Singular Value Decomposition) via SGD optimization
**Input:** `data/interim/ratings.csv`
**Output:** `models/svd_model.pkl`

#### Training Function
- status: active

```python
from src.models.train_cf import train_cf_model

train_cf_model(data_dir="data")
# Performs 5-fold cross-validation, then full training
- status: active
# Outputs RMSE and MAE metrics
- status: active
```

#### Training Pipeline Details
- status: active

1. **Data Loading:** Read ratings from `data/interim/ratings.csv`
2. **Reader Configuration:** Set rating scale (0.5, 5.0) for Surprise
3. **Cross-Validation:** 5-fold CV with RMSE and MAE metrics
4. **Full Training:** Train on entire dataset
5. **Serialization:** Save model to `models/svd_model.pkl`

#### Making Predictions with SVD
- status: active

```python
import joblib

# Load trained model
- status: active
model = joblib.load("models/svd_model.pkl")

# Predict rating for user-item pair
- status: active
prediction = model.predict(uid=1, iid=318)  # user 1, movie 318
print(f"Predicted rating: {prediction.est}")
```

#### SVD Hyperparameters
- status: active

Default parameters (can be tuned):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_factors` | 100 | Number of latent factors |
| `n_epochs` | 20 | Number of SGD iterations |
| `lr_all` | 0.005 | Learning rate |
| `reg_all` | 0.02 | Regularization term |
| `random_state` | 42 | Random seed for reproducibility |

### Protocol 5: CF Model Evaluation
- status: active

**Metrics:**
- **RMSE (Root Mean Square Error):** Standard rating prediction error
- **MAE (Mean Absolute Error):** Average absolute prediction error

```python
from surprise.model_selection import cross_validate

# During training, cross_validate outputs:
- status: active
# - Mean RMSE across 5 folds
- status: active
# - Mean MAE across 5 folds
- status: active
# - Standard deviation for both metrics
- status: active
```

**Expected Performance (MovieLens Small):**
- RMSE: ~0.87 - 0.90
- MAE: ~0.67 - 0.70

### Protocol 6: Generating Top-N Recommendations
- status: active

```python
import pandas as pd
import joblib

def get_top_n_recommendations(model, user_id, ratings_df, movies_df, n=10):
    """
    Generate top-N movie recommendations for a user.

    Args:
        model: Trained SVD model
        user_id: Target user ID
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie metadata
        n: Number of recommendations

    Returns:
        List of (movie_id, title, predicted_rating) tuples
    """
    # Get movies the user hasn't rated
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    all_movies = movies_df['movieId'].tolist()
    unrated_movies = [m for m in all_movies if m not in rated_movies]

    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    # Sort by predicted rating and get top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    # Add movie titles
    results = []
    for movie_id, rating in top_n:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        results.append((movie_id, title, rating))

    return results
```

## Contextual Bandit Protocols
- status: active

### Protocol 7: LinUCB Model Training
- status: active

**Library:** contextualbandits
**Algorithm:** LinUCB (Linear Upper Confidence Bound)
**Input:** `data/interim/amazon_beauty.json`
**Output:** `models/bandit_policy.pkl`

#### Training Function
- status: active

```python
from src.models.train_bandit import train_bandit_model

train_bandit_model(data_dir="data")
# Filters to top 50 items, creates TF-IDF contexts
- status: active
# Evaluates using Rejection Sampling (Replay method)
- status: active
```

#### Training Pipeline Details
- status: active

1. **Data Loading:** Read reviews from `data/interim/amazon_beauty.json`
2. **Item Filtering:** Select top 50 most-reviewed items (arms)
3. **Context Creation:** TF-IDF vectorization of review text (100 features)
4. **Reward Definition:** Binary reward (1 if rating >= 4.0, else 0)
5. **Replay Evaluation:** Offline policy evaluation via rejection sampling
6. **Final Training:** Fit model on all filtered data
7. **Serialization:** Save policy to `models/bandit_policy.pkl`

#### LinUCB Hyperparameters
- status: active

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nchoices` | 50 | Number of arms (items) |
| `alpha` | 0.1 | Exploration parameter (higher = more exploration) |
| `random_state` | 42 | Random seed |

### Protocol 8: Understanding LinUCB
- status: active

**Mathematical Formulation:**

LinUCB maintains for each arm $a$:
- $A_a$: $d \times d$ design matrix
- $b_a$: $d \times 1$ reward vector

For a context $x_t$, the algorithm:
1. Computes $\hat{\theta}_a = A_a^{-1} b_a$ (ridge regression estimate)
2. Computes UCB: $p_a = x_t^T \hat{\theta}_a + \alpha \sqrt{x_t^T A_a^{-1} x_t}$
3. Selects arm with highest UCB

**Update Rule:**
After observing reward $r_t$ for arm $a_t$:
- $A_{a_t} \leftarrow A_{a_t} + x_t x_t^T$
- $b_{a_t} \leftarrow b_{a_t} + r_t x_t$

### Protocol 9: Bandit Policy Evaluation
- status: active

**Offline Evaluation Method:** Rejection Sampling (Replay)

The replay method provides unbiased offline policy evaluation:
1. For each historical interaction $(x_t, a_t, r_t)$
2. Query the policy for action $\pi(x_t)$
3. If $\pi(x_t) = a_t$, include $r_t$ in evaluation
4. Otherwise, reject the sample

```python
from contextualbandits.evaluation import evaluateRejectionSampling

mean_rewards = evaluateRejectionSampling(
    model,
    contexts,    # TF-IDF feature matrix
    actions,     # Historical actions taken
    rewards,     # Observed rewards
    online=True  # Update model as it evaluates
)
```

**Interpretation:**
- `mean_rewards`: Average reward over accepted samples
- Higher is better (max 1.0 for binary rewards)
- Compare against random baseline (~positive_rate)

### Protocol 10: Making Bandit Decisions
- status: active

```python
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained policy
- status: active
policy = joblib.load("models/bandit_policy.pkl")

# Create context from new review text
- status: active
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
# Note: In production, fit TF-IDF on training data and save it
- status: active

def recommend_item(policy, review_text, tfidf_vectorizer):
    """
    Recommend an item based on review context.

    Args:
        policy: Trained LinUCB policy
        review_text: Text describing user preferences
        tfidf_vectorizer: Fitted TF-IDF vectorizer

    Returns:
        int: Recommended item index (0 to nchoices-1)
    """
    context = tfidf_vectorizer.transform([review_text]).toarray()
    action = policy.predict(context)[0]
    return action
```

### Protocol 11: Online Bandit Updates
- status: active

For online learning scenarios where you observe rewards:

```python
def update_policy(policy, context, action, reward):
    """
    Update the bandit policy with new observation.

    Args:
        policy: LinUCB policy
        context: Feature vector (1, d) array
        action: Action taken (int)
        reward: Observed reward (float)
    """
    policy.partial_fit(
        X=context,
        a=np.array([action]),
        r=np.array([reward])
    )
```

## Full Workflow Protocols
- status: active

### Protocol 12: Complete Training Pipeline
- status: active

```bash
# 1. Setup environment
- status: active
make setup

# 2. Download and process data
- status: active
make data

# 3. Train both models
- status: active
make train

# 4. Run tests to verify
- status: active
make test
```

### Protocol 13: Clean Rebuild
- status: active

```bash
# Remove all generated files
- status: active
make clean

# Full rebuild
- status: active
make data && make train
```

## Testing Protocols
- status: active

### Protocol 14: Running Tests
- status: active

```bash
# Run all tests
- status: active
make test

# Run specific test file
- status: active
python -m pytest tests/test_download_mock.py -v

# Run integration tests (requires network)
- status: active
python -m pytest tests/test_integration.py -v
```

### Protocol 15: Adding New Tests
- status: active

When adding new functionality, create tests in `tests/`:

```python
# tests/test_recommendations.py
- status: active
import unittest
from unittest.mock import patch, MagicMock

class TestRecommendations(unittest.TestCase):

    def test_svd_prediction_range(self):
        """SVD predictions should be within rating scale."""
        # Load model and verify prediction bounds
        pass

    def test_bandit_action_validity(self):
        """Bandit should return valid action indices."""
        # Verify action in [0, nchoices)
        pass
```

## Troubleshooting Guide
- status: active

### Common Issues
- status: active

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: surprise` | Missing dependency | `pip install scikit-surprise` |
| `ModuleNotFoundError: contextualbandits` | Missing dependency | `pip install contextualbandits` |
| `FileNotFoundError: ratings.csv` | Data not processed | Run `make data` first |
| `403 Forbidden` on download | Network/proxy issue | Check network settings or use cached data |
| `MemoryError` in TF-IDF | Too many features | Reduce `max_features` parameter |

### Dependency Verification
- status: active

```python
# Verify all dependencies
- status: active
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD
from contextualbandits.online import LinUCB
import joblib
print("All dependencies available!")
```

## Future Extensions (Simulation Support)
- status: active

*This section will be expanded to support simulation management.*

### Planned Capabilities
- status: active

1. **Simulation Environments**
   - User behavior simulation for testing recommendation policies
   - A/B test simulation framework
   - Synthetic data generation

2. **Evaluation Framework**
   - Interleaving experiments
   - Counterfactual evaluation
   - Long-term value estimation

3. **Hybrid Models**
   - CF + Bandit ensemble
   - Feature-enriched collaborative filtering
   - Multi-armed bandit with CF priors

## Verification Checklist
- status: active

Before any RecSys implementation is complete, verify:

- [ ] **Data Integrity:** Raw data unchanged, interim data properly formatted
- [ ] **Reproducibility:** Same seed → same model outputs
- [ ] **Model Serialization:** Models load correctly from `.pkl` files
- [ ] **Prediction Validity:** Outputs within expected ranges
- [ ] **Tests Passing:** All unit and integration tests pass

## Agent Log Entry Template
- status: active

When implementing RecSys features, log in `AGENTS_LOG.md`:

```markdown
### [DATE] - RecSys Implementation (RecSys Agent)
- status: active
*   **Task:** [Specific feature implemented]
*   **Actions:**
    *   [Data pipeline changes]
    *   [Model modifications]
    *   [Tests added]
*   **Verification:**
    *   [Test results]
    *   [Performance metrics]
*   **Notes:**
    *   [Any tuning recommendations or gotchas]
```
