# Temporal Topic Analysis Pipeline - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular Python pipeline to cluster 50k social media posts, analyze topic evolution over time, and create interactive visualizations.

**Architecture:** Three standalone Python scripts (clustering, temporal analysis, visualization) with YAML configuration, Parquet caching, and Plotly interactive outputs.

**Tech Stack:** Python 3.8+, HDBSCAN, UMAP, Pandas, Plotly, PyYAML, scikit-learn

---

## Task 1: Project Setup & Configuration

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `scripts/__init__.py`
- Create: `output/.gitkeep`
- Create: `output/visualizations/.gitkeep`

**Step 1: Create requirements.txt**

```bash
cat > requirements.txt << 'EOF'
numpy>=1.21.0
pandas>=2.0.0
pyarrow>=12.0.0
hdbscan>=0.8.33
umap-learn>=0.5.3
scikit-learn>=1.3.0
plotly>=5.17.0
tqdm>=4.65.0
pyyaml>=6.0
anthropic>=0.18.0
EOF
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Create config.yaml**

```yaml
# Input files
input:
  embeddings: "embeddings.npy"
  data: "clustered_data_labeled.csv"
  text_column: "text"
  timestamp_column: "timestamp"

# Clustering parameters
clustering:
  algorithm: "hdbscan"
  min_cluster_size: 50
  min_samples: 10
  target_clusters: 50

# UMAP for visualization
umap:
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1
  metric: "cosine"

# Temporal analysis
temporal:
  default_window: "1M"  # 1 Month

# Output paths
output:
  dir: "output"
  clusters: "output/clusters.parquet"
  temporal_stats: "output/temporal_stats.parquet"
  temporal_events: "output/temporal_events.parquet"
  viz_dir: "output/visualizations"

# LLM labeling (for future)
llm:
  enabled: false
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  sample_size: 20
```

**Step 4: Create directory structure**

```bash
mkdir -p scripts output/visualizations
touch scripts/__init__.py
touch output/.gitkeep
touch output/visualizations/.gitkeep
```

**Step 5: Verify structure**

Run: `ls -R`
Expected: See `scripts/`, `output/`, `output/visualizations/`, `config.yaml`, `requirements.txt`

**Step 6: Commit setup**

```bash
git add requirements.txt config.yaml scripts/ output/
git commit -m "feat: add project setup and configuration

- Add dependencies in requirements.txt
- Create config.yaml for flexible configuration
- Set up directory structure for scripts and output

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Script 1 - Clustering (Part A: Core Functions)

**Files:**
- Create: `scripts/1_cluster_embeddings.py`
- Create: `tests/test_clustering.py`

**Step 1: Write test for data loading**

Create `tests/test_clustering.py`:
```python
import pytest
import numpy as np
import pandas as pd
from scripts.clustering_utils import load_data, validate_inputs

def test_load_data_validates_shape_mismatch():
    """Test that load_data raises error when embeddings and data don't match"""
    # Create mismatched data
    embeddings = np.random.rand(100, 768)
    data = pd.DataFrame({'text': ['test'] * 50})

    with pytest.raises(ValueError, match="Mismatch"):
        validate_inputs(embeddings, data, ['text'])

def test_load_data_validates_missing_columns():
    """Test that load_data raises error for missing required columns"""
    embeddings = np.random.rand(100, 768)
    data = pd.DataFrame({'wrong_column': ['test'] * 100})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_inputs(embeddings, data, ['text', 'timestamp'])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_clustering.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.clustering_utils'"

**Step 3: Create utility functions**

Create `scripts/clustering_utils.py`:
```python
"""Utility functions for clustering pipeline"""
import numpy as np
import pandas as pd
from typing import Tuple, List

def validate_inputs(embeddings: np.ndarray, data: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate that embeddings and data are compatible"""

    if len(embeddings) != len(data):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(data)} rows. "
            "Ensure embeddings.npy and CSV are aligned."
        )

    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def load_data(embeddings_path: str, data_path: str, config: dict) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and CSV data with validation"""

    # Load files
    embeddings = np.load(embeddings_path)
    data = pd.read_csv(data_path, low_memory=False)

    # Validate
    required_cols = [
        config['input']['text_column'],
        config['input']['timestamp_column']
    ]
    validate_inputs(embeddings, data, required_cols)

    return embeddings, data
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_clustering.py -v`
Expected: PASS (2 tests)

**Step 5: Commit utilities**

```bash
git add scripts/clustering_utils.py tests/test_clustering.py
git commit -m "feat: add data loading and validation utilities

- Add validate_inputs for shape and column checking
- Add load_data with automatic validation
- Add tests for error cases

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Script 1 - Clustering (Part B: HDBSCAN & UMAP)

**Files:**
- Modify: `scripts/clustering_utils.py`
- Modify: `tests/test_clustering.py`

**Step 1: Write test for clustering**

Add to `tests/test_clustering.py`:
```python
from scripts.clustering_utils import cluster_embeddings

def test_cluster_embeddings_returns_labels_and_coords():
    """Test that clustering returns cluster labels and UMAP coordinates"""
    embeddings = np.random.rand(100, 768)
    config = {
        'clustering': {
            'min_cluster_size': 5,
            'min_samples': 3
        },
        'umap': {
            'n_components': 2,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'cosine'
        }
    }

    cluster_labels, umap_coords = cluster_embeddings(embeddings, config)

    assert len(cluster_labels) == 100
    assert umap_coords.shape == (100, 2)
    assert cluster_labels.dtype in [np.int32, np.int64]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_clustering.py::test_cluster_embeddings_returns_labels_and_coords -v`
Expected: FAIL with "ImportError: cannot import name 'cluster_embeddings'"

**Step 3: Implement clustering function**

Add to `scripts/clustering_utils.py`:
```python
import hdbscan
import umap

def cluster_embeddings(embeddings: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Run HDBSCAN clustering and UMAP dimensionality reduction"""

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        min_samples=config['clustering']['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # UMAP for 2D visualization
    reducer = umap.UMAP(**config['umap'])
    umap_coords = reducer.fit_transform(embeddings)

    return cluster_labels, umap_coords
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_clustering.py::test_cluster_embeddings_returns_labels_and_coords -v`
Expected: PASS

**Step 5: Commit clustering implementation**

```bash
git add scripts/clustering_utils.py tests/test_clustering.py
git commit -m "feat: add HDBSCAN clustering and UMAP reduction

- Implement cluster_embeddings with HDBSCAN
- Add UMAP dimensionality reduction for visualization
- Test returns correct shapes and types

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Script 1 - Clustering (Part C: Auto-Labeling)

**Files:**
- Modify: `scripts/clustering_utils.py`
- Modify: `tests/test_clustering.py`

**Step 1: Write test for cluster labeling**

Add to `tests/test_clustering.py`:
```python
from scripts.clustering_utils import generate_cluster_labels

def test_generate_cluster_labels_creates_labels():
    """Test that cluster labels are generated from text"""
    df = pd.DataFrame({
        'text': [
            'machine learning model training',
            'deep learning neural networks',
            'machine learning algorithms',
        ] * 10,  # Repeat to have enough for clustering
        'cluster': [0] * 15 + [1] * 15
    })

    labels = generate_cluster_labels(df, n_top_terms=2)

    assert isinstance(labels, dict)
    assert 0 in labels
    assert 1 in labels
    assert len(labels[0].split(',')) == 2  # 2 terms
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_clustering.py::test_generate_cluster_labels_creates_labels -v`
Expected: FAIL with "ImportError: cannot import name 'generate_cluster_labels'"

**Step 3: Implement auto-labeling**

Add to `scripts/clustering_utils.py`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_cluster_labels(df: pd.DataFrame, n_top_terms: int = 3) -> dict:
    """Generate cluster labels using top TF-IDF terms"""

    cluster_labels = {}

    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Uncategorized/Noise"
            continue

        cluster_texts = df[df['cluster'] == cluster_id]['text'].fillna('')

        # TF-IDF to find characteristic terms
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Get top terms by average TF-IDF score
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_tfidf.argsort()[-n_top_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            cluster_labels[cluster_id] = ', '.join(top_terms).title()
        except:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"

    return cluster_labels
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_clustering.py::test_generate_cluster_labels_creates_labels -v`
Expected: PASS

**Step 5: Commit auto-labeling**

```bash
git add scripts/clustering_utils.py tests/test_clustering.py
git commit -m "feat: add automatic cluster labeling with TF-IDF

- Generate cluster labels from top TF-IDF terms
- Handle noise cluster (-1) specially
- Test label generation from sample data

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Script 1 - Clustering (Part D: Main Script)

**Files:**
- Create: `scripts/1_cluster_embeddings.py`

**Step 1: Write main clustering script**

Create `scripts/1_cluster_embeddings.py`:
```python
#!/usr/bin/env python3
"""
Script 1: Cluster Embeddings
Applies HDBSCAN clustering to embeddings and generates UMAP coordinates
"""
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from clustering_utils import load_data, cluster_embeddings, generate_cluster_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Cluster social media embeddings')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--force', action='store_true', help='Force rerun even if output exists')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Check if output exists
    output_path = config['output']['clusters']
    if Path(output_path).exists() and not args.force:
        logger.info(f"✓ {output_path} exists, skipping clustering. Use --force to rerun.")
        return

    # Create output directory
    Path(config['output']['dir']).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading embeddings and data...")
    embeddings, df = load_data(
        config['input']['embeddings'],
        config['input']['data'],
        config
    )
    logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

    # Run clustering
    logger.info("Running HDBSCAN clustering...")
    cluster_labels, umap_coords = cluster_embeddings(embeddings, config)

    # Add results to dataframe
    df['cluster'] = cluster_labels
    df['umap_x'] = umap_coords[:, 0]
    df['umap_y'] = umap_coords[:, 1]

    # Generate cluster labels
    logger.info("Generating cluster labels...")
    label_map = generate_cluster_labels(df, n_top_terms=3)
    df['cluster_label'] = df['cluster'].map(label_map)

    # Add cluster sizes
    cluster_sizes = df['cluster'].value_counts().to_dict()
    df['cluster_size'] = df['cluster'].map(cluster_sizes)

    # Log statistics
    n_clusters = len([c for c in df['cluster'].unique() if c != -1])
    noise_count = (df['cluster'] == -1).sum()
    noise_pct = (noise_count / len(df)) * 100

    logger.info(f"✓ Found {n_clusters} clusters (excluding noise)")
    logger.info(f"✓ Noise points: {noise_count} ({noise_pct:.1f}%)")
    logger.info(f"✓ Largest cluster: {max(cluster_sizes.values())} posts")

    # Save results
    logger.info(f"Saving results to {output_path}...")
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Complete! Saved to {output_path}")

if __name__ == '__main__':
    main()
```

**Step 2: Test script runs**

Run: `python scripts/1_cluster_embeddings.py --config config.yaml`
Expected: Script loads data, runs clustering, saves output/clusters.parquet

**Step 3: Verify output**

Run: `python -c "import pandas as pd; df = pd.read_parquet('output/clusters.parquet'); print(df[['cluster', 'cluster_label', 'umap_x', 'umap_y']].head())"`
Expected: Shows clustered data with new columns

**Step 4: Make script executable**

Run: `chmod +x scripts/1_cluster_embeddings.py`

**Step 5: Commit main script**

```bash
git add scripts/1_cluster_embeddings.py
git commit -m "feat: add clustering main script

- Load embeddings and CSV data
- Run HDBSCAN clustering
- Generate UMAP coordinates
- Auto-label clusters with TF-IDF
- Save to Parquet with progress logging

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Script 2 - Temporal Analysis (Part A: Basic Aggregation)

**Files:**
- Create: `scripts/temporal_utils.py`
- Create: `tests/test_temporal.py`

**Step 1: Write test for temporal aggregation**

Create `tests/test_temporal.py`:
```python
import pytest
import pandas as pd
import numpy as np
from scripts.temporal_utils import temporal_aggregation

def test_temporal_aggregation_creates_time_windows():
    """Test that temporal aggregation groups by time windows"""
    dates = pd.date_range('2026-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'cluster': [0, 1] * 50,
        'text': ['test'] * 100,
        'likes': np.random.randint(0, 100, 100),
        'reach': np.random.randint(0, 1000, 100),
        'shares': np.random.randint(0, 50, 100),
        'comments': np.random.randint(0, 20, 100),
        'Author': [f'user{i%10}' for i in range(100)],
        'Sentiment': ['positive', 'neutral', 'negative'] * 33 + ['positive']
    })

    result = temporal_aggregation(df, time_window='1M')

    assert 'cluster' in result.columns
    assert 'time_window' in result.columns
    assert 'post_count' in result.columns
    assert len(result) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_temporal.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.temporal_utils'"

**Step 3: Implement basic temporal aggregation**

Create `scripts/temporal_utils.py`:
```python
"""Utility functions for temporal analysis pipeline"""
import pandas as pd
import numpy as np
from typing import Dict

def temporal_aggregation(df: pd.DataFrame, time_window: str = '1M') -> pd.DataFrame:
    """Aggregate cluster activity by time windows"""

    # Ensure timestamp is datetime
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group by cluster and time window
    grouped = df.groupby([
        'cluster',
        pd.Grouper(key='timestamp', freq=time_window)
    ]).agg({
        'text': 'count',
        'likes': ['sum', 'mean', 'std'],
        'reach': ['sum', 'mean', 'std'],
        'shares': ['sum', 'mean'],
        'comments': ['sum', 'mean'],
        'Author': 'nunique',
        'Sentiment': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    # Flatten column names
    grouped.columns = [
        'cluster', 'time_window', 'post_count',
        'total_likes', 'avg_likes', 'std_likes',
        'total_reach', 'avg_reach', 'std_reach',
        'total_shares', 'avg_shares',
        'total_comments', 'avg_comments',
        'unique_authors', 'sentiment_dist'
    ]

    # Fill NaN values
    grouped = grouped.fillna(0)

    return grouped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_temporal.py -v`
Expected: PASS

**Step 5: Commit basic temporal aggregation**

```bash
git add scripts/temporal_utils.py tests/test_temporal.py
git commit -m "feat: add basic temporal aggregation

- Group posts by cluster and time window
- Calculate volume, engagement, and author metrics
- Support flexible time windows (1D, 1W, 1M, etc.)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Script 2 - Temporal Analysis (Part B: Advanced Metrics)

**Files:**
- Modify: `scripts/temporal_utils.py`
- Modify: `tests/test_temporal.py`

**Step 1: Write test for advanced metrics**

Add to `tests/test_temporal.py`:
```python
from scripts.temporal_utils import calculate_advanced_metrics

def test_calculate_advanced_metrics_adds_volatility():
    """Test that advanced metrics include volatility and growth"""
    df = pd.DataFrame({
        'cluster': [0] * 10,
        'time_window': pd.date_range('2026-01-01', periods=10, freq='W'),
        'post_count': [10, 15, 12, 20, 18, 25, 22, 30, 28, 35],
        'total_likes': [100, 150, 120, 200, 180, 250, 220, 300, 280, 350]
    })

    result = calculate_advanced_metrics(df)

    assert 'volume_pct_change' in result.columns
    assert 'volume_volatility' in result.columns
    assert 'momentum' in result.columns
    assert 'anomaly_score' in result.columns
    assert 'lifecycle_state' in result.columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_temporal.py::test_calculate_advanced_metrics_adds_volatility -v`
Expected: FAIL with "ImportError: cannot import name 'calculate_advanced_metrics'"

**Step 3: Implement advanced metrics**

Add to `scripts/temporal_utils.py`:
```python
def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility, growth, and lifecycle indicators"""

    df = df.copy()

    # Initialize columns
    df['volume_pct_change'] = 0.0
    df['engagement_growth'] = 0.0
    df['volume_volatility'] = 0.0
    df['momentum'] = 0.0
    df['anomaly_score'] = 0.0
    df['lifecycle_state'] = 'stable'

    for cluster in df['cluster'].unique():
        mask = df['cluster'] == cluster
        cluster_data = df[mask].sort_values('time_window').copy()

        if len(cluster_data) < 2:
            continue

        # Growth metrics
        cluster_data['volume_pct_change'] = cluster_data['post_count'].pct_change()

        total_engagement = (
            cluster_data['total_likes'] +
            cluster_data['total_shares'] +
            cluster_data['total_comments']
        )
        cluster_data['engagement_growth'] = total_engagement.pct_change()

        # Volatility (rolling std)
        cluster_data['volume_volatility'] = cluster_data['post_count'].rolling(window=3, min_periods=1).std()

        # Momentum (deviation from moving average)
        cluster_data['volume_ma'] = cluster_data['post_count'].rolling(window=3, min_periods=1).mean()
        cluster_data['momentum'] = cluster_data['post_count'] - cluster_data['volume_ma']

        # Anomaly score (z-score)
        mean_vol = cluster_data['post_count'].mean()
        std_vol = cluster_data['post_count'].std()
        if std_vol > 0:
            cluster_data['anomaly_score'] = (cluster_data['post_count'] - mean_vol) / std_vol

        # Lifecycle classification
        cluster_data['lifecycle_state'] = classify_lifecycle_vectorized(
            cluster_data['volume_pct_change'],
            cluster_data['post_count'],
            mean_vol
        )

        # Update main dataframe
        df.loc[mask, cluster_data.columns] = cluster_data.values

    return df

def classify_lifecycle_vectorized(pct_change: pd.Series, volume: pd.Series, overall_mean: float) -> pd.Series:
    """Classify lifecycle state for each time period"""

    # Calculate recent metrics (last 3 periods)
    recent_growth = pct_change.rolling(window=3, min_periods=1).mean()

    # Classification logic
    states = pd.Series('stable', index=pct_change.index)
    states[volume < overall_mean * 0.2] = 'dormant'
    states[(volume >= overall_mean * 0.2) & (recent_growth > 0.5)] = 'emerging'
    states[(volume >= overall_mean * 0.2) & (recent_growth > 0.1) & (recent_growth <= 0.5)] = 'trending'
    states[(volume >= overall_mean * 0.2) & (recent_growth < -0.3)] = 'declining'

    return states
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_temporal.py::test_calculate_advanced_metrics_adds_volatility -v`
Expected: PASS

**Step 5: Commit advanced metrics**

```bash
git add scripts/temporal_utils.py tests/test_temporal.py
git commit -m "feat: add advanced temporal metrics

- Calculate volatility, growth rate, momentum
- Add anomaly score for spike detection
- Classify lifecycle states (emerging, trending, stable, declining, dormant)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Script 2 - Temporal Analysis (Part C: Event Detection)

**Files:**
- Modify: `scripts/temporal_utils.py`
- Modify: `tests/test_temporal.py`

**Step 1: Write test for event detection**

Add to `tests/test_temporal.py`:
```python
from scripts.temporal_utils import detect_events

def test_detect_events_finds_spikes():
    """Test that event detection finds anomalies"""
    df = pd.DataFrame({
        'cluster': [0] * 10,
        'cluster_label': ['Test Cluster'] * 10,
        'time_window': pd.date_range('2026-01-01', periods=10, freq='W'),
        'post_count': [10, 12, 11, 50, 13, 12, 10, 60, 11, 10],  # Spikes at idx 3, 7
        'anomaly_score': [0.1, 0.2, 0.1, 3.5, 0.1, 0.2, 0.1, 4.0, 0.1, 0.1]
    })

    events = detect_events(df, spike_threshold=2.0)

    assert len(events) == 2  # Two spikes
    assert 'event_type' in events.columns
    assert all(events['event_type'] == 'spike')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_temporal.py::test_detect_events_finds_spikes -v`
Expected: FAIL with "ImportError: cannot import name 'detect_events'"

**Step 3: Implement event detection**

Add to `scripts/temporal_utils.py`:
```python
def detect_events(df: pd.DataFrame, spike_threshold: float = 2.0) -> pd.DataFrame:
    """Identify significant events (spikes and drops)"""

    # Find anomalies exceeding threshold
    events = df[df['anomaly_score'].abs() > spike_threshold].copy()

    # Classify event type
    events['event_type'] = events['anomaly_score'].apply(
        lambda x: 'spike' if x > 0 else 'drop'
    )

    # Sort by magnitude
    events = events.sort_values('anomaly_score', ascending=False, key=abs)

    return events
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_temporal.py::test_detect_events_finds_spikes -v`
Expected: PASS

**Step 5: Commit event detection**

```bash
git add scripts/temporal_utils.py tests/test_temporal.py
git commit -m "feat: add event detection for spikes and drops

- Detect anomalies above threshold (default 2.0 sigma)
- Classify as spikes or drops
- Sort by magnitude for reporting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Script 2 - Temporal Analysis (Part D: Main Script)

**Files:**
- Create: `scripts/2_temporal_analysis.py`

**Step 1: Write main temporal analysis script**

Create `scripts/2_temporal_analysis.py`:
```python
#!/usr/bin/env python3
"""
Script 2: Temporal Analysis
Analyzes cluster activity over time with volatility metrics
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path
import logging

from temporal_utils import temporal_aggregation, calculate_advanced_metrics, detect_events

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze temporal patterns in clustered data')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--window', help='Time window (overrides config), e.g., "1W", "1M", "1D"')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load clustered data
    clusters_path = config['output']['clusters']
    if not Path(clusters_path).exists():
        raise FileNotFoundError(
            f"{clusters_path} not found. Run 1_cluster_embeddings.py first."
        )

    logger.info(f"Loading clustered data from {clusters_path}...")
    df = pd.read_parquet(clusters_path)
    logger.info(f"Loaded {len(df)} posts across {df['cluster'].nunique()} clusters")

    # Determine time window
    time_window = args.window or config['temporal']['default_window']
    logger.info(f"Using time window: {time_window}")

    # Temporal aggregation
    logger.info("Aggregating by time windows...")
    temporal_df = temporal_aggregation(df, time_window=time_window)

    # Add cluster labels
    cluster_label_map = df[['cluster', 'cluster_label']].drop_duplicates().set_index('cluster')['cluster_label'].to_dict()
    temporal_df['cluster_label'] = temporal_df['cluster'].map(cluster_label_map)

    # Calculate advanced metrics
    logger.info("Calculating advanced metrics (volatility, growth, lifecycle)...")
    temporal_df = calculate_advanced_metrics(temporal_df)

    # Detect events
    logger.info("Detecting events (spikes and drops)...")
    events_df = detect_events(temporal_df, spike_threshold=2.0)

    # Log statistics
    logger.info(f"✓ Analyzed {len(temporal_df)} cluster-time windows")
    logger.info(f"✓ Detected {len(events_df)} significant events")

    lifecycle_counts = temporal_df['lifecycle_state'].value_counts()
    logger.info(f"✓ Lifecycle states: {lifecycle_counts.to_dict()}")

    # Save results
    output_path = config['output']['temporal_stats']
    events_path = config['output']['temporal_events']

    logger.info(f"Saving temporal stats to {output_path}...")
    temporal_df.to_parquet(output_path, index=False)

    logger.info(f"Saving events to {events_path}...")
    events_df.to_parquet(events_path, index=False)

    logger.info(f"✓ Complete! Temporal analysis saved.")

if __name__ == '__main__':
    main()
```

**Step 2: Test script runs**

Run: `python scripts/2_temporal_analysis.py --config config.yaml`
Expected: Script loads clusters, calculates temporal metrics, saves parquet files

**Step 3: Verify output**

Run: `python -c "import pandas as pd; df = pd.read_parquet('output/temporal_stats.parquet'); print(df[['cluster_label', 'time_window', 'post_count', 'lifecycle_state']].head(10))"`
Expected: Shows temporal data with metrics

**Step 4: Make script executable**

Run: `chmod +x scripts/2_temporal_analysis.py`

**Step 5: Commit temporal analysis script**

```bash
git add scripts/2_temporal_analysis.py
git commit -m "feat: add temporal analysis main script

- Aggregate clusters by time windows
- Calculate volatility, growth, lifecycle metrics
- Detect significant events (spikes/drops)
- Support flexible time windows via CLI

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Script 3 - Visualizations (Part A: Cluster Map)

**Files:**
- Create: `scripts/visualization_utils.py`
- Create: `tests/test_visualization.py`

**Step 1: Write test for cluster map creation**

Create `tests/test_visualization.py`:
```python
import pytest
import pandas as pd
import plotly.graph_objects as go
from scripts.visualization_utils import create_cluster_map

def test_create_cluster_map_returns_figure():
    """Test that cluster map creates a valid Plotly figure"""
    df = pd.DataFrame({
        'umap_x': [1, 2, 3, 4, 5],
        'umap_y': [1, 2, 3, 4, 5],
        'cluster': [0, 0, 1, 1, 1],
        'cluster_label': ['Topic A', 'Topic A', 'Topic B', 'Topic B', 'Topic B'],
        'cluster_size': [2, 2, 3, 3, 3],
        'text': ['sample'] * 5
    })

    fig = create_cluster_map(df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.visualization_utils'"

**Step 3: Implement cluster map visualization**

Create `scripts/visualization_utils.py`:
```python
"""Utility functions for visualization pipeline"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

def create_cluster_map(df: pd.DataFrame, title: str = "Topic Clusters - HDBSCAN on Gemini Embeddings") -> go.Figure:
    """Create interactive UMAP scatter plot of clusters"""

    # Sample for hover (show first 100 chars of text)
    df = df.copy()
    df['text_preview'] = df['text'].fillna('').str[:100]

    fig = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color='cluster_label',
        size='cluster_size',
        hover_data={
            'cluster_label': True,
            'cluster_size': True,
            'text_preview': True,
            'umap_x': False,
            'umap_y': False
        },
        title=title,
        width=1200,
        height=800,
        labels={
            'umap_x': 'UMAP Dimension 1',
            'umap_y': 'UMAP Dimension 2',
            'cluster_label': 'Topic'
        }
    )

    fig.update_traces(
        marker=dict(
            opacity=0.6,
            line=dict(width=0.5, color='white')
        )
    )

    fig.update_layout(
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization.py -v`
Expected: PASS

**Step 5: Commit cluster map visualization**

```bash
git add scripts/visualization_utils.py tests/test_visualization.py
git commit -m "feat: add cluster map visualization

- Create interactive UMAP scatter plot with Plotly
- Color by cluster, size by cluster size
- Hover shows label, size, text preview

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Script 3 - Visualizations (Part B: Temporal Heatmap)

**Files:**
- Modify: `scripts/visualization_utils.py`
- Modify: `tests/test_visualization.py`

**Step 1: Write test for heatmap**

Add to `tests/test_visualization.py`:
```python
from scripts.visualization_utils import create_temporal_heatmap

def test_create_temporal_heatmap_returns_figure():
    """Test that temporal heatmap creates a valid Plotly figure"""
    df = pd.DataFrame({
        'cluster_label': ['Topic A', 'Topic A', 'Topic B', 'Topic B'],
        'time_window': pd.date_range('2026-01-01', periods=2, freq='M').tolist() * 2,
        'post_count': [10, 20, 15, 25],
        'lifecycle_state': ['emerging', 'trending', 'stable', 'declining']
    })

    fig = create_temporal_heatmap(df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization.py::test_create_temporal_heatmap_returns_figure -v`
Expected: FAIL with "ImportError: cannot import name 'create_temporal_heatmap'"

**Step 3: Implement temporal heatmap**

Add to `scripts/visualization_utils.py`:
```python
def create_temporal_heatmap(temporal_df: pd.DataFrame) -> go.Figure:
    """Create activity heatmap with lifecycle coloring"""

    # Pivot for heatmap: clusters × time windows
    heatmap_data = temporal_df.pivot(
        index='cluster_label',
        columns='time_window',
        values='post_count'
    ).fillna(0)

    # Sort by total activity
    row_sums = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.loc[row_sums.sort_values(ascending=False).index]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[str(col.date()) for col in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='YlOrRd',
        hovertemplate='%{y}<br>%{x}<br>Posts: %{z}<extra></extra>',
        colorbar=dict(title="Post Count")
    ))

    fig.update_layout(
        title='Topic Activity Over Time',
        xaxis_title='Time Period',
        yaxis_title='Topic Cluster',
        height=max(600, len(heatmap_data) * 20),
        width=1200,
        font=dict(size=10)
    )

    return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization.py::test_create_temporal_heatmap_returns_figure -v`
Expected: PASS

**Step 5: Commit temporal heatmap**

```bash
git add scripts/visualization_utils.py tests/test_visualization.py
git commit -m "feat: add temporal heatmap visualization

- Create heatmap of cluster activity over time
- Sort clusters by total activity
- Dynamic height based on cluster count

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Script 3 - Visualizations (Part C: Event Timeline)

**Files:**
- Modify: `scripts/visualization_utils.py`
- Modify: `tests/test_visualization.py`

**Step 1: Write test for event timeline**

Add to `tests/test_visualization.py`:
```python
from scripts.visualization_utils import create_event_timeline

def test_create_event_timeline_returns_figure():
    """Test that event timeline creates a valid Plotly figure"""
    temporal_df = pd.DataFrame({
        'cluster': [0, 0, 1, 1],
        'cluster_label': ['Topic A', 'Topic A', 'Topic B', 'Topic B'],
        'time_window': pd.date_range('2026-01-01', periods=2, freq='M').tolist() * 2,
        'post_count': [10, 20, 15, 25]
    })

    events_df = pd.DataFrame({
        'cluster_label': ['Topic A'],
        'time_window': [pd.Timestamp('2026-02-01')],
        'post_count': [20],
        'event_type': ['spike']
    })

    fig = create_event_timeline(temporal_df, events_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization.py::test_create_event_timeline_returns_figure -v`
Expected: FAIL with "ImportError: cannot import name 'create_event_timeline'"

**Step 3: Implement event timeline**

Add to `scripts/visualization_utils.py`:
```python
def create_event_timeline(temporal_df: pd.DataFrame, events_df: pd.DataFrame) -> go.Figure:
    """Create timeline with spike annotations"""

    fig = go.Figure()

    # Get top 10 clusters by activity
    top_clusters = temporal_df.groupby('cluster_label')['post_count'].sum().nlargest(10).index

    # Add line for each top cluster
    for cluster_label in top_clusters:
        cluster_data = temporal_df[temporal_df['cluster_label'] == cluster_label].sort_values('time_window')

        fig.add_trace(go.Scatter(
            x=cluster_data['time_window'],
            y=cluster_data['post_count'],
            name=cluster_label,
            mode='lines+markers',
            line=dict(width=2),
            hovertemplate='%{y} posts<extra></extra>'
        ))

    # Add event annotations for spikes
    for _, event in events_df.iterrows():
        if event['cluster_label'] in top_clusters:
            fig.add_annotation(
                x=event['time_window'],
                y=event['post_count'],
                text=f"🔥",
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                arrowwidth=2,
                ax=0,
                ay=-40
            )

    fig.update_layout(
        title='Cluster Activity Timeline with Events',
        xaxis_title='Time',
        yaxis_title='Post Volume',
        hovermode='x unified',
        height=600,
        width=1200,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization.py::test_create_event_timeline_returns_figure -v`
Expected: PASS

**Step 5: Commit event timeline**

```bash
git add scripts/visualization_utils.py tests/test_visualization.py
git commit -m "feat: add event timeline visualization

- Show top 10 clusters over time
- Annotate spikes with flame emoji
- Unified hover mode for easy comparison

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Script 3 - Visualizations (Part D: Main Script)

**Files:**
- Create: `scripts/3_visualize_clusters.py`

**Step 1: Write main visualization script**

Create `scripts/3_visualize_clusters.py`:
```python
#!/usr/bin/env python3
"""
Script 3: Visualize Clusters
Creates interactive visualizations of clusters and temporal patterns
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path
import logging
import webbrowser

from visualization_utils import create_cluster_map, create_temporal_heatmap, create_event_timeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Create interactive visualizations')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--open-browser', action='store_true', help='Open visualizations in browser')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Check required files exist
    clusters_path = config['output']['clusters']
    temporal_path = config['output']['temporal_stats']
    events_path = config['output']['temporal_events']

    if not Path(clusters_path).exists():
        raise FileNotFoundError(f"{clusters_path} not found. Run 1_cluster_embeddings.py first.")

    if not Path(temporal_path).exists():
        raise FileNotFoundError(f"{temporal_path} not found. Run 2_temporal_analysis.py first.")

    # Load data
    logger.info("Loading data...")
    clusters_df = pd.read_parquet(clusters_path)
    temporal_df = pd.read_parquet(temporal_path)
    events_df = pd.read_parquet(events_path) if Path(events_path).exists() else pd.DataFrame()

    logger.info(f"Loaded {len(clusters_df)} posts, {len(temporal_df)} time windows, {len(events_df)} events")

    # Create output directory
    viz_dir = Path(config['output']['viz_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    logger.info("Creating cluster map...")
    cluster_map = create_cluster_map(clusters_df)
    cluster_map_path = viz_dir / 'cluster_map.html'
    cluster_map.write_html(str(cluster_map_path))
    logger.info(f"✓ Saved to {cluster_map_path}")

    logger.info("Creating temporal heatmap...")
    heatmap = create_temporal_heatmap(temporal_df)
    heatmap_path = viz_dir / 'temporal_heatmap.html'
    heatmap.write_html(str(heatmap_path))
    logger.info(f"✓ Saved to {heatmap_path}")

    if len(events_df) > 0:
        logger.info("Creating event timeline...")
        timeline = create_event_timeline(temporal_df, events_df)
        timeline_path = viz_dir / 'event_timeline.html'
        timeline.write_html(str(timeline_path))
        logger.info(f"✓ Saved to {timeline_path}")
    else:
        logger.info("No events detected, skipping timeline")

    logger.info(f"✓ Complete! All visualizations saved to {viz_dir}")

    # Open in browser if requested
    if args.open_browser:
        logger.info("Opening visualizations in browser...")
        webbrowser.open(str(cluster_map_path.absolute()))
        webbrowser.open(str(heatmap_path.absolute()))
        if len(events_df) > 0:
            webbrowser.open(str(timeline_path.absolute()))

if __name__ == '__main__':
    main()
```

**Step 2: Test script runs**

Run: `python scripts/3_visualize_clusters.py --config config.yaml`
Expected: Script creates HTML files in output/visualizations/

**Step 3: Verify outputs**

Run: `ls -lh output/visualizations/`
Expected: See cluster_map.html, temporal_heatmap.html, event_timeline.html

**Step 4: Test opening in browser**

Run: `python scripts/3_visualize_clusters.py --config config.yaml --open-browser`
Expected: Visualizations open in default browser

**Step 5: Make script executable**

Run: `chmod +x scripts/3_visualize_clusters.py`

**Step 6: Commit visualization script**

```bash
git add scripts/3_visualize_clusters.py
git commit -m "feat: add visualization main script

- Create cluster map, temporal heatmap, event timeline
- Save as interactive HTML files
- Optional browser auto-open with --open-browser

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Documentation & README

**Files:**
- Create: `README.md`
- Create: `docs/USAGE.md`

**Step 1: Create README**

Create `README.md`:
```markdown
# Social Intelligence - Temporal Topic Analysis

Analyze 50k+ social media posts with advanced clustering and temporal analysis.

## Features

- 🎯 **HDBSCAN Clustering**: Discover ~50 natural topics from Gemini embeddings
- 📊 **Temporal Analysis**: Track topic evolution with volatility metrics
- 📈 **Interactive Visualizations**: Explore clusters, heatmaps, and timelines
- 🔮 **LLM-Ready**: Prepared for future volatility analysis with Claude

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Step 1: Cluster embeddings (creates ~50 topics)
python scripts/1_cluster_embeddings.py --config config.yaml

# Step 2: Analyze temporal patterns (monthly windows by default)
python scripts/2_temporal_analysis.py --config config.yaml

# Step 3: Create visualizations
python scripts/3_visualize_clusters.py --config config.yaml --open-browser
```

### 3. Explore Outputs

- `output/clusters.parquet` - Clustered posts with UMAP coordinates
- `output/temporal_stats.parquet` - Time-windowed metrics
- `output/visualizations/` - Interactive HTML visualizations

## Configuration

Edit `config.yaml` to customize:

- **Clustering**: `min_cluster_size`, `min_samples` (HDBSCAN sensitivity)
- **Time Windows**: `1D` (daily), `1W` (weekly), `1M` (monthly), `3M` (quarterly)
- **UMAP**: Visualization parameters

## Flexible Time Windows

```bash
# Weekly analysis
python scripts/2_temporal_analysis.py --window "1W"

# Daily analysis
python scripts/2_temporal_analysis.py --window "1D"
```

## Outputs

### Visualizations

- **Cluster Map**: UMAP scatter plot, color-coded by topic
- **Temporal Heatmap**: Activity intensity by cluster over time
- **Event Timeline**: Top clusters with spike annotations

### Metrics

- **Basic**: Post volume, engagement, unique authors
- **Advanced**: Volatility, growth rate, momentum, anomaly scores
- **Lifecycle**: emerging, trending, stable, declining, dormant

## Data Requirements

- `embeddings.npy`: (N, 3072) float32 array from Gemini API
- `clustered_data_labeled.csv`: CSV with columns: text, timestamp, likes, reach, shares, comments, Author, Sentiment

## Architecture

```
scripts/
├── 1_cluster_embeddings.py   # HDBSCAN + UMAP
├── 2_temporal_analysis.py    # Volatility metrics
└── 3_visualize_clusters.py   # Interactive plots

output/
├── clusters.parquet           # Clustered data
├── temporal_stats.parquet     # Metrics by time window
└── visualizations/            # HTML files
```

## Testing

```bash
pytest tests/ -v
```

## Future: LLM Volatility Analysis

The pipeline prepares data for LLM-powered volatility analysis:

```python
# Future: Script 4
python scripts/4_llm_volatility_analysis.py --config config.yaml
```

Enable in `config.yaml`:
```yaml
llm:
  enabled: true
  provider: "anthropic"
```

## License

MIT
```

**Step 2: Create usage documentation**

Create `docs/USAGE.md`:
```markdown
# Usage Guide

## Running the Pipeline

### Full Pipeline

```bash
# Run all three scripts sequentially
python scripts/1_cluster_embeddings.py --config config.yaml
python scripts/2_temporal_analysis.py --config config.yaml
python scripts/3_visualize_clusters.py --config config.yaml --open-browser
```

### Resume from Cache

Scripts automatically skip if output exists:

```bash
# This will skip clustering if clusters.parquet exists
python scripts/1_cluster_embeddings.py

# Force rerun
python scripts/1_cluster_embeddings.py --force
```

## Configuration Examples

### Experiment with Different Time Windows

```bash
# Daily granularity
python scripts/2_temporal_analysis.py --window "1D"

# Weekly (good for detecting weekly trends)
python scripts/2_temporal_analysis.py --window "1W"

# Quarterly (long-term patterns)
python scripts/2_temporal_analysis.py --window "3M"
```

### Adjust Clustering Sensitivity

Edit `config.yaml`:

```yaml
clustering:
  min_cluster_size: 30   # Smaller = more clusters (more granular)
  min_samples: 5         # Smaller = more sensitive (finds smaller clusters)
```

## Reading Outputs

### Clusters

```python
import pandas as pd

df = pd.read_parquet('output/clusters.parquet')
print(df[['cluster', 'cluster_label', 'text']].head())

# Check cluster distribution
print(df['cluster'].value_counts())
```

### Temporal Stats

```python
temporal = pd.read_parquet('output/temporal_stats.parquet')
print(temporal[['cluster_label', 'time_window', 'post_count', 'lifecycle_state']].head())

# Find emerging topics
emerging = temporal[temporal['lifecycle_state'] == 'emerging']
print(emerging['cluster_label'].unique())
```

### Events

```python
events = pd.read_parquet('output/temporal_events.parquet')
print(events[['cluster_label', 'time_window', 'event_type', 'anomaly_score']].head())
```

## Troubleshooting

### "Mismatch: X embeddings vs Y rows"

Ensure `embeddings.npy` and CSV have the same number of rows:

```python
import numpy as np
import pandas as pd

embeddings = np.load('embeddings.npy')
df = pd.read_csv('clustered_data_labeled.csv')
print(f"Embeddings: {len(embeddings)}, CSV: {len(df)}")
```

### "Missing required columns"

Check your CSV has required columns:

```python
df = pd.read_csv('clustered_data_labeled.csv')
required = ['text', 'timestamp', 'likes', 'reach']
print(f"Missing: {set(required) - set(df.columns)}")
```

### Low cluster count (<30 clusters)

Increase sensitivity:

```yaml
clustering:
  min_cluster_size: 30  # Decrease from 50
  min_samples: 5        # Decrease from 10
```

## Performance

- **Script 1** (Clustering): ~15-30 min for 50k posts
- **Script 2** (Temporal): ~2-5 min
- **Script 3** (Visualization): ~1-2 min

Use `--force` only when needed to avoid re-running expensive clustering.
```

**Step 3: Commit documentation**

```bash
git add README.md docs/USAGE.md
git commit -m "docs: add README and usage guide

- Add quick start instructions
- Document configuration options
- Add troubleshooting section
- Explain output formats and metrics

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:
```python
"""Integration test for full pipeline"""
import pytest
import subprocess
import pandas as pd
from pathlib import Path

def test_full_pipeline_executes():
    """Test that all three scripts run successfully"""

    # Script 1: Clustering
    result = subprocess.run(
        ['python', 'scripts/1_cluster_embeddings.py', '--config', 'config.yaml'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script 1 failed: {result.stderr}"
    assert Path('output/clusters.parquet').exists()

    # Verify clusters output
    df = pd.read_parquet('output/clusters.parquet')
    assert 'cluster' in df.columns
    assert 'umap_x' in df.columns
    assert 'cluster_label' in df.columns

    # Script 2: Temporal Analysis
    result = subprocess.run(
        ['python', 'scripts/2_temporal_analysis.py', '--config', 'config.yaml'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script 2 failed: {result.stderr}"
    assert Path('output/temporal_stats.parquet').exists()

    # Verify temporal stats
    temporal = pd.read_parquet('output/temporal_stats.parquet')
    assert 'volume_volatility' in temporal.columns
    assert 'lifecycle_state' in temporal.columns

    # Script 3: Visualization
    result = subprocess.run(
        ['python', 'scripts/3_visualize_clusters.py', '--config', 'config.yaml'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script 3 failed: {result.stderr}"
    assert Path('output/visualizations/cluster_map.html').exists()
    assert Path('output/visualizations/temporal_heatmap.html').exists()
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v -s`
Expected: All three scripts run successfully, outputs created

**Step 3: Verify all outputs**

Run: `ls -lh output/ output/visualizations/`
Expected: See all parquet files and HTML visualizations

**Step 4: Commit integration test**

```bash
git add tests/test_integration.py
git commit -m "test: add full pipeline integration test

- Test all three scripts run successfully
- Verify outputs are created with expected schemas
- Ensure end-to-end workflow completes

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

✅ **All scripts execute without errors**
✅ **Clustering produces ~50 clusters** (HDBSCAN finds natural boundaries)
✅ **Temporal analysis calculates all metrics** (volatility, growth, lifecycle)
✅ **Visualizations render in browser** (interactive HTML files)
✅ **Tests pass** (unit tests + integration test)
✅ **Documentation complete** (README + usage guide)

---

## Next Steps After Implementation

1. **Run on full dataset** (50k posts)
2. **Tune HDBSCAN parameters** if cluster count is off
3. **Experiment with time windows** (daily, weekly, monthly)
4. **Prepare for Script 4** (LLM volatility analysis)
5. **Share visualizations** (HTML files are standalone and shareable)

---

**End of Implementation Plan**
