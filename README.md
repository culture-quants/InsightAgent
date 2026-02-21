# InsightAgent — Temporal Topic Analysis Pipeline

Analyses 50k social media posts with Gemini embeddings to discover topics,
track their evolution over time, and visualise volatility.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run scripts in order. Each caches its output so downstream scripts are fast.

### 1. Cluster Embeddings

```bash
python scripts/1_cluster_embeddings.py --config config.yaml
python scripts/1_cluster_embeddings.py --config config.yaml --force  # rerun
```

Outputs: `output/clusters.parquet`

### 2. Temporal Analysis

```bash
python scripts/2_temporal_analysis.py                      # monthly (default)
python scripts/2_temporal_analysis.py --window "1W"        # weekly
python scripts/2_temporal_analysis.py --window "1D"        # daily
python scripts/2_temporal_analysis.py --window "3M"        # quarterly
```

Outputs: `output/temporal_stats.parquet`, `output/temporal_events.parquet`,
`output/cluster_trend_snapshots_1M.parquet`, `output/cluster_trend_snapshots_1M.csv`

### 3. Visualizations

```bash
python scripts/3_visualize_clusters.py
python scripts/3_visualize_clusters.py --open-browser      # auto-open in browser
```

Outputs (standalone HTML, no Python needed to view):
- `output/visualizations/cluster_map.html`
- `output/visualizations/temporal_heatmap.html`
- `output/visualizations/event_timeline.html`
- `output/visualizations/volatility_dashboard.html`

### 4. Theme Activation (No LLM, TF-IDF)

This optional step computes deterministic theme activations from a curated lexicon.

```bash
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode events
python scripts/4_theme_activation_tfidf.py --config config.yaml --mode windows --n-jobs 8 --force
```

Outputs:
- `output/theme_activations_windows.parquet`
- `output/theme_activations_events.parquet`

Theme definitions are loaded from:
- `theme_lexicon.yaml` (replace sample themes with your full 50-theme lexicon)

### 5. Event Direction Forecast

Builds a per-cluster historical state database (volatility + direction signals),
then predicts likely direction for a new event text.

```bash
python scripts/5_event_direction_forecast.py --config config.yaml
python scripts/5_event_direction_forecast.py --config config.yaml --event-text "Google launches new Gemini robotics model"
python scripts/5_event_direction_forecast.py --config config.yaml --event-text "Google launches new Gemini robotics model" --model-path output/models/direction_model.pkl --force
```

Outputs:
- `output/cluster_state_database.parquet`
- `output/event_direction_prediction.json`

### 6. Train Direction Model

Train supervised next-window direction models (Logistic + RandomForest baseline),
evaluate against heuristic, and save the best artifact.

```bash
python scripts/6_train_direction_model.py --config config.yaml --force
```

Outputs:
- `output/models/direction_model.pkl`
- `output/models/direction_model_metrics.json`

### 6b. Forecast Dashboard

Generate a standalone HTML dashboard for the latest forecast output.

```bash
python scripts/6b_visualize_forecast.py --config config.yaml
```

Output:
- `output/visualizations/forecast_dashboard.html`

## Configuration

Edit `config.yaml` to adjust:
- `clustering.min_cluster_size` — minimum posts per cluster (default: 50)
- `temporal.default_window` — Pandas frequency string (default: `1M`)
- `output.cluster_trend_snapshots_1m` / `output.cluster_trend_snapshots_1m_csv` — frontend-ready monthly snapshots
- `llm.enabled` — enable Claude-powered volatility analysis (future)
- `llm.enabled` — enable optional Gemini semantic rerank for theme/event matching
- `theme_activation.*` — TF-IDF theme scoring mode, thresholds, and parallelism
- `model_training.*` — direction model selection, split, epsilon, and output paths
- Built-in strong baselines include `random_forest`, `extra_trees`, `hist_gbm`, and `logistic`
- Optional boosted models include `xgboost`, `lightgbm`, and `catboost` (auto-skip if not installed)
- `model_training.walk_forward_*` — walk-forward CV controls
- `model_training.target_epsilon_grid` — epsilon tuning candidates
- `model_training.class_threshold_candidates` — class probability threshold tuning
- `model_training.calibrate_probabilities` — probability calibration toggle
- `event_forecast.risk_band_quantiles` — risk bucket thresholds

## Data Files and Git

Large local artifacts are intentionally excluded from version control using `.gitignore`,
including:

- `clustered_data_labeled.csv`
- `embeddings.npy`
- `catboost_info/`

Keep these files locally for pipeline runs; do not commit them to GitHub.
