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

Outputs: `output/temporal_stats.parquet`, `output/temporal_events.parquet`

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

## Configuration

Edit `config.yaml` to adjust:
- `clustering.min_cluster_size` — minimum posts per cluster (default: 50)
- `temporal.default_window` — Pandas frequency string (default: `1M`)
- `llm.enabled` — enable Claude-powered volatility analysis (future)
