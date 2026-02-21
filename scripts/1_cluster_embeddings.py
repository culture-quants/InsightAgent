"""
Script 1: Cluster Embeddings
- Loads Gemini embeddings (50k x 3072) + CSV data
- Runs HDBSCAN clustering
- Generates UMAP 2D coordinates
- Creates TF-IDF cluster labels
- Saves output/clusters.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_inputs(embeddings_path, data_path, config):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    print("Loading data for validation...")
    embeddings = np.load(embeddings_path)
    data = pd.read_csv(data_path)

    if len(embeddings) != len(data):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(data)} rows")

    text_col = config['input']['text_column']
    ts_col = config['input']['timestamp_column']
    required_cols = [text_col, ts_col]
    missing = set(required_cols) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate timestamps
    pd.to_datetime(data[ts_col], errors='coerce')

    return embeddings, data


def generate_cluster_labels(df, text_column, n_top_terms=3):
    """Generate TF-IDF keyword labels per cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    labels = {}
    unique_clusters = sorted(df['cluster'].unique())

    print("Generating cluster labels via TF-IDF...")
    for cluster_id in tqdm(unique_clusters):
        if cluster_id == -1:
            labels[-1] = "Uncategorized/Noise"
            continue

        cluster_texts = df[df['cluster'] == cluster_id][text_column].fillna("").tolist()
        if not cluster_texts:
            labels[cluster_id] = f"Cluster {cluster_id}"
            continue

        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', max_df=0.9, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-n_top_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            labels[cluster_id] = ", ".join(top_terms)
        except Exception:
            labels[cluster_id] = f"Cluster {cluster_id}"

    return labels


def run_clustering(embeddings, config):
    import hdbscan
    import umap

    print(f"Running HDBSCAN on {len(embeddings)} embeddings...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        min_samples=config['clustering']['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(cluster_labels):.1f}%)")

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_components=config['umap']['n_components'],
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist'],
        metric=config['umap']['metric'],
        verbose=True
    )
    umap_coords = reducer.fit_transform(embeddings)

    return cluster_labels, umap_coords


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings with HDBSCAN + UMAP")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--force', action='store_true', help='Rerun even if output exists')
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = config['output']['clusters']

    if os.path.exists(output_path) and not args.force:
        print(f"Output already exists: {output_path} (use --force to rerun)")
        sys.exit(0)

    embeddings, data = validate_inputs(
        config['input']['embeddings'],
        config['input']['data'],
        config
    )

    cluster_labels, umap_coords = run_clustering(embeddings, config)

    data['cluster'] = cluster_labels
    data['umap_x'] = umap_coords[:, 0]
    data['umap_y'] = umap_coords[:, 1]

    text_col = config['input']['text_column']
    cluster_label_map = generate_cluster_labels(data, text_col)
    data['cluster_label'] = data['cluster'].map(cluster_label_map)

    cluster_sizes = data['cluster'].value_counts()
    data['cluster_size'] = data['cluster'].map(cluster_sizes)

    os.makedirs(config['output']['dir'], exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Saved {len(data)} rows to {output_path}")


if __name__ == '__main__':
    main()
