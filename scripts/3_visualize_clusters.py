"""
Script 3: Interactive Visualizations
Creates 4 standalone Plotly HTML dashboards:
  1. cluster_map.html         - UMAP scatter coloured by cluster
  2. temporal_heatmap.html    - clusters × time, colour = post volume
  3. event_timeline.html      - multi-line volume chart with spike annotations
  4. volatility_dashboard.html - 4-panel metrics overview
"""

import argparse
import os
import sys
import webbrowser

import pandas as pd
import yaml


LIFECYCLE_BADGES = {
    'emerging':  '🟢',
    'trending':  '🔥',
    'stable':    '🔵',
    'declining': '🔻',
    'dormant':   '⚫',
}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def cluster_map(clusters_df, out_path):
    import plotly.express as px

    df = clusters_df.copy()
    df['cluster_str'] = df['cluster'].astype(str)

    # Sample for readability if very large
    sample = df if len(df) <= 20000 else df.sample(20000, random_state=42)

    text_col = 'text' if 'text' in sample.columns else sample.columns[0]
    hover_data = {}
    for col in ['cluster_label', 'cluster_size', 'likes', 'reach']:
        if col in sample.columns:
            hover_data[col] = True

    fig = px.scatter(
        sample, x='umap_x', y='umap_y',
        color='cluster_str',
        size='cluster_size' if 'cluster_size' in sample.columns else None,
        size_max=20,
        hover_name=text_col,
        hover_data=hover_data,
        title='Cluster Map (UMAP)',
        labels={'cluster_str': 'Cluster'},
        template='plotly_dark',
        opacity=0.7,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(legend_title_text='Cluster', height=700)
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def temporal_heatmap(temporal_df, out_path):
    import plotly.graph_objects as go

    df = temporal_df.copy()
    df['time_window'] = pd.to_datetime(df['time_window'])

    pivot = df.pivot_table(index='cluster_label', columns='time_window', values='post_count', fill_value=0)

    # Lifecycle badge per cluster
    lifecycle = df.groupby('cluster_label')['lifecycle_state'].first()
    y_labels = [f"{LIFECYCLE_BADGES.get(lifecycle.get(lbl, 'stable'), '')} {lbl}" for lbl in pivot.index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c.date()) for c in pivot.columns],
        y=y_labels,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Cluster: %{y}<br>Period: %{x}<br>Posts: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title='Topic Activity Heatmap (posts per period)',
        xaxis_title='Time Period',
        yaxis_title='Cluster',
        template='plotly_dark',
        height=max(400, len(pivot) * 20),
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def event_timeline(temporal_df, events_df, out_path):
    import plotly.graph_objects as go

    df = temporal_df.copy()
    df['time_window'] = pd.to_datetime(df['time_window'])
    events = events_df.copy()
    if not events.empty:
        events['time_window'] = pd.to_datetime(events['time_window'])

    fig = go.Figure()

    top_clusters = df.groupby('cluster')['post_count'].sum().nlargest(15).index
    for cluster_id in top_clusters:
        cdf = df[df['cluster'] == cluster_id].sort_values('time_window')
        label = cdf['cluster_label'].iloc[0] if len(cdf) else str(cluster_id)
        fig.add_trace(go.Scatter(
            x=cdf['time_window'], y=cdf['post_count'],
            mode='lines', name=label, hovertemplate='%{y} posts<br>%{x}<extra>' + label + '</extra>'
        ))

    # Spike annotations
    if not events.empty:
        spikes = events[events['event_type'] == 'spike']
        fig.add_trace(go.Scatter(
            x=spikes['time_window'], y=spikes['post_count'],
            mode='markers', name='Spikes',
            marker=dict(color='red', size=10, symbol='triangle-up'),
            hovertemplate='SPIKE: %{y} posts<br>%{x}<extra></extra>'
        ))

    fig.update_layout(
        title='Topic Event Timeline (top 15 clusters)',
        xaxis_title='Time',
        yaxis_title='Post Volume',
        template='plotly_dark',
        height=600,
        legend=dict(orientation='v', x=1.01),
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def volatility_dashboard(temporal_df, out_path):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = temporal_df.copy()
    df['time_window'] = pd.to_datetime(df['time_window'])

    cluster_summary = df.groupby('cluster_label').agg(
        avg_volatility=('volume_volatility', 'mean'),
        avg_growth=('volume_pct_change', 'mean'),
        avg_market_share=('market_share', 'mean'),
        lifecycle=('lifecycle_state', lambda x: x.mode()[0] if len(x) > 0 else 'stable'),
    ).reset_index().sort_values('avg_volatility', ascending=False).head(20)

    lifecycle_counts = df.groupby('cluster')['lifecycle_state'].first().value_counts().reset_index()
    lifecycle_counts.columns = ['lifecycle_state', 'count']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Volume Volatility by Cluster (top 20)',
            'Avg Growth Rate by Cluster',
            'Avg Market Share by Cluster',
            'Lifecycle State Distribution',
        ]
    )

    # Panel 1: Volatility
    fig.add_trace(go.Bar(
        x=cluster_summary['avg_volatility'],
        y=cluster_summary['cluster_label'],
        orientation='h', name='Volatility',
        marker_color='orange',
    ), row=1, col=1)

    # Panel 2: Growth rate
    colors = ['green' if g >= 0 else 'red' for g in cluster_summary['avg_growth']]
    fig.add_trace(go.Bar(
        x=cluster_summary['avg_growth'],
        y=cluster_summary['cluster_label'],
        orientation='h', name='Growth Rate',
        marker_color=colors,
    ), row=1, col=2)

    # Panel 3: Market share
    fig.add_trace(go.Bar(
        x=cluster_summary['avg_market_share'],
        y=cluster_summary['cluster_label'],
        orientation='h', name='Market Share',
        marker_color='steelblue',
    ), row=2, col=1)

    # Panel 4: Lifecycle pie
    fig.add_trace(go.Pie(
        labels=lifecycle_counts['lifecycle_state'],
        values=lifecycle_counts['count'],
        name='Lifecycle',
    ), row=2, col=2)

    fig.update_layout(
        title='Volatility Dashboard',
        template='plotly_dark',
        height=900,
        showlegend=False,
    )
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive cluster visualizations")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    viz_dir = config['output']['viz_dir']
    os.makedirs(viz_dir, exist_ok=True)

    clusters_path = config['output']['clusters']
    stats_path = config['output']['temporal_stats']
    events_path = config['output']['temporal_events']

    for path in [clusters_path, stats_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run scripts 1 and 2 first.")
            sys.exit(1)

    print("Loading data...")
    clusters_df = pd.read_parquet(clusters_path)
    temporal_df = pd.read_parquet(stats_path)
    events_df = pd.read_parquet(events_path) if os.path.exists(events_path) else pd.DataFrame()

    print("Generating visualizations...")
    paths = {
        'cluster_map': os.path.join(viz_dir, 'cluster_map.html'),
        'temporal_heatmap': os.path.join(viz_dir, 'temporal_heatmap.html'),
        'event_timeline': os.path.join(viz_dir, 'event_timeline.html'),
        'volatility_dashboard': os.path.join(viz_dir, 'volatility_dashboard.html'),
    }

    cluster_map(clusters_df, paths['cluster_map'])
    temporal_heatmap(temporal_df, paths['temporal_heatmap'])
    event_timeline(temporal_df, events_df, paths['event_timeline'])
    volatility_dashboard(temporal_df, paths['volatility_dashboard'])

    print("\nAll visualizations generated:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if args.open_browser:
        webbrowser.open(f"file://{os.path.abspath(paths['cluster_map'])}")


if __name__ == '__main__':
    main()
