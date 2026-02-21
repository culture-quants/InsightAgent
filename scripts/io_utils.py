import os

import pandas as pd


def read_parquet_safe(path):
    """Read parquet with a legacy pyarrow fallback for occasional metadata issues."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        err = str(exc).lower()
        if any(token in err for token in ("histogram", "repetition", "mismatch")):
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(path, use_legacy_dataset=True)
                return table.to_pandas()
            except Exception as legacy_exc:
                raise RuntimeError(
                    f"Could not read parquet with fallback: {path} ({legacy_exc})"
                ) from exc
        raise RuntimeError(f"Could not read parquet: {path} ({exc})") from exc


def normalize_window_freq(window):
    return "1ME" if window == "1M" else window


def build_temporal_fallback_from_clusters_csv(csv_path, ts_col, window):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fallback CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if "cluster" not in df.columns:
        raise ValueError("Fallback CSV must contain a 'cluster' column.")
    if ts_col not in df.columns:
        raise ValueError(f"Fallback CSV missing timestamp column: {ts_col}")

    df = df[df["cluster"] != -1].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col)
    freq = normalize_window_freq(window)

    rows = []
    for cluster_id, cdf in df.groupby("cluster"):
        resampled = cdf.resample(freq).agg(post_count=("cluster", "count")).fillna(0)
        resampled["cluster"] = cluster_id
        resampled["volume_pct_change"] = resampled["post_count"].pct_change().fillna(0)
        resampled["volume_volatility"] = (
            resampled["post_count"].rolling(3, min_periods=1).std().fillna(0)
        )
        rolling_mean = resampled["post_count"].rolling(3, min_periods=1).mean()
        resampled["momentum"] = resampled["post_count"] - rolling_mean
        rows.append(resampled.reset_index().rename(columns={ts_col: "time_window"}))

    if not rows:
        return pd.DataFrame(
            columns=[
                "time_window",
                "cluster",
                "post_count",
                "volume_pct_change",
                "volume_volatility",
                "momentum",
                "market_share",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    total_per_window = out.groupby("time_window")["post_count"].transform("sum")
    out["market_share"] = out["post_count"] / total_per_window.replace(0, pd.NA)
    out["market_share"] = out["market_share"].fillna(0.0)
    return out
