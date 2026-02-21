"""
Script 4: Theme Activation (No LLM)

Computes theme activations using a curated lexicon + TF-IDF:
- Builds per-cluster/time-window documents from clusters.parquet
- Scores each theme by summing TF-IDF weights of matched lexicon terms
- Activates themes via threshold(s)
- Saves mode-specific parquet output
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import yaml
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.io_utils import read_parquet_safe
else:
    from .io_utils import read_parquet_safe


EXTRA_STOPWORDS = {
    "im", "ive", "id", "youre", "theyre", "weve", "dont", "didnt", "doesnt",
    "cant", "couldnt", "wouldnt", "wont", "isnt", "arent", "wasnt", "werent",
    "thats", "theres", "here", "also", "via", "etc", "amp"
}
STOPWORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_clusters_with_fallback(clusters_parquet_path, clusters_csv_path=None):
    """Prefer clusters parquet; fallback to CSV if parquet is unreadable."""
    try:
        return read_parquet_safe(clusters_parquet_path)
    except Exception as e:
        if clusters_csv_path and os.path.exists(clusters_csv_path):
            print(
                "WARN: Could not read clusters parquet; falling back to CSV: "
                f"{clusters_csv_path} ({e})"
            )
            return pd.read_csv(clusters_csv_path, low_memory=False)
        raise


def _try_stem(word):
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return stemmer.stem(word)
    except Exception:
        return word


def normalize_text(text, do_stem=False):
    text = "" if text is None else str(text).lower()
    tokens = TOKEN_RE.findall(text)
    cleaned = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if do_stem:
            tok = _try_stem(tok)
        if tok and tok not in STOPWORDS:
            cleaned.append(tok)
    return " ".join(cleaned)


def _normalize_worker(item):
    text, do_stem = item
    return normalize_text(text, do_stem=do_stem)


def normalize_corpus(texts, do_stem=False, n_jobs=1, min_parallel_docs=2000):
    if n_jobs <= 1 or len(texts) < min_parallel_docs:
        return [normalize_text(t, do_stem=do_stem) for t in texts]

    workers = min(n_jobs, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_normalize_worker, [(t, do_stem) for t in texts], chunksize=200))


def load_theme_lexicon(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    themes = data.get("themes", [])
    if not themes:
        raise ValueError(f"No themes found in lexicon file: {path}")
    return themes


def normalize_theme_terms(theme, do_stem=False):
    terms = []
    for field in ["keywords", "synonyms", "phrases"]:
        vals = theme.get(field, []) or []
        terms.extend(vals)
    if not terms:
        terms = [theme.get("name", "")]
    normalized = []
    for t in terms:
        nt = normalize_text(t, do_stem=do_stem)
        if nt:
            normalized.append(nt)
    return sorted(set(normalized))


def get_time_window_freq(config, arg_window):
    window = arg_window or config["temporal"]["default_window"]
    # Pandas deprecates "M" in favor of "ME".
    if window == "1M":
        return "1ME"
    return window


def build_documents(clusters_df, text_col, ts_col, freq):
    df = clusters_df.copy()
    if text_col not in df.columns:
        raise ValueError(f"Missing text column in clusters data: {text_col}")
    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp column in clusters data: {ts_col}")
    if "cluster" not in df.columns:
        raise ValueError("clusters.parquet missing required column: cluster")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.set_index(ts_col)

    grouped = (
        df.groupby(["cluster", pd.Grouper(freq=freq)])
        .agg(
            document=(text_col, lambda x: " ".join(x.fillna("").astype(str))),
            source_posts=("cluster", "count"),
            cluster_label=("cluster_label", "first") if "cluster_label" in df.columns else ("cluster", "first"),
        )
        .reset_index()
        .rename(columns={ts_col: "time_window"})
    )
    return grouped


def resolve_targets(mode, stats_df, events_df):
    if mode == "windows":
        targets = stats_df.copy()
    elif mode == "events":
        targets = events_df.copy()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if "cluster" not in targets.columns or "time_window" not in targets.columns:
        raise ValueError(f"{mode} data must include 'cluster' and 'time_window' columns")
    targets["time_window"] = pd.to_datetime(targets["time_window"], errors="coerce")
    targets = targets.dropna(subset=["time_window"]).copy()
    return targets


def windows_targets_fallback(docs_df):
    out = docs_df[["cluster", "time_window", "cluster_label"]].copy()
    out["post_count"] = docs_df["source_posts"].astype(int)
    return out


def compute_theme_scores(vectorizer, X, themes, do_stem=False, n_jobs=1):
    vocab = vectorizer.vocabulary_
    theme_terms = {}
    theme_indices = {}
    for theme in themes:
        tid = theme.get("id")
        if tid is None:
            raise ValueError("Each theme must have an 'id'.")
        terms = normalize_theme_terms(theme, do_stem=do_stem)
        idxs = [vocab[t] for t in terms if t in vocab]
        theme_terms[tid] = terms
        theme_indices[tid] = sorted(set(idxs))

    def score_one(item):
        tid, idxs = item
        if not idxs:
            return tid, [0.0] * X.shape[0]
        arr = X[:, idxs].sum(axis=1).A1
        return tid, arr.tolist()

    items = list(theme_indices.items())
    scores = {}
    if n_jobs > 1 and len(items) > 3:
        workers = min(n_jobs, os.cpu_count() or 1, len(items))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for tid, vals in ex.map(score_one, items):
                scores[tid] = vals
    else:
        for it in items:
            tid, vals = score_one(it)
            scores[tid] = vals

    return scores, theme_terms, theme_indices


def threshold_for_theme(theme, cfg):
    per_theme = cfg.get("per_theme_threshold", {}) or {}
    tid = str(theme["id"])
    return float(per_theme.get(tid, cfg.get("global_threshold", 0.03)))


def main():
    parser = argparse.ArgumentParser(description="Theme activation with TF-IDF (no LLM)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default=None, choices=["windows", "events"])
    parser.add_argument("--window", default=None, help="Pandas freq string; defaults to config.temporal.default_window")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for large corpora")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(config_path)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(config_dir, path)

    ta_cfg = config.get("theme_activation", {})
    mode = args.mode or ta_cfg.get("mode", "windows")
    freq = get_time_window_freq(config, args.window)
    n_jobs = args.n_jobs if args.n_jobs is not None else int(ta_cfg.get("n_jobs", 1))
    do_stem = bool(ta_cfg.get("preprocessing", {}).get("stem", False))
    min_df = int(ta_cfg.get("tfidf", {}).get("min_df", 2))
    max_df = float(ta_cfg.get("tfidf", {}).get("max_df", 0.9))
    max_features = int(ta_cfg.get("tfidf", {}).get("max_features", 50000))
    ngram = ta_cfg.get("tfidf", {}).get("ngram_range", [1, 2])
    min_parallel_docs = int(ta_cfg.get("preprocessing", {}).get("min_parallel_docs", 2000))

    clusters_path = resolve(config["output"]["clusters"])
    stats_path = resolve(config["output"]["temporal_stats"])
    events_path = resolve(config["output"]["temporal_events"])
    clusters_csv_path = resolve(config["input"]["data"])
    lexicon_path = resolve(ta_cfg.get("theme_lexicon", "theme_lexicon.yaml"))
    out_base = resolve(ta_cfg.get("output", "output/theme_activations.parquet"))
    out_path = out_base.replace(".parquet", f"_{mode}.parquet")

    if os.path.exists(out_path) and not args.force:
        print(f"Output exists: {out_path} (use --force to rerun)")
        return

    if not os.path.exists(clusters_path) and not os.path.exists(clusters_csv_path):
        raise FileNotFoundError(
            f"Required input missing: {clusters_path} and fallback CSV {clusters_csv_path}"
        )
    if not os.path.exists(stats_path):
        print(f"WARN: Missing {stats_path}; windows mode will fallback to derived targets.")
    if mode == "events" and not os.path.exists(events_path):
        raise FileNotFoundError(f"Required input missing for events mode: {events_path}")
    if not os.path.exists(lexicon_path):
        raise FileNotFoundError(f"Theme lexicon not found: {lexicon_path}")

    print(f"Loading inputs (mode={mode}, window={freq})...")
    clusters_df = read_clusters_with_fallback(clusters_path, clusters_csv_path=clusters_csv_path)
    try:
        stats_df = read_parquet_safe(stats_path) if os.path.exists(stats_path) else pd.DataFrame()
    except Exception as e:
        print(f"WARN: Could not read temporal stats parquet: {e}")
        stats_df = pd.DataFrame()
    try:
        events_df = read_parquet_safe(events_path) if os.path.exists(events_path) else pd.DataFrame()
    except Exception as e:
        print(f"WARN: Could not read temporal events parquet: {e}")
        events_df = pd.DataFrame()
    themes = load_theme_lexicon(lexicon_path)
    print(f"Loaded {len(themes)} themes from {lexicon_path}")

    text_col = config["input"]["text_column"]
    ts_col = config["input"]["timestamp_column"]
    docs_df = build_documents(clusters_df, text_col=text_col, ts_col=ts_col, freq=freq)
    if mode == "windows" and stats_df.empty:
        print("Using fallback windows targets derived from clusters/time windows.")
        targets_df = windows_targets_fallback(docs_df)
    else:
        targets_df = resolve_targets(mode, stats_df=stats_df, events_df=events_df)

    merged = targets_df.merge(
        docs_df[["cluster", "time_window", "document", "source_posts", "cluster_label"]],
        on=["cluster", "time_window"],
        how="left",
        suffixes=("", "_doc"),
    )
    merged["document"] = merged["document"].fillna("")
    merged["source_posts"] = merged["source_posts"].fillna(0).astype(int)
    if "cluster_label" not in merged.columns and "cluster_label_doc" in merged.columns:
        merged["cluster_label"] = merged["cluster_label_doc"]

    print(f"Scoring {len(merged):,} units...")
    merged["document_normalized"] = normalize_corpus(
        merged["document"].tolist(),
        do_stem=do_stem,
        n_jobs=n_jobs,
        min_parallel_docs=min_parallel_docs,
    )
    merged["document_token_count"] = merged["document_normalized"].str.split().str.len().fillna(0).astype(int)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(int(ngram[0]), int(ngram[1])),
    )
    X = vectorizer.fit_transform(merged["document_normalized"].tolist())

    scores, theme_terms, theme_indices = compute_theme_scores(
        vectorizer=vectorizer,
        X=X,
        themes=themes,
        do_stem=do_stem,
        n_jobs=n_jobs,
    )

    explain_terms = {}
    for theme in themes:
        tid = theme["id"]
        tcol = f"theme_{tid}"
        scol = f"{tcol}_score"
        threshold = threshold_for_theme(theme, ta_cfg)
        merged[scol] = scores.get(tid, [0.0] * len(merged))
        merged[tcol] = (merged[scol] >= threshold).astype(int)
        explain_terms[tcol] = {
            "matched_terms": [t for t in theme_terms.get(tid, []) if t in vectorizer.vocabulary_],
            "matched_feature_count": len(theme_indices.get(tid, [])),
            "threshold": threshold,
        }

    merged["activation_reasoning"] = (
        "TF-IDF weighted lexicon matching with stopword filtering and configurable thresholds."
    )
    merged["activation_explain"] = json.dumps(explain_terms, ensure_ascii=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_parquet(out_path, index=False)
    elapsed = time.time() - started
    active_cols = [f"theme_{t['id']}" for t in themes]
    avg_active = float(merged[active_cols].sum(axis=1).mean()) if active_cols else 0.0

    print(f"Saved {len(merged):,} rows to {out_path}")
    print(f"Avg activated themes per unit: {avg_active:.2f}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
