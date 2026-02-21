import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TemporalSnapshotExportTests(unittest.TestCase):
    def test_temporal_analysis_exports_frontend_monthly_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "output")
            os.makedirs(output_dir, exist_ok=True)

            clusters_path = os.path.join(output_dir, "clusters.parquet")
            temporal_stats_path = os.path.join(output_dir, "temporal_stats.parquet")
            temporal_events_path = os.path.join(output_dir, "temporal_events.parquet")
            snapshots_1m_path = os.path.join(output_dir, "cluster_trend_snapshots_1M.parquet")
            snapshots_1m_csv_path = os.path.join(output_dir, "cluster_trend_snapshots_1M.csv")

            df = pd.DataFrame(
                [
                    {"cluster": 0, "cluster_label": "A", "timestamp": "2025-01-15", "Author": "u1", "likes": 10},
                    {"cluster": 0, "cluster_label": "A", "timestamp": "2025-02-15", "Author": "u2", "likes": 12},
                    {"cluster": 1, "cluster_label": "B", "timestamp": "2025-01-20", "Author": "u3", "likes": 8},
                    {"cluster": 1, "cluster_label": "B", "timestamp": "2025-02-20", "Author": "u4", "likes": 7},
                ]
            )
            df.to_parquet(clusters_path, index=False)

            config = {
                "input": {"timestamp_column": "timestamp"},
                "temporal": {"default_window": "1M"},
                "output": {
                    "clusters": clusters_path,
                    "temporal_stats": temporal_stats_path,
                    "temporal_events": temporal_events_path,
                    "cluster_trend_snapshots_1m": snapshots_1m_path,
                    "cluster_trend_snapshots_1m_csv": snapshots_1m_csv_path,
                },
            }
            config_path = os.path.join(tmp, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "2_temporal_analysis.py"),
                "--config",
                config_path,
                "--force",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(
                res.returncode,
                0,
                msg=f"Temporal analysis failed\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )

            self.assertTrue(os.path.exists(snapshots_1m_path))
            self.assertTrue(os.path.exists(snapshots_1m_csv_path))

            snapshots = pd.read_parquet(snapshots_1m_path)
            self.assertGreater(len(snapshots), 0)
            for col in [
                "time_window",
                "cluster",
                "cluster_label",
                "post_count",
                "market_share",
                "volume_pct_change",
                "volume_volatility",
                "momentum",
                "lifecycle_state",
                "anomaly_score",
            ]:
                self.assertIn(col, snapshots.columns)


if __name__ == "__main__":
    unittest.main()
