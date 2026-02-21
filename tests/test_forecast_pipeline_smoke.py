import json
import os
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ForecastPipelineSmokeTests(unittest.TestCase):
    def test_train_forecast_visualize_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "output")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

            temporal_path = os.path.join(output_dir, "temporal_stats.parquet")
            theme_windows_path = os.path.join(output_dir, "theme_activations_windows.parquet")
            model_dir = os.path.join(output_dir, "models")
            model_path = os.path.join(model_dir, "direction_model.pkl")
            metrics_path = os.path.join(model_dir, "direction_model_metrics.json")
            prediction_path = os.path.join(output_dir, "event_direction_prediction.json")
            dashboard_path = os.path.join(output_dir, "visualizations", "forecast_dashboard.html")
            cluster_db_path = os.path.join(output_dir, "cluster_state_database.parquet")

            temporal_df = pd.DataFrame(
                [
                    {"cluster": 0, "time_window": "2025-01-31", "post_count": 10, "volume_pct_change": 0.00, "volume_volatility": 0.40, "momentum": 0.0, "market_share": 0.50},
                    {"cluster": 0, "time_window": "2025-02-28", "post_count": 11, "volume_pct_change": 0.10, "volume_volatility": 0.45, "momentum": 1.0, "market_share": 0.52},
                    {"cluster": 0, "time_window": "2025-03-31", "post_count": 13, "volume_pct_change": 0.18, "volume_volatility": 0.50, "momentum": 1.5, "market_share": 0.55},
                    {"cluster": 0, "time_window": "2025-04-30", "post_count": 15, "volume_pct_change": 0.15, "volume_volatility": 0.60, "momentum": 2.0, "market_share": 0.57},
                    {"cluster": 1, "time_window": "2025-01-31", "post_count": 20, "volume_pct_change": 0.00, "volume_volatility": 0.60, "momentum": 0.0, "market_share": 0.50},
                    {"cluster": 1, "time_window": "2025-02-28", "post_count": 18, "volume_pct_change": -0.10, "volume_volatility": 0.70, "momentum": -1.0, "market_share": 0.48},
                    {"cluster": 1, "time_window": "2025-03-31", "post_count": 16, "volume_pct_change": -0.11, "volume_volatility": 0.75, "momentum": -1.2, "market_share": 0.45},
                    {"cluster": 1, "time_window": "2025-04-30", "post_count": 14, "volume_pct_change": -0.12, "volume_volatility": 0.80, "momentum": -1.4, "market_share": 0.43},
                ]
            )
            temporal_df.to_parquet(temporal_path, index=False)

            theme_windows_df = pd.DataFrame(
                [
                    {"cluster": 0, "time_window": "2025-01-31", "theme_0_score": 0.80, "theme_1_score": 0.10},
                    {"cluster": 0, "time_window": "2025-02-28", "theme_0_score": 0.85, "theme_1_score": 0.09},
                    {"cluster": 0, "time_window": "2025-03-31", "theme_0_score": 0.90, "theme_1_score": 0.08},
                    {"cluster": 0, "time_window": "2025-04-30", "theme_0_score": 0.92, "theme_1_score": 0.07},
                    {"cluster": 1, "time_window": "2025-01-31", "theme_0_score": 0.10, "theme_1_score": 0.78},
                    {"cluster": 1, "time_window": "2025-02-28", "theme_0_score": 0.09, "theme_1_score": 0.82},
                    {"cluster": 1, "time_window": "2025-03-31", "theme_0_score": 0.08, "theme_1_score": 0.86},
                    {"cluster": 1, "time_window": "2025-04-30", "theme_0_score": 0.07, "theme_1_score": 0.88},
                ]
            )
            theme_windows_df.to_parquet(theme_windows_path, index=False)

            lexicon_path = os.path.join(tmp, "theme_lexicon.yaml")
            with open(lexicon_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "themes": [
                            {"id": 0, "name": "Theme Up", "keywords": ["robotics", "launch"], "synonyms": [], "phrases": []},
                            {"id": 1, "name": "Theme Down", "keywords": ["layoff", "cuts"], "synonyms": [], "phrases": []},
                        ]
                    },
                    f,
                    sort_keys=False,
                )

            clusters_csv_path = os.path.join(tmp, "clusters.csv")
            pd.DataFrame(
                [{"cluster": 0, "timestamp": "2025-01-31", "text": "robotics launch"}, {"cluster": 1, "timestamp": "2025-01-31", "text": "layoff cuts"}]
            ).to_csv(clusters_csv_path, index=False)

            config_path = os.path.join(tmp, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "input": {
                            "data": clusters_csv_path,
                            "text_column": "text",
                            "timestamp_column": "timestamp",
                        },
                        "temporal": {"default_window": "1M"},
                        "output": {
                            "dir": output_dir,
                            "temporal_stats": temporal_path,
                            "viz_dir": os.path.join(output_dir, "visualizations"),
                        },
                        "theme_activation": {
                            "theme_lexicon": lexicon_path,
                            "output": os.path.join(output_dir, "theme_activations.parquet"),
                            "global_threshold": 0.03,
                        },
                        "event_forecast": {
                            "cluster_db_output": cluster_db_path,
                            "prediction_output": prediction_path,
                            "activation_threshold": 0.1,
                            "target_epsilon": 0.05,
                            "risk_band_quantiles": {"low": 0.33, "high": 0.66},
                        },
                        "model_training": {
                            "test_fraction": 0.25,
                            "target_epsilon": 0.05,
                            "models": ["logistic", "random_forest"],
                            "output_model_path": model_path,
                            "output_metrics_path": metrics_path,
                        },
                        "llm": {"enabled": False, "provider": "google", "model": "gemini-2.5-flash"},
                    },
                    f,
                    sort_keys=False,
                )

            train_cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "6_train_direction_model.py"),
                "--config",
                config_path,
                "--force",
            ]
            train_res = subprocess.run(train_cmd, capture_output=True, text=True, check=False)
            self.assertEqual(
                train_res.returncode,
                0,
                msg=f"Train failed\nSTDOUT:\n{train_res.stdout}\nSTDERR:\n{train_res.stderr}",
            )
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            self.assertIn("walk_forward_cv", metrics)
            self.assertIn("threshold_tuning", metrics)
            self.assertIn("epsilon_tuning", metrics)
            self.assertIn("calibration", metrics)

            infer_cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "5_event_direction_forecast.py"),
                "--config",
                config_path,
                "--event-text",
                "Google launches new robotics model",
                "--model-path",
                model_path,
                "--force",
            ]
            infer_res = subprocess.run(infer_cmd, capture_output=True, text=True, check=False)
            self.assertEqual(
                infer_res.returncode,
                0,
                msg=f"Inference failed\nSTDOUT:\n{infer_res.stdout}\nSTDERR:\n{infer_res.stderr}",
            )

            with open(prediction_path, encoding="utf-8") as f:
                pred = json.load(f)
            for key in ("predicted_direction", "predicted_volatility", "confidence", "risk_band", "method"):
                self.assertIn(key, pred)
            self.assertIn("class_probabilities", pred)

            viz_cmd = [
                sys.executable,
                os.path.join(ROOT, "scripts", "6b_visualize_forecast.py"),
                "--config",
                config_path,
            ]
            viz_res = subprocess.run(viz_cmd, capture_output=True, text=True, check=False)
            self.assertEqual(
                viz_res.returncode,
                0,
                msg=f"Viz failed\nSTDOUT:\n{viz_res.stdout}\nSTDERR:\n{viz_res.stderr}",
            )
            self.assertTrue(os.path.exists(dashboard_path))


if __name__ == "__main__":
    unittest.main()
