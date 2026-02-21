import unittest

import pandas as pd

from scripts.event_forecast import (
    build_cluster_profiles,
    estimate_event_direction,
    score_event_against_lexicon,
)


class EventForecastTests(unittest.TestCase):
    def test_score_event_against_lexicon_activates_relevant_theme(self):
        themes = [
            {
                "id": 1,
                "name": "Weather Forecasting AI",
                "keywords": ["weather ai", "forecast"],
                "synonyms": ["climate model"],
                "phrases": ["weather forecasting"],
            },
            {
                "id": 2,
                "name": "Gaming AI",
                "keywords": ["chess", "alphazero"],
                "synonyms": [],
                "phrases": [],
            },
        ]
        event_text = "New weather forecasting model improves climate prediction."
        scores = score_event_against_lexicon(event_text, themes)
        self.assertGreater(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)

    def test_build_cluster_profiles_outputs_direction_signals(self):
        temporal_df = pd.DataFrame(
            [
                {"cluster": 1, "time_window": "2025-01-01", "post_count": 10, "volume_pct_change": 0.0, "momentum": 0.0, "volume_volatility": 1.0},
                {"cluster": 1, "time_window": "2025-02-01", "post_count": 12, "volume_pct_change": 0.2, "momentum": 2.0, "volume_volatility": 1.3},
                {"cluster": 1, "time_window": "2025-03-01", "post_count": 15, "volume_pct_change": 0.25, "momentum": 3.0, "volume_volatility": 1.5},
                {"cluster": 2, "time_window": "2025-01-01", "post_count": 20, "volume_pct_change": 0.0, "momentum": 0.0, "volume_volatility": 2.0},
                {"cluster": 2, "time_window": "2025-02-01", "post_count": 17, "volume_pct_change": -0.15, "momentum": -2.0, "volume_volatility": 2.3},
                {"cluster": 2, "time_window": "2025-03-01", "post_count": 15, "volume_pct_change": -0.12, "momentum": -2.5, "volume_volatility": 2.5},
            ]
        )
        profiles = build_cluster_profiles(temporal_df)
        c1 = profiles.loc[profiles["cluster"] == 1].iloc[0]
        c2 = profiles.loc[profiles["cluster"] == 2].iloc[0]
        self.assertEqual(c1["direction_label"], "up")
        self.assertEqual(c2["direction_label"], "down")
        self.assertGreater(c1["direction_score"], 0.0)
        self.assertLess(c2["direction_score"], 0.0)

    def test_estimate_event_direction_returns_weighted_summary(self):
        themes = [
            {"id": 1, "name": "Weather", "keywords": ["weather"], "synonyms": [], "phrases": []},
            {"id": 2, "name": "Gaming", "keywords": ["chess"], "synonyms": [], "phrases": []},
        ]
        profiles = pd.DataFrame(
            [
                {"cluster": 1, "direction_score": 0.6, "direction_label": "up", "avg_volatility": 1.4},
                {"cluster": 2, "direction_score": -0.5, "direction_label": "down", "avg_volatility": 2.1},
            ]
        )
        result = estimate_event_direction("weather model warning", themes, profiles, activation_threshold=0.05)
        self.assertIn(result["predicted_direction"], {"up", "down", "flat"})
        self.assertGreater(result["activated_theme_count"], 0)
        self.assertGreater(result["predicted_volatility"], 0.0)

    def test_estimate_event_direction_supports_per_theme_thresholds(self):
        themes = [
            {"id": 1, "name": "Weather", "keywords": ["weather"], "synonyms": [], "phrases": []},
            {"id": 2, "name": "AI", "keywords": ["ai"], "synonyms": [], "phrases": []},
        ]
        profiles = pd.DataFrame(
            [
                {"cluster": 1, "direction_score": 0.3, "direction_label": "up", "avg_volatility": 1.0},
                {"cluster": 2, "direction_score": 0.4, "direction_label": "up", "avg_volatility": 1.2},
            ]
        )
        result = estimate_event_direction(
            "weather ai model",
            themes,
            profiles,
            activation_threshold=1.0,
            per_theme_thresholds={"2": 5.0},
        )
        activated_ids = {x["theme_id"] for x in result["activated_themes"]}
        self.assertIn(1, activated_ids)
        self.assertNotIn(2, activated_ids)


if __name__ == "__main__":
    unittest.main()
