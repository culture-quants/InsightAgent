"""
Script 6b: Forecast dashboard visualization.
"""

import argparse
import json
import os

import yaml


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_dashboard(pred):
    activated = pred.get("activated_themes", [])[:10]
    rows = []
    for item in activated:
        rows.append(
            "<tr>"
            f"<td>{item.get('theme_id')}</td>"
            f"<td>{item.get('theme_name','')}</td>"
            f"<td>{item.get('score',0.0):.3f}</td>"
            f"<td>{item.get('weight',0.0):.3f}</td>"
            f"<td>{item.get('direction_label','')}</td>"
            "</tr>"
        )
    table_rows = "\n".join(rows) if rows else "<tr><td colspan='5'>No activated themes</td></tr>"
    llm_meta = pred.get("llm_semantic", {})

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Forecast Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 10px; }}
    .k {{ color: #6b7280; font-size: 13px; }} .v {{ font-size: 18px; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; }} th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Forecast Dashboard</h1>
  <div class="card">
    <h3>Event</h3>
    <p>{pred.get("event_text","")}</p>
  </div>
  <div class="card">
    <h3>Prediction</h3>
    <div class="grid">
      <div><div class="k">Direction</div><div class="v">{pred.get("predicted_direction","flat")}</div></div>
      <div><div class="k">Confidence</div><div class="v">{float(pred.get("confidence", 0.0)):.3f}</div></div>
      <div><div class="k">Expected Volatility</div><div class="v">{float(pred.get("predicted_volatility", 0.0)):.3f}</div></div>
      <div><div class="k">Risk Band</div><div class="v">{pred.get("risk_band","medium")}</div></div>
      <div><div class="k">Method</div><div class="v">{pred.get("method","heuristic")}</div></div>
      <div><div class="k">Activated Themes</div><div class="v">{int(pred.get("activated_theme_count", 0))}</div></div>
    </div>
  </div>
  <div class="card">
    <h3>Top Contributing Themes</h3>
    <table>
      <thead><tr><th>ID</th><th>Theme</th><th>Score</th><th>Weight</th><th>Cluster Direction</th></tr></thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h3>Semantic Matching Layer</h3>
    <p>LLM semantic enabled: <code>{llm_meta.get("enabled", False)}</code></p>
    <p>Provider/model: <code>{llm_meta.get("provider","n/a")} / {llm_meta.get("model","n/a")}</code></p>
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate forecast dashboard HTML.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    config_dir = os.path.dirname(config_path)

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(config_dir, path)

    pred_path = resolve(config.get("event_forecast", {}).get("prediction_output", "output/event_direction_prediction.json"))
    dashboard_path = resolve(
        config.get("event_forecast", {}).get("dashboard_output", "output/visualizations/forecast_dashboard.html")
    )
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)

    html = render_dashboard(pred)
    os.makedirs(os.path.dirname(dashboard_path) or ".", exist_ok=True)
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved forecast dashboard to {dashboard_path}")


if __name__ == "__main__":
    main()
