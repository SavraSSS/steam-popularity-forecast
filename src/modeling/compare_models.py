import json

import pandas as pd

from .evaluate_baseline import ROOT, TARGET, calculate_metrics, load_splits
from .search_models import JOURNAL, PARAMETERS, SEED, predict_config, snapshot_dataset


def load_best_configs(version):
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in JOURNAL.glob("*.json")
    ]
    best = {}
    for model, parameters in PARAMETERS.items():
        candidates = [
            row for row in records
            if row["dataset_sha256"] == version
            and row["model"] == model
            and row["seed"] == SEED
            and row["params"] in parameters
        ]
        if not candidates:
            raise ValueError(
                f"Для {model} нет экспериментов на текущей выборке. "
                "Сначала запусти python -m src.modeling.search_models"
            )
        best[model] = min(candidates, key=lambda row: (row["mae"], row["id"]))
    return best


def evaluate(name, actual, predicted):
    mae, rmse = calculate_metrics(actual, predicted)
    print(f"{name}: rows={len(actual)}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return {"model": name, "rows": len(actual), "mae": mae, "rmse": rmse}


def main():
    version, snapshot = snapshot_dataset()
    best = load_best_configs(version)
    train, validation, test = load_splits(snapshot)
    development = pd.concat([train, validation])

    predictions = test[["appid", "Game_Name", "month", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "actual_players"})
    predictions["forecast_month"] = predictions["month"] + pd.offsets.MonthBegin(1)
    predictions["baseline"] = test["Avg_players"]

    print(f"Training + validation rows: {len(development)}")
    metrics = [evaluate("Baseline", test[TARGET], predictions["baseline"])]

    for model, label in (("ridge", "Ridge"), ("random_forest", "Random forest")):
        config = best[model]
        print(f"{label}: experiment={config['id']}, validation MAE={config['mae']:.2f}")
        predictions[model] = predict_config(
            development, test, model, config["params"], config["text"], config["seed"]
        )
        metrics.append(evaluate(label, test[TARGET], predictions[model]))

    output_dir = ROOT / "artifacts"
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()