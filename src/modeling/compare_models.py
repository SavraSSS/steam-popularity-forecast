import numpy as np
import pandas as pd
from .evaluate_baseline import ROOT, TARGET, calculate_metrics, load_splits
from .train_forest import build_model as build_forest
from .train_ridge import FEATURES, build_model as build_ridge


def evaluate(name, data, predicted):
    mae, rmse = calculate_metrics(data[TARGET], predicted)
    print(f"{name}: rows={len(data)}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return {"model": name, "rows": len(data), "mae": mae, "rmse": rmse}


def main():
    train, validation, test = load_splits()
    development = pd.concat([train, validation])

    train_features = np.log1p(development[FEATURES])
    train_change = (
        np.log1p(development[TARGET])
        - np.log1p(development["Avg_players"])
    )
    test_features = np.log1p(test[FEATURES])
    current_players = np.log1p(test["Avg_players"])

    predictions = test[["appid", "Game_Name", "month", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "actual_players"})
    predictions["forecast_month"] = predictions["month"] + pd.offsets.MonthBegin(1)
    predictions["baseline"] = test["Avg_players"]

    print(f"Training + validation rows: {len(development)}")
    metrics = [evaluate("Baseline", test, predictions["baseline"])]

    for name, column, build_model in (
        ("Ridge", "ridge", build_ridge),
        ("Random forest", "random_forest", build_forest),
    ):
        model = build_model()
        model.fit(train_features, train_change)
        predicted_change = model.predict(test_features)
        predictions[column] = np.expm1(current_players + predicted_change).clip(lower=0)
        metrics.append(evaluate(name, test, predictions[column]))

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()