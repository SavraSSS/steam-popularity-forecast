import numpy as np
import pandas as pd

from .evaluate_baseline import TARGET, calculate_metrics, load_splits
from .train_forest import build_model as build_forest
from .train_ridge import FEATURES, build_model as build_ridge


def print_metrics(name, data, predicted):
    mae, rmse = calculate_metrics(data[TARGET], predicted)
    print(f"{name}: rows={len(data)}, MAE={mae:.2f}, RMSE={rmse:.2f}")


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

    print(f"Training + validation rows: {len(development)}")
    print_metrics("Baseline", test, test["Avg_players"])

    for name, build_model in (
        ("Ridge", build_ridge),
        ("Random forest", build_forest),
    ):
        model = build_model()
        model.fit(train_features, train_change)
        predicted_change = model.predict(test_features)
        predicted_players = np.expm1(current_players + predicted_change).clip(lower=0)
        print_metrics(name, test, predicted_players)


if __name__ == "__main__":
    main()