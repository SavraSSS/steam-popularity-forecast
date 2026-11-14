import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .evaluate_baseline import TARGET, calculate_metrics, load_splits
from .train_ridge import FEATURES


def main():
    train, validation, _ = load_splits()

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=20,
        random_state=42,
    )
    train_change = np.log1p(train[TARGET]) - np.log1p(train["Avg_players"])
    model.fit(np.log1p(train[FEATURES]), train_change)

    predicted_change = model.predict(np.log1p(validation[FEATURES]))
    predicted_players = np.expm1(
        np.log1p(validation["Avg_players"]) + predicted_change
    ).clip(lower=0)

    for name, predicted in (
        ("Baseline", validation["Avg_players"]),
        ("Random forest", predicted_players),
    ):
        mae, rmse = calculate_metrics(validation[TARGET], predicted)
        print(f"{name}: rows={len(validation)}, MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()