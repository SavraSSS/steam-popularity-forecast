import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate_baseline import TARGET, calculate_metrics, load_splits


FEATURES = [
    "Avg_players",
    "avg_players_lag_1",
    "avg_players_lag_3",
    "avg_players_lag_6",
    "avg_players_roll_3",
    "avg_players_roll_6",
]

def build_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=20))

def main():
    train, validation, _ = load_splits()

    model = build_model()
    train_change = np.log1p(train[TARGET]) - np.log1p(train["Avg_players"])
    model.fit(np.log1p(train[FEATURES]), train_change)

    predicted_change = model.predict(np.log1p(validation[FEATURES]))
    predicted_players = np.expm1(
        np.log1p(validation["Avg_players"]) + predicted_change
    ).clip(lower=0)

    for name, predicted in (
        ("Baseline", validation["Avg_players"]),
        ("Ridge", predicted_players),
    ):
        mae, rmse = calculate_metrics(validation[TARGET], predicted)
        print(f"{name}: rows={len(validation)}, MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()