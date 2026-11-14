import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compare_review_features import make_features
from .evaluate_baseline import TARGET, calculate_metrics, load_splits


SEED = 42
PARAMETERS = {
    "ridge": [{"alpha": 5}, {"alpha": 20}],
    "random_forest": [
        {"n_estimators": 100, "min_samples_leaf": 20},
        {"n_estimators": 100, "min_samples_leaf": 40},
    ],
}


def build_model(model_name, params, seed=SEED):
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    raise ValueError(f"Неизвестная модель: {model_name}")


def evaluate_config(train, validation, model_name, params, include_text, seed=SEED):
    train_features = make_features(train, include_text, include_month=False)
    validation_features = make_features(validation, include_text, include_month=False)

    model = build_model(model_name, params, seed)
    train_change = np.log1p(train[TARGET]) - np.log1p(train["Avg_players"])
    model.fit(train_features, train_change)

    change = model.predict(validation_features)
    predicted = np.expm1(np.log1p(validation["Avg_players"]) + change).clip(lower=0)
    return calculate_metrics(validation[TARGET], predicted)


def main():
    train, validation, _ = load_splits()
    baseline_mae, baseline_rmse = calculate_metrics(
        validation[TARGET], validation["Avg_players"]
    )

    results = []
    for model_name, options in PARAMETERS.items():
        for params in options:
            for include_text in (False, True):
                mae, rmse = evaluate_config(
                    train, validation, model_name, params, include_text
                )
                results.append({
                    "model": model_name,
                    "params": params,
                    "text": include_text,
                    "mae": mae,
                    "rmse": rmse,
                })

    table = pd.DataFrame(results).sort_values("mae")
    print(f"Validation rows: {len(validation)}")
    print(f"Baseline: MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    best = table.iloc[0]
    winner = best["model"] if best["mae"] < baseline_mae else "baseline"
    print(f"Selected by validation MAE: {winner}")


if __name__ == "__main__":
    main()