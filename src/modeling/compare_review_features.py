import numpy as np
import pandas as pd
from .evaluate_baseline import TARGET, calculate_metrics, load_splits
from .train_forest import build_model
from .train_ridge import FEATURES


TEXT_FEATURES = ["mean_review_words", "mean_review_exclamations"]


def make_features(data, include_text, include_month):
    features = np.log1p(data[FEATURES])

    if include_text:
        text = np.log1p(data[TEXT_FEATURES].fillna(0))
        features = features.join(text)

    if include_month:
        month = pd.Categorical(
            data["month"].dt.month,
            categories=range(1, 13),
        )
        month_features = pd.get_dummies(month, prefix="month", dtype=float)
        month_features.index = data.index
        features = features.join(month_features)

    return features


def main():
    train, validation, _ = load_splits()

    for name, part in (("Train", train), ("Validation", validation)):
        covered = part[TEXT_FEATURES].notna().all(axis=1).sum()
        print(f"{name}: text coverage={covered}/{len(part)}")

    current_train = np.log1p(train["Avg_players"])
    train_change = np.log1p(train[TARGET]) - current_train
    current_validation = np.log1p(validation["Avg_players"])

    for name, include_text, include_month in (
        ("Forest: activity only", False, False),
        ("Forest: activity + text", True, False),
        ("Forest: activity + month", False, True),
        ("Forest: activity + text + month", True, True),
    ):
        model = build_model()
        model.fit(make_features(train, include_text, include_month), train_change)
        change = model.predict(make_features(validation, include_text, include_month))
        predicted = np.expm1(current_validation + change).clip(lower=0)

        mae, rmse = calculate_metrics(validation[TARGET], predicted)
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()