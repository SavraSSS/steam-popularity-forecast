import numpy as np

from .evaluate_baseline import TARGET, calculate_metrics, load_splits
from .train_forest import build_model
from .train_ridge import FEATURES


TEXT_FEATURES = ["mean_review_words", "mean_review_exclamations"]


def make_features(data, include_text):
    features = np.log1p(data[FEATURES])
    if include_text:
        text = np.log1p(data[TEXT_FEATURES].fillna(0))
        features = features.join(text)
    return features


def main():
    train, validation, _ = load_splits()

    for name, part in (("Train", train), ("Validation", validation)):
        covered = part[TEXT_FEATURES].notna().all(axis=1).sum()
        print(f"{name}: text coverage={covered}/{len(part)}")

    current_train = np.log1p(train["Avg_players"])
    train_change = np.log1p(train[TARGET]) - current_train
    current_validation = np.log1p(validation["Avg_players"])

    for name, include_text in (
        ("Forest without text", False),
        ("Forest with text", True),
    ):
        model = build_model()
        model.fit(make_features(train, include_text), train_change)
        change = model.predict(make_features(validation, include_text))
        predicted = np.expm1(current_validation + change).clip(lower=0)

        mae, rmse = calculate_metrics(validation[TARGET], predicted)
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()