import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .evaluate_baseline import TARGET, load_splits


VALIDATION_YEARS = (2018, 2019, 2020)
MODELS = ("baseline", "neighbors", "random_forest")
ACTIVITY_FEATURES = [
    "Avg_players",
    "avg_players_lag_1",
    "avg_players_lag_3",
    "avg_players_lag_6",
    "avg_players_roll_3",
    "avg_players_roll_6",
]
TEXT_FEATURES = ["mean_review_words", "mean_review_exclamations"]


def steam_features(data):
    activity = np.log1p(data[ACTIVITY_FEATURES])
    text = np.log1p(data[TEXT_FEATURES].fillna(0))
    features = activity.join(text)
    online = features["Avg_players"]
    features["peak_to_average"] = np.log1p(data["Peak_Players"]) - online
    features["monthly_momentum"] = online - features["avg_players_lag_1"]
    features["deviation_from_recent_level"] = (
        online - features["avg_players_roll_3"]
    )
    return features


def fit_model(name, train):
    if name == "baseline":
        return None
    if name == "neighbors":
        model = make_pipeline(
            RobustScaler(),
            KNeighborsRegressor(n_neighbors=25, weights="distance"),
        )
    elif name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=40, random_state=42
        )
    else:
        raise ValueError(f"Неизвестная модель: {name}")

    growth = np.log1p(train[TARGET]) - np.log1p(train["Avg_players"])
    return model.fit(steam_features(train), growth)


def predict_players(model, data):
    current = data["Avg_players"]
    if model is None:
        return current

    growth = model.predict(steam_features(data))
    return np.expm1(np.log1p(current) + growth).clip(lower=0)


def metrics(rows):
    error = rows["prediction"] - rows[TARGET]
    return {
        "rows": len(rows),
        "games": rows["appid"].nunique(),
        "game_mae": error.abs().groupby(rows["appid"]).mean().mean(),
        "mae": error.abs().mean(),
        "rmse": error.pow(2).mean() ** 0.5,
    }


def validate_models(data):
    target_month = data["month"] + pd.offsets.MonthBegin(1)
    results = []

    for year in VALIDATION_YEARS:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = start + pd.DateOffset(years=1)
        train = data.loc[target_month < start]
        validation = data.loc[(target_month >= start) & (target_month < end)]
        if train.empty or validation.empty:
            raise ValueError(f"Недостаточно данных для окна {year}")

        for name in MODELS:
            model = fit_model(name, train)
            rows = validation[["appid", TARGET]].copy()
            rows["prediction"] = predict_players(model, validation)
            results.append(rows.assign(model=name, year=year))

    return pd.concat(results, ignore_index=True)

def evaluate_test(development, test):
    current = test["Avg_players"]
    actual_direction = np.sign(test[TARGET] - current)
    results = []

    for name in MODELS:
        model = fit_model(name, development)
        rows = test[["appid", TARGET]].copy()
        rows["prediction"] = predict_players(model, test)

        predicted_direction = np.sign(rows["prediction"] - current)
        direction_accuracy = actual_direction.eq(predicted_direction).mean()

        results.append({
            "model": name,
            **metrics(rows),
            "direction_pct": 100 * direction_accuracy,
        })

    return pd.DataFrame(results)

def main():
    train, validation, test = load_splits()
    development = pd.concat([train, validation])
    predictions = validate_models(development)

    by_year = [
        {"year": year, "model": name, **metrics(rows)}
        for (year, name), rows in predictions.groupby(["year", "model"])
    ]
    summary = [
        {"model": name, **metrics(rows)}
        for name, rows in predictions.groupby("model")
    ]
    ranking = pd.DataFrame(summary).sort_values(["game_mae", "rmse", "model"])

    print(pd.DataFrame(by_year).to_string(index=False, float_format="{:.2f}".format))
    print("\nВсе окна")
    print(ranking.to_string(index=False, float_format="{:.2f}".format))
    print(f"Лучшая модель по средней MAE игр: {ranking.iloc[0]['model']}")
    print("\nИсторический тест")
    test_results = evaluate_test(development, test)
    print(test_results.to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    main()