from functools import partial

import pandas as pd

from .evaluate_baseline import DATASET, TARGET, load_splits
from .monthly_models import (
    TEXT_FEATURES,
    fit_model,
    metrics,
    predict_players,
    steam_features,
    validate_models,
)


DYNAMICS = (
    "peak_to_average",
    "monthly_momentum",
    "deviation_from_recent_level",
)


def make_variant(data, drop=(), include_month=False):
    features = steam_features(data).drop(columns=list(drop))
    if include_month:
        month = pd.Categorical(data["month"].dt.month, categories=range(1, 13))
        encoded = pd.get_dummies(month, prefix="month", dtype=float)
        features = features.join(encoded.set_axis(data.index))
    return features

def compare_calendar_on_test(train, test):
    variants = (
        ("Без месяца", steam_features),
        ("С месяцем", partial(make_variant, include_month=True)),
    )
    results = []
    for name, builder in variants:
        model = fit_model("neighbors", train, builder)
        rows = test[["appid", TARGET]].copy()
        rows["prediction"] = predict_players(model, test, builder)
        results.append({"Вариант": name, **metrics(rows)})

    return pd.DataFrame(results)

def main():
    train, validation, test = load_splits()
    development = pd.concat([train, validation])
    variants = (
        ("Текущие признаки", steam_features),
        ("Без текста отзывов", partial(make_variant, drop=TEXT_FEATURES)),
        ("Без динамики Steam", partial(make_variant, drop=DYNAMICS)),
        ("С календарным месяцем", partial(make_variant, include_month=True)),
    )

    results = []
    for name, builder in variants:
        predictions = validate_models(
            development, models=("neighbors",), feature_builder=builder
        )
        results.append({"Вариант": name, **metrics(predictions)})

    print(pd.DataFrame(results).to_string(index=False, float_format="{:.2f}".format))
    historical = compare_calendar_on_test(development, test)
    print("\nИсторический тест · январь–сентябрь 2021")
    print(historical.to_string(index=False, float_format="{:.2f}".format))

    data = pd.read_csv(DATASET, parse_dates=["month"])
    target_month = data["month"] + pd.offsets.MonthBegin(1)
    recent = data.loc[target_month.between("2026-08-01", "2026-10-01")]

    if not recent.empty:
        production_data = pd.concat([development, test])
        recent_comparison = compare_calendar_on_test(production_data, recent)
        print("\nСвежий период · август–октябрь 2026")
        print(recent_comparison.to_string(index=False, float_format="{:.2f}".format))


if __name__ == "__main__":
    main()