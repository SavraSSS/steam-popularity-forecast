from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "model_dataset.csv"
TARGET = "target_next_month"


def load_splits():
    data = pd.read_csv(DATASET, parse_dates=["month"])
    forecast_month = data["month"] + pd.offsets.MonthBegin(1)

    validation_start = pd.Timestamp("2020-01-01")
    test_start = pd.Timestamp("2021-01-01")

    train = data.loc[forecast_month < validation_start]
    validation = data.loc[forecast_month.between(
        validation_start, test_start, inclusive="left"
    )]
    test = data.loc[forecast_month >= test_start]

    if train.empty or validation.empty or test.empty:
        raise ValueError("Одна из частей временной выборки пуста")

    return train, validation, test


def calculate_metrics(actual, predicted):
    error = actual - predicted
    return error.abs().mean(), error.pow(2).mean() ** 0.5


def main():
    train, validation, test = load_splits()

    print(f"Training rows: {len(train)}")
    for name, part in (("Validation", validation), ("Test", test)):
        mae, rmse = calculate_metrics(part[TARGET], part["Avg_players"])
        print(f"{name}: rows={len(part)}, MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()