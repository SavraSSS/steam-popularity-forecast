from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "model_dataset.csv"


def main():
    data = pd.read_csv(DATASET, parse_dates=["month"])
    data["forecast_month"] = data["month"] + pd.offsets.MonthBegin(1)

    validation_start = pd.Timestamp("2020-01-01")
    test_start = pd.Timestamp("2021-01-01")

    months = data["forecast_month"]
    train = data.loc[months < validation_start]
    validation = data.loc[(months >= validation_start) & (months < test_start)]
    test = data.loc[months >= test_start]

    print(f"Training rows: {len(train)}")
    for name, part in (("Validation", validation), ("Test", test)):
        error = part["Avg_players"] - part["target_next_month"]
        mae = error.abs().mean()
        rmse = error.pow(2).mean() ** 0.5
        print(f"{name}: rows={len(part)}, MAE={mae:.2f}, RMSE={rmse:.2f}")


if __name__ == "__main__":
    main()