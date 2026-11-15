import argparse
import json

import numpy as np
import pandas as pd

from .forecast_next_month import ARTIFACTS, HISTORY, load_active_model, make_release
from .register_model import write_atomic
from .monthly_models import steam_features


def score_releases(releases, history):
    facts = history.loc[
        np.isfinite(history["Avg_players"]), ["appid", "month", "Avg_players"]
    ].rename(columns={"month": "forecast_month", "Avg_players": "actual_players"})

    scored = releases.merge(
        facts, on=["appid", "forecast_month"], how="left", validate="one_to_one"
    )
    known = scored["actual_players"].notna()
    scored["status"] = np.where(known, "scored", "pending")
    scored["absolute_error"] = (
        scored["forecast_players"] - scored["actual_players"]
    ).abs()

    current = scored["current_players"]
    predicted_direction = np.sign(scored["forecast_players"] - current)
    actual_direction = np.sign(scored["actual_players"] - current)
    scored["direction_correct"] = predicted_direction.eq(actual_direction).astype(
        "boolean"
    )
    scored.loc[~known, "direction_correct"] = pd.NA
    return scored

def assess_quality(scored, model_id, model_name):
    eligible = scored.loc[
        scored["status"].eq("scored")
        & scored["model_version"].eq(model_id)
        & scored["model"].eq(model_name)
    ].copy()
    eligible["baseline_error"] = (
        eligible["actual_players"] - eligible["current_players"]
    ).abs()

    monthly = eligible.groupby("forecast_month").agg(
        games=("appid", "nunique"),
        model_mae=("absolute_error", "mean"),
        baseline_mae=("baseline_error", "mean"),
    ).tail(3)

    if len(monthly) < 3 or monthly["games"].lt(3).any():
        return monthly, "insufficient_data"

    expected = pd.date_range(monthly.index[0], periods=3, freq="MS")
    if not monthly.index.equals(expected):
        return monthly, "insufficient_data"

    worse = monthly["model_mae"].gt(1.10 * monthly["baseline_mae"])
    return monthly, "retrain_recommended" if worse.all() else "no_alert"

def assess_input_drift(history, month, appids):
    columns = [
        "Avg_players",
        "peak_to_average",
        "monthly_momentum",
        "deviation_from_recent_level",
    ]
    window = history.loc[
        history["appid"].isin(appids)
        & history["month"].between(month - pd.DateOffset(months=12), month)
    ]

    features = steam_features(window)[columns].replace([np.inf, -np.inf], np.nan)
    features["appid"] = window["appid"]
    features["month"] = window["month"]

    past = features.loc[features["month"].lt(month)].groupby("appid")
    current = features.loc[features["month"].eq(month)].set_index("appid")
    results = []

    for column in columns:
        reference = past[column]
        # Нижняя граница защищает от деления на ноль при стабильной истории.
        spread = (reference.quantile(0.75) - reference.quantile(0.25)).clip(lower=0.05)
        scores = (current[column] - reference.median()) / spread
        scores = scores.where(reference.count().reindex(scores.index).ge(6)).dropna()
        shift = float(scores.median()) if len(scores) >= 3 else None
        results.append({
            "feature": column,
            "games": len(scores),
            "median_shift": shift,
            "flagged": shift is not None and abs(shift) > 1.5,
        })

    if any(row["flagged"] for row in results):
        decision = "input_drift"
    elif any(row["median_shift"] is None for row in results):
        decision = "insufficient_data"
    else:
        decision = "no_drift"

    return results, decision

def main():
    parser = argparse.ArgumentParser(description="Сверка прогнозов с фактическим онлайном")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    history = pd.read_csv(HISTORY, parse_dates=["month"])
    active = load_active_model()
    if args.demo:
        observation_months = pd.date_range("2026-07-01", periods=3, freq="MS")
        releases = pd.concat(
            [
                make_release(history.loc[history["month"].le(month)], *active)
                for month in observation_months
            ],
            ignore_index=True,
        )
        print("Историческая проверка — факты за август–октябрь 2026")
    else:
        paths = sorted((ARTIFACTS / "forecast_history").glob("*.csv"))
        if not paths:
            parser.exit(1, "Сначала выпусти прогноз через src.modeling.forecast_next_month\n")

        releases = pd.concat(
            [pd.read_csv(path, parse_dates=["forecast_month"]) for path in paths],
            ignore_index=True,
        )

    scored = score_releases(releases, history)
    filename = "forecast_monitoring_demo.csv" if args.demo else "forecast_monitoring.csv"
    write_atomic(
        ARTIFACTS / filename,
        scored.to_csv(index=False).encode("utf-8"),
    )

    columns = [
        "Game_Name", "status", "forecast_players", "actual_players",
        "absolute_error", "direction_correct",
    ]
    print(scored[columns].to_string(index=False, na_rep="—", float_format="{:.2f}".format))

    complete = scored.loc[scored["status"].eq("scored")]
    print(f"Сверено: {len(complete)}, ожидают факта: {len(scored) - len(complete)}")
    if not complete.empty:
        print(f"MAE: {complete['absolute_error'].mean():.2f}")
        print(f"Угадано направление: {100 * complete['direction_correct'].mean():.2f}%")
    monthly, decision = assess_quality(scored, active[0], active[1]["model"])
    if not monthly.empty:
        print("\nОшибки по месяцам")
        print(monthly.to_string(float_format="{:.2f}".format))

    print(f"Сигнал качества: {decision}")
    latest_target = scored["forecast_month"].max()
    observation_month = latest_target - pd.offsets.MonthBegin(1)
    latest_games = scored.loc[scored["forecast_month"].eq(latest_target), "appid"]
    drift_rows, drift_decision = assess_input_drift(
        history, observation_month, latest_games
    )

    print(f"\nСдвиг признаков за {observation_month:%Y-%m}")
    print(pd.DataFrame(drift_rows).to_string(index=False, float_format="{:.2f}".format))
    print(f"Сигнал по признакам: {drift_decision}")
    if not args.demo:
        status = {
            "model_version": active[0],
            "decision": decision,
            "months": monthly.reset_index().to_dict(orient="records"),
            "input_drift": drift_decision,
            "input_features": drift_rows,
        }
        write_atomic(
            ARTIFACTS / "forecast_quality.json",
            json.dumps(status, default=str, indent=2).encode("utf-8"),
        )

if __name__ == "__main__":
    main()