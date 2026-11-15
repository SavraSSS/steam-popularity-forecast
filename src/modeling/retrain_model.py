import argparse
import json
from importlib.metadata import version

import pandas as pd

from .evaluate_baseline import DATASET, TARGET
from .forecast_next_month import ARTIFACTS, load_active_model
from .monthly_models import fit_model, predict_players
from .register_model import save_model_version


def game_mae(data, predictions):
    errors = (predictions - data[TARGET]).abs()
    return float(errors.groupby(data["appid"]).mean().mean())


def main():
    parser = argparse.ArgumentParser(description="Переобучение и проверка кандидата")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    active_id, active_info, active_model = load_active_model()
    if not args.demo:
        status_path = ARTIFACTS / "forecast_quality.json"
        if not status_path.exists():
            parser.exit(1, "Сначала запусти src.modeling.monitor_forecasts\n")

        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["model_version"] != active_id:
            parser.exit(1, "После смены версии заново запусти мониторинг\n")

        quality_alert = status["decision"] == "retrain_recommended"
        drift_alert = status["input_drift"] == "input_drift"
        if not (quality_alert or drift_alert):
            print("Переобучение пропущено — сигналов нет")
            return

    data = pd.read_csv(DATASET, parse_dates=["month"])
    data["target_month"] = data["month"] + pd.offsets.MonthBegin(1)

    last_month = data["target_month"].max()
    check_months = pd.date_range(end=last_month, periods=3, freq="MS")
    first_month = check_months[0]
    train = data.loc[data["target_month"].lt(first_month)]
    check = data.loc[data["target_month"].isin(check_months)]

    games_per_month = check.groupby("target_month")["appid"].nunique()
    if games_per_month.reindex(check_months, fill_value=0).lt(3).any():
        print("Переобучение отложено — нужны три месяца с фактом по трём играм")
        return

    if pd.Timestamp(active_info["trained_through"]) >= first_month:
        print("Переобучение отложено — пока мало месяцев после обучения активной модели")
        return

    if train["target_month"].max() <= pd.Timestamp(active_info["trained_through"]):
        print("Переобучение пропущено — до проверочного окна нет новых фактов")
        return

    candidate_name = active_info["model"]
    if candidate_name == "baseline":
        candidate_name = "neighbors"
    candidate = fit_model(candidate_name, train)

    scores = {
        "baseline": game_mae(check, check["Avg_players"]),
        "active": game_mae(check, predict_players(active_model, check)),
        "candidate": game_mae(check, predict_players(candidate, check)),
    }

    approved = (
        scores["active"] > 0
        and scores["candidate"] <= 0.95 * scores["active"]
        and scores["candidate"] <= scores["baseline"]
    )
    decision = "approved" if approved else "rejected"

    print(f"Проверка: {first_month:%Y-%m} — {last_month:%Y-%m}, строк: {len(check)}")
    for name, score in scores.items():
        print(f"{name}: MAE по играм {score:.2f}")
    print(f"Допуск кандидата: {decision}")

    if args.demo:
        return

    if hasattr(candidate, "steps"):
        parameters = {
            name: step.get_params(deep=False) for name, step in candidate.steps
        }
    else:
        parameters = candidate.get_params(deep=False)

    metadata = {
        "model": candidate_name,
        "trained_through": train["target_month"].max().date().isoformat(),
        "training_rows": len(train),
        "features": list(candidate.feature_names_in_),
        "parameters": parameters,
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn")
        },
        "compared_against": active_id,
        "check_months": [month.date().isoformat() for month in check_months],
        "check_rows": len(check),
        "check_mae_by_game": scores,
        "decision": decision,
    }
    candidate_id = save_model_version(candidate, metadata)
    print(f"Версия кандидата: {candidate_id}")


if __name__ == "__main__":
    main()