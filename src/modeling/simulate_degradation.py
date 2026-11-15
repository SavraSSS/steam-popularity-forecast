import pandas as pd

from .forecast_next_month import ARTIFACTS, HISTORY, load_active_model
from .monitor_forecasts import assess_quality, score_releases


def main():
    path = ARTIFACTS / "forecast_monitoring_demo.csv"
    if not path.is_file():
        raise SystemExit("Сначала запусти src.modeling.monitor_forecasts --demo")

    reference = pd.read_csv(path, parse_dates=["forecast_month"])
    history = pd.read_csv(HISTORY, parse_dates=["month"])
    model_id, info, _ = load_active_model()

    if not reference["model_version"].eq(model_id).all():
        raise SystemExit("Запусти мониторинг с --demo для текущей версии модели")

    _, before = assess_quality(reference, model_id, info["model"])
    releases = reference.drop(
        columns=["actual_players", "status", "absolute_error", "direction_correct"]
    ).copy()
    releases["forecast_players"] *= 1.5

    scored = score_releases(releases, history)
    monthly, after = assess_quality(scored, model_id, info["model"])

    print(f"Сигнал на исходных прогнозах: {before}")
    print("\nИскусственное завышение прогнозов на 50%")
    print(monthly.to_string(float_format="{:.2f}".format))
    print(f"Сигнал после ухудшения: {after}")


if __name__ == "__main__":
    main()