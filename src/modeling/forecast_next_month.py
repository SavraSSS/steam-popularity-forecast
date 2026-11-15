import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .evaluate_baseline import ROOT
from .monthly_models import predict_players, steam_features
from .register_model import REGISTRY, write_atomic


ARTIFACTS = ROOT / "artifacts"
HISTORY = ROOT / "data" / "processed" / "prepared_dataset.csv"


def load_active_model():
    active = json.loads((REGISTRY / "active.json").read_text(encoding="utf-8"))
    model_id = active["model_id"]
    folder = REGISTRY / model_id
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return model_id, metadata, joblib.load(folder / "model.joblib")


def make_release(history, model_id, metadata, model):
    observed = history.loc[
        history["month"].notna() & np.isfinite(history["Avg_players"])
    ]
    if observed.empty:
        raise ValueError("В витрине нет наблюдений по онлайну")

    month = observed["month"].max()
    if month < pd.Timestamp(metadata["trained_through"]):
        raise ValueError("Версия модели обучена на данных позднее месяца наблюдения")

    snapshot = observed.loc[observed["month"].eq(month)]
    if snapshot["appid"].duplicated().any():
        raise ValueError("За месяц наблюдения есть повторяющиеся игры")

    release = snapshot[["appid", "Game_Name", "month", "Avg_players"]].rename(
        columns={"Avg_players": "current_players"}
    )
    release["forecast_month"] = month + pd.offsets.MonthBegin(1)
    release["forecast_players"] = release["current_players"]
    release["model"] = "baseline"
    release["model_version"] = model_id
    release["issued_at"] = datetime.now(timezone.utc).isoformat()

    if model is not None:
        complete = np.isfinite(steam_features(snapshot)).all(axis=1)
        eligible = snapshot.loc[complete]
        if not eligible.empty:
            release.loc[eligible.index, "forecast_players"] = predict_players(
                model, eligible
            )
            release.loc[eligible.index, "model"] = metadata["model"]

    release["change_players"] = release["forecast_players"] - release["current_players"]
    current = release["current_players"].where(release["current_players"].gt(0))
    release["change_pct"] = 100 * release["change_players"] / current
    return release


def publish_release(release):
    target_month = release["forecast_month"].iloc[0]
    path = ARTIFACTS / "forecast_history" / f"{target_month:%Y-%m}.csv"

    added = len(release)
    if path.exists():
        previous = pd.read_csv(path, parse_dates=["month", "forecast_month"])
        new_games = release.loc[~release["appid"].isin(previous["appid"])]
        added = len(new_games)
        release = (
            pd.concat([previous, new_games], ignore_index=True) if added else previous
        )

    release = release.sort_values(
        ["forecast_players", "appid"], ascending=[False, True]
    )
    content = release.to_csv(index=False).encode("utf-8")
    if added:
        write_atomic(path, content)
    write_atomic(ARTIFACTS / "latest_forecasts.csv", content)
    return release, added


def main():
    history = pd.read_csv(HISTORY, parse_dates=["month"])
    release, added = publish_release(make_release(history, *load_active_model()))

    print(f"Last observed month: {release['month'].iloc[0]:%Y-%m}")
    print(f"Forecast month: {release['forecast_month'].iloc[0]:%Y-%m}")
    print(f"Games forecast: {len(release)}")
    print(f"New archived forecasts: {added}")
    print(
        release[["Game_Name", "model", "forecast_players", "change_pct"]]
        .to_string(index=False, float_format="{:.2f}".format)
    )


if __name__ == "__main__":
    main()