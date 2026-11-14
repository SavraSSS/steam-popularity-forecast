import pandas as pd

from .evaluate_baseline import ROOT


def main():
    history = pd.read_csv(
        ROOT / "data" / "processed" / "prepared_dataset.csv",
        usecols=["appid", "Game_Name", "month", "Avg_players"],
        parse_dates=["month"],
    )
    observed = history.dropna(subset=["month", "Avg_players"])
    if observed.empty:
        raise ValueError("В витрине нет наблюдений по онлайну")

    latest_month = observed["month"].max()
    forecast_month = latest_month + pd.offsets.MonthBegin(1)

    forecasts = observed.loc[observed["month"].eq(latest_month)].copy()
    forecasts = forecasts.rename(columns={"Avg_players": "current_players"})
    forecasts["forecast_month"] = forecast_month
    forecasts["forecast_players"] = forecasts["current_players"]
    forecasts["model"] = "baseline"
    forecasts = forecasts.sort_values(
        ["forecast_players", "appid"], ascending=[False, True]
    )

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    forecasts.to_csv(output_dir / "latest_forecasts.csv", index=False)

    print(f"Last observed month: {latest_month:%Y-%m}")
    print(f"Forecast month: {forecast_month:%Y-%m}")
    print(f"Games forecast: {len(forecasts)}")


if __name__ == "__main__":
    main()