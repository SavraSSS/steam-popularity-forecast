import json
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
HISTORY = ROOT / "data" / "processed" / "prepared_dataset.csv"
REGISTRY = ARTIFACTS / "model_registry"

MODEL_NAMES = {
    "baseline": "Базовый прогноз",
    "neighbors": "Ближайшие соседи",
    "random_forest": "Случайный лес",
}
STATUS_NAMES = {
    "scored": "Сверен",
    "pending": "Ожидает факта",
    "no_alert": "Ухудшение не обнаружено",
    "retrain_recommended": "Обнаружено ухудшение",
    "no_drift": "Сдвиг признаков не обнаружен",
    "input_drift": "Обнаружен сдвиг признаков",
    "insufficient_data": "Недостаточно данных",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def show_game(history, forecast):
    observed = history.dropna(subset=["Avg_players"])
    names = (
        observed.dropna(subset=["Game_Name"])
        .sort_values("month")
        .drop_duplicates("appid", keep="last")
        .set_index("appid")["Game_Name"]
    )
    names = names.reindex(observed["appid"].unique())
    names = names.fillna(names.index.to_series().map(lambda value: f"Steam {value}"))
    names.update(forecast["Game_Name"])

    options = list(forecast.index) + sorted(
        names.index.difference(forecast.index),
        key=lambda value: names.at[value].casefold(),
    )
    appid = st.selectbox(
        "Игра",
        options=options,
        format_func=lambda value: names.at[value],
        key="selected_game",
    )

    game = history.loc[history["appid"].eq(appid)]
    row = forecast.loc[appid] if appid in forecast.index else None

    if row is None:
        last_month = observed.loc[observed["appid"].eq(appid), "month"].max()
        st.info(f"Последний доступный месяц — {last_month:%m.%Y}. Показываем историю игры.")
    else:
        current, predicted, change = st.columns(3)
        current.metric("Последний средний онлайн", f"{row['current_players']:,.0f}")
        predicted.metric("Прогноз среднего онлайна", f"{row['forecast_players']:,.0f}")
        pct = row["change_pct"]
        change.metric("Изменение", f"{pct:+.2f}%" if pd.notna(pct) else "—")
        st.caption(
            f"{MODEL_NAMES.get(row['model'], row['model'])} · Версия {row['model_version']}"
        )
        game = game.loc[game["month"].le(row["month"])]
    chart = (
        game.sort_values("month").tail(12).set_index("month")
        [["Avg_players", "Peak_Players"]]
        .rename(columns={"Avg_players": "Средний онлайн", "Peak_Players": "Пиковый онлайн"})
    )
    chart["Прогноз среднего"] = float("nan")
    if row is not None:
        chart.loc[row["month"], "Прогноз среднего"] = row["current_players"]
        chart.loc[row["forecast_month"], "Прогноз среднего"] = row["forecast_players"]
    series = (
        chart.sort_index().reset_index()
        .melt(id_vars="month", var_name="series", value_name="players")
        .dropna(subset=["players"])
    )
    colors = alt.Scale(
        domain=["Средний онлайн", "Пиковый онлайн", "Прогноз среднего"],
        range=["#234A85", "#69788D", "#B56816"],
    )
    line = alt.Chart(series).mark_line(point=True).encode(
        x=alt.X(
            "month:T",
            title=None,
            axis=alt.Axis(format="%m.%Y", tickCount=7),
            scale=alt.Scale(nice=False),
        ),
        y=alt.Y("players:Q", title="Игроки", scale=alt.Scale(zero=True)),
        color=alt.Color("series:N", title=None, scale=colors),
        strokeDash=alt.StrokeDash(
            "series:N",
            scale=alt.Scale(
                domain=["Средний онлайн", "Пиковый онлайн", "Прогноз среднего"],
                range=[[1, 0], [1, 0], [6, 4]],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("month:T", title="Месяц", format="%m.%Y"),
            alt.Tooltip("series:N", title="Показатель"),
            alt.Tooltip("players:Q", title="Игроки", format=",.0f"),
        ],
    ).properties(height=320)
    st.altair_chart(line, width="stretch")
    return appid


def show_monitoring(appid):
    actual_path = ARTIFACTS / "forecast_monitoring.csv"
    if actual_path.is_file():
        monitored = pd.read_csv(actual_path)
        pending = monitored.loc[
            monitored["appid"].eq(appid) & monitored["status"].eq("pending"),
            "forecast_month",
        ]
        if not pending.empty:
            months = ", ".join(pending.str.slice(0, 7).sort_values().unique())
            st.caption(f"Выпуски за {months} ожидают фактических данных.")

    sources = [
        (actual_path, "Сохранённые выпуски, по которым поступил факт"),
        (
            ARTIFACTS / "forecast_monitoring_demo.csv",
            "Историческая проверка модели · август–октябрь 2026",
        ),
    ]
    for path, caption in sources:
        if not path.is_file():
            continue

        monitored = pd.read_csv(path)
        game = monitored.loc[
            monitored["appid"].eq(appid) & monitored["status"].eq("scored")
        ].copy()
        if game.empty:
            continue

        game["forecast_month"] = game["forecast_month"].str.slice(0, 7)
        game["direction_correct"] = (
            game["direction_correct"].astype("string").str.lower()
            .map({"true": "Да", "false": "Нет"})
        )
        labels = {
            "forecast_month": "Месяц",
            "forecast_players": "Прогноз",
            "actual_players": "Факт",
            "absolute_error": "Ошибка, игроков",
            "direction_correct": "Направление угадано",
        }
        st.caption(caption)
        st.dataframe(
            game.sort_values("forecast_month", ascending=False)[list(labels)]
            .rename(columns=labels).round(2),
            hide_index=True,
        )
        break
    else:
        st.caption("Для этой игры пока нет прогнозов с известным фактом.")

    quality_path = ARTIFACTS / "forecast_quality.json"
    if quality_path.is_file():
        quality = read_json(quality_path)
        st.caption(
            f"Версия {quality['model_version']} · {STATUS_NAMES[quality['decision']]} "
            f"· {STATUS_NAMES[quality['input_drift']]}"
        )


def show_model_quality():
    active = read_json(REGISTRY / "active.json")["model_id"]
    info = read_json(REGISTRY / active / "metadata.json")

    with st.expander("Качество активной модели"):
        st.caption(f"Версия {active}")
        if "historical_test" in info:
            table = pd.DataFrame(info["historical_test"])
            table["model"] = table["model"].map(MODEL_NAMES)
            labels = {
                "model": "Модель",
                "game_mae": "MAE по играм",
                "mae": "MAE",
                "rmse": "RMSE",
                "direction_pct": "Направление, %",
            }
            st.dataframe(table[list(labels)].rename(columns=labels).round(2), hide_index=True)
        elif "check_mae_by_game" in info:
            scores = info["check_mae_by_game"]
            st.caption(f"Период допуска · {', '.join(info['check_months'])}")
            st.dataframe(pd.DataFrame([{
                "MAE кандидата": scores["candidate"],
                "MAE предыдущей модели": scores["active"],
                "MAE базового прогноза": scores["baseline"],
            }]).round(2), hide_index=True)


@st.fragment(run_every="30s")
def dashboard():
    path = ARTIFACTS / "latest_forecasts.csv"
    if not path.is_file() or not HISTORY.is_file():
        st.info("Сначала выполни python -m src.update из корня проекта.")
        return

    forecast = pd.read_csv(path, parse_dates=["month", "forecast_month"]).set_index("appid")
    if forecast.empty:
        st.info("В текущем выпуске нет прогнозов.")
        return

    history = pd.read_csv(
        HISTORY,
        usecols=["appid", "Game_Name", "month", "Avg_players", "Peak_Players"],
        parse_dates=["month"],
    )
    st.subheader(f"Рост и спад онлайна · {forecast['forecast_month'].max():%Y-%m}")

    order = st.radio("Сортировка", ["Сначала рост", "Сначала спад"], horizontal=True)
    labels = {
        "Game_Name": "Игра",
        "current_players": "Последний онлайн",
        "forecast_players": "Прогноз",
        "change_players": "Изменение, игроков",
        "change_pct": "Изменение, %",
    }
    ranked = forecast.sort_values("change_pct", ascending=order == "Сначала спад")
    st.dataframe(ranked[list(labels)].rename(columns=labels).round(2), hide_index=True)

    appid = show_game(history, forecast)
    show_monitoring(appid)
    if (REGISTRY / "active.json").is_file():
        show_model_quality()


st.set_page_config(page_title="Динамика игр Steam", layout="wide")
st.title("Динамика игр Steam")
dashboard()