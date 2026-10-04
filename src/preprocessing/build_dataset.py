from pathlib import Path
import csv

import kagglehub
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    print("Получение данных...")

    player_path = Path(
        kagglehub.dataset_download(
            "jackogozaly/steam-player-data"
        )
    )

    games_path = Path(
        kagglehub.dataset_download(
            "fronkongames/steam-games-dataset"
        )
    )

    reviews_path = Path(
        kagglehub.dataset_download(
            "forgemaster/steam-reviews-dataset",
            path="reviews-1-115.csv"
        )
    )

    player_df = pd.read_csv(
        player_path / "Valve_Player_Data.csv"
    )

    games_file = games_path / "games.csv"

    with open(
        games_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        header = next(csv.reader(file))

    fixed_header = []

    for column in header:
        if column == "DiscountDLC count":
            fixed_header.extend(
                ["Discount", "DLC count"]
            )
        else:
            fixed_header.append(column)

    games_df = pd.read_csv(
        games_file,
        names=fixed_header,
        skiprows=1,
        low_memory=False
    )

    reviews_df = pd.read_csv(
        reviews_path,
        compression="zip"
    )

    print("Данные получены")

    return player_df, games_df, reviews_df


def prepare_player(player_df):
    print("Обработка PLAYER...")

    player = player_df.copy()

    player["appid"] = (
        player["URL"]
        .str.extract(r"/app/(\d+)")
        .astype("Int64")
    )

    player["Date"] = pd.to_datetime(
        player["Date"],
        errors="coerce"
    )

    player = player.dropna(
        subset=[
            "appid",
            "Date",
            "Avg_players",
            "Peak_Players"
        ]
    )

    player["month"] = (
        player["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    player = (
        player
        .sort_values(["appid", "month"])
        .reset_index(drop=True)
    )

    return player


def owners_to_numeric(value):
    if pd.isna(value):
        return None

    parts = (
        str(value)
        .replace(",", "")
        .split("-")
    )

    if len(parts) != 2:
        return None

    lower = float(parts[0].strip())
    upper = float(parts[1].strip())

    return (lower + upper) / 2


def prepare_games(games_df, player_ids):
    print("Обработка GAMES...")

    games = games_df.rename(
        columns={"AppID": "appid"}
    ).copy()

    games = games[
        games["appid"].isin(player_ids)
    ].copy()

    game_features = [
        "appid",
        "Name",
        "Release date",
        "Estimated owners",
        "Required age",
        "Price",
        "Discount",
        "DLC count",
        "Windows",
        "Mac",
        "Linux",
        "Metacritic score",
        "Positive",
        "Negative",
        "Recommendations",
        "Developers",
        "Publishers",
        "Categories",
        "Genres",
        "Tags"
    ]

    games = games[game_features].copy()

    games["Release date"] = pd.to_datetime(
        games["Release date"],
        format="mixed",
        errors="coerce"
    )

    games["Estimated owners"] = (
        games["Estimated owners"]
        .apply(owners_to_numeric)
    )

    games["Categories"] = (
        games["Categories"]
        .fillna("Unknown")
    )

    games["Genres"] = (
        games["Genres"]
        .fillna("Unknown")
    )

    games["Tags"] = (
        games["Tags"]
        .fillna("Unknown")
    )

    return games


def prepare_reviews(reviews_df, player_ids):
    print("Обработка REVIEWS...")

    reviews = reviews_df.copy()

    reviews = reviews.drop_duplicates()

    reviews["review_date"] = pd.to_datetime(
        reviews["unix_timestamp_created"],
        unit="s",
        errors="coerce"
    )

    reviews = reviews[
        reviews["appid"].isin(player_ids)
    ].copy()

    reviews["month"] = (
        reviews["review_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    reviews["voted_up"] = (
        reviews["voted_up"]
        .astype(int)
    )

    reviews_monthly = (
        reviews
        .groupby(["appid", "month"])
        .agg(
            review_count=("review", "size"),
            positive_share=("voted_up", "mean"),
            avg_playtime=("playtime_forever", "mean"),
            avg_votes_up=("votes_up", "mean")
        )
        .reset_index()
    )

    return reviews_monthly


def regularize_player_history(player):
    print("Восстановление календарной сетки...")

    full_rows = []

    for appid, group in player.groupby("appid"):
        group = (
            group
            .sort_values("month")
            .copy()
        )

        full_months = pd.date_range(
            start=group["month"].min(),
            end=group["month"].max(),
            freq="MS"
        )

        group = (
            group
            .set_index("month")
            .reindex(full_months)
            .rename_axis("month")
            .reset_index()
        )

        group["appid"] = appid

        full_rows.append(group)

    return pd.concat(
        full_rows,
        ignore_index=True
    )


def build_features(
    player,
    games,
    reviews
):
    print("Формирование итоговой витрины...")

    dataset = player.merge(
        reviews,
        on=["appid", "month"],
        how="left"
    )

    dataset = dataset.merge(
        games,
        on="appid",
        how="left"
    )

    dataset = (
        dataset
        .sort_values(["appid", "month"])
        .reset_index(drop=True)
    )

    grouped = dataset.groupby("appid")

    # Целевая переменная - среднее число игроков в следующем месяце
    dataset["target_next_month"] = (
        grouped["Avg_players"]
        .shift(-1)
    )

    # Лаговые признаки
    dataset["avg_players_lag_1"] = (
        grouped["Avg_players"].shift(1)
    )

    dataset["avg_players_lag_3"] = (
        grouped["Avg_players"].shift(3)
    )

    dataset["avg_players_lag_6"] = (
        grouped["Avg_players"].shift(6)
    )

    # Скользящие признаки
    dataset["avg_players_roll_3"] = (
        grouped["Avg_players"]
        .transform(
            lambda x:
            x.shift(1).rolling(3).mean()
        )
    )

    dataset["avg_players_roll_6"] = (
        grouped["Avg_players"]
        .transform(
            lambda x:
            x.shift(1).rolling(6).mean()
        )
    )

    # Календарные признаки
    dataset["year"] = dataset["month"].dt.year
    dataset["month_num"] = dataset["month"].dt.month
    dataset["quarter"] = dataset["month"].dt.quarter

    # Возраст игры
    dataset["game_age_months"] = (
        (
            dataset["month"].dt.year
            - dataset["Release date"].dt.year
        ) * 12
        + dataset["month"].dt.month
        - dataset["Release date"].dt.month
    )

    return dataset


def make_model_dataset(dataset):
    print("Подготовка ML-выборки...")

    model_dataset = dataset.dropna(
        subset=[
            "Avg_players",
            "target_next_month",
            "avg_players_lag_1",
            "avg_players_lag_3",
            "avg_players_lag_6",
            "avg_players_roll_3",
            "avg_players_roll_6"
        ]
    ).copy()

    return model_dataset


def save_data(dataset, model_dataset):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    full_path = (
        PROCESSED_DIR
        / "prepared_dataset.csv"
    )

    model_path = (
        PROCESSED_DIR
        / "model_dataset.csv"
    )

    dataset.to_csv(
        full_path,
        index=False
    )

    model_dataset.to_csv(
        model_path,
        index=False
    )

    print("\nДанные сохранены:")
    print(full_path)
    print(model_path)


def main():
    player_df, games_df, reviews_df = load_data()

    player = prepare_player(player_df)

    player_ids = set(
        player["appid"]
        .dropna()
        .astype(int)
    )

    games = prepare_games(
        games_df,
        player_ids
    )

    reviews = prepare_reviews(
        reviews_df,
        player_ids
    )

    player_regular = regularize_player_history(
        player
    )

    dataset = build_features(
        player_regular,
        games,
        reviews
    )

    model_dataset = make_model_dataset(
        dataset
    )

    save_data(
        dataset,
        model_dataset
    )

    print("\nИтог:")
    print(
        "Исходных игр:",
        player["appid"].nunique()
    )
    print(
        "Строк полной витрины:",
        len(dataset)
    )
    print(
        "Строк ML-выборки:",
        len(model_dataset)
    )
    print(
        "Игр в ML-выборке:",
        model_dataset["appid"].nunique()
    )


if __name__ == "__main__":
    main()