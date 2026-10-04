from pathlib import Path
from datetime import datetime

import pandas as pd
import requests


STEAM_CURRENT_PLAYERS_URL = (
    "https://api.steampowered.com/"
    "ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
)

OUTPUT_PATH = Path("data/api/player_counts.csv")


def get_current_players(appid):
    response = requests.get(
        STEAM_CURRENT_PLAYERS_URL,
        params={"appid": appid},
        timeout=10
    )

    response.raise_for_status()

    data = response.json()["response"]

    return {
        "appid": appid,
        "timestamp": datetime.now(),
        "player_count": data["player_count"]
    }


def save_player_count(data):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    row = pd.DataFrame([data])

    row.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not OUTPUT_PATH.exists(),
        index=False
    )


if __name__ == "__main__":
    result = get_current_players(730)

    save_player_count(result)

    print(result)
    print(f"Данные сохранены в {OUTPUT_PATH}")