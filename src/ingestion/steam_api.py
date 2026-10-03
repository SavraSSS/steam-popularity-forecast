import requests
from datetime import datetime


STEAM_CURRENT_PLAYERS_URL = (
    "https://api.steampowered.com/"
    "ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
)


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


if __name__ == "__main__":
    result = get_current_players(730)
    print(result)