from pathlib import Path

import kagglehub


DATASETS = {
    "player_data": {
        "dataset": "jackogozaly/steam-player-data",
        "file": None,
    },
    "games": {
        "dataset": "fronkongames/steam-games-dataset",
        "file": None,
    },
    "reviews": {
        "dataset": "forgemaster/steam-reviews-dataset",
        "file": "reviews-1-115.csv",
    },
}


def download_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Неизвестный источник: {name}")

    config = DATASETS[name]

    print(f"\nЗагрузка источника: {name}")

    try:
        if config["file"] is None:
            path = kagglehub.dataset_download(
                config["dataset"]
            )
        else:
            path = kagglehub.dataset_download(
                config["dataset"],
                path=config["file"]
            )

        path = Path(path)

        print(f"Успешно: {path}")

        return path

    except Exception as error:
        print(f"Ошибка при загрузке {name}: {error}")
        return None


def download_all():
    paths = {}

    for name in DATASETS:
        paths[name] = download_dataset(name)

    return paths


if __name__ == "__main__":
    paths = download_all()

    print("\nРезультаты загрузки:")

    for name, path in paths.items():
        status = "OK" if path is not None else "ERROR"
        print(f"{name}: {status}")