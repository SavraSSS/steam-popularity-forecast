import argparse
import subprocess
import sys
from pathlib import Path
from src.dashboard.server import publish_results

ROOT = Path(__file__).resolve().parents[1]


def run_step(title, module, *args):
    print(f"\n{title}", flush=True)
    subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Обновить данные и пересчитать прогноз Steam"
    )
    parser.add_argument("appids", nargs="+", type=int, metavar="appid")
    args = parser.parse_args()
    if any(appid <= 0 for appid in args.appids):
        parser.error("appid должен быть положительным")

    try:
        run_step(
            "Загрузка месячной истории",
            "src.ingestion.steam_charts",
            *map(str, args.appids),
            "--save",
        )
        run_step("Пересборка витрины", "src.preprocessing.build_dataset")
        run_step("Подбор моделей на валидации", "src.modeling.search_models")
        run_step("Расчёт тестовых метрик", "src.modeling.compare_models")
        run_step("Прогноз на следующий месяц", "src.modeling.forecast_next_month")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, "\nОбновление прервано. См. ошибку выше.\n")
    publish_results()
    print("\nОбновление завершено.")


if __name__ == "__main__":
    main()