import argparse
from datetime import date, datetime, timezone
from html.parser import HTMLParser

import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sqlite3
from contextlib import closing
from pathlib import Path

import time

MONTHS = {
    name: number
    for number, name in enumerate(
        "January February March April May June July August "
        "September October November December".split(),
        start=1,
    )
}

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "raw" / "steam_charts.sqlite"

class ActivityTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def fetch_monthly_activity(appid):
    url = f"https://steamcharts.com/app/{appid}"
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=(5, 20))
        response.raise_for_status()

    parser = ActivityTableParser()
    parser.feed(response.text)
    current_month = datetime.now(timezone.utc).date().replace(day=1)

    records = []
    for cells in parser.rows:
        if len(cells) != 5:
            continue

        label = cells[0].split()
        if len(label) != 2 or label[0] not in MONTHS or not label[1].isdigit():
            continue

        month = date(int(label[1]), MONTHS[label[0]], 1)
        if month >= current_month:
            continue

        average = float(cells[1].replace(",", ""))
        peak = int(cells[4].replace(",", ""))
        if not 0 <= average <= peak:
            raise ValueError(f"Некорректный онлайн для {appid} за {month}")

        records.append({
            "appid": appid,
            "month": month,
            "Avg_players": average,
            "Peak_Players": peak,
        })

    if not records:
        raise ValueError(f"Steam Charts не вернул месячную историю для {appid}")

    return pd.DataFrame(records).sort_values("month").reset_index(drop=True)

def save_monthly_activity(history):
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    records = [
        (int(row.appid), row.month.isoformat(), float(row.Avg_players), int(row.Peak_Players))
        for row in history.itertuples(index=False)
    ]

    with closing(sqlite3.connect(DATABASE)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS monthly_activity (
                    appid INTEGER NOT NULL,
                    month TEXT NOT NULL,
                    avg_players REAL NOT NULL,
                    peak_players INTEGER NOT NULL,
                    PRIMARY KEY (appid, month)
                )
            """)
            result = connection.executemany("""
                INSERT INTO monthly_activity (appid, month, avg_players, peak_players)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (appid, month) DO NOTHING
            """, records)
            return result.rowcount

def log_load(appid, started_at, status, received, added, detail=""):
    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(DATABASE)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS load_log (
                    id INTEGER PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    appid INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    received INTEGER NOT NULL,
                    added INTEGER NOT NULL,
                    detail TEXT NOT NULL
                )
            """)
            connection.execute("""
                INSERT INTO load_log
                    (started_at, appid, status, received, added, detail)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (started_at, appid, status, received, added, detail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("appids", nargs="+", type=int, metavar="appid")
    parser.add_argument(
        "--save", action="store_true", help="Сохранить историю в SQLite"
    )
    args = parser.parse_args()
    if any(appid <= 0 for appid in args.appids):
        parser.error("appid должен быть положительным")

    appids = list(dict.fromkeys(args.appids))
    failed = []

    for index, appid in enumerate(appids):
        if index > 0:
            time.sleep(1)

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            history = fetch_monthly_activity(appid)
        except (requests.RequestException, ValueError) as error:
            if args.save:
                log_load(appid, started_at, "error", 0, 0, str(error))
            failed.append(appid)
            print(f"{appid}: ошибка загрузки — {error}")
            continue

        print(f"{appid}: месяцев получено — {len(history)}")
        print(history.tail(5).to_string(index=False))

        if args.save:
            added = save_monthly_activity(history)
            log_load(appid, started_at, "ok", len(history), added)
            print(f"Новых месяцев сохранено: {added}")

    if failed:
        parser.exit(1, f"Не удалось загрузить игры: {', '.join(map(str, failed))}\n")


if __name__ == "__main__":
    main()