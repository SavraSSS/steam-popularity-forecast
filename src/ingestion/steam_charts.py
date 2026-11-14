import argparse
from datetime import date, datetime, timezone
from html.parser import HTMLParser

import pandas as pd
import requests

MONTHS = {
    name: number
    for number, name in enumerate(
        "January February March April May June July August "
        "September October November December".split(),
        start=1,
    )
}


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
    response = requests.get(url, timeout=(5, 20))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int)
    args = parser.parse_args()
    if args.appid <= 0:
        parser.error("appid должен быть положительным")

    try:
        history = fetch_monthly_activity(args.appid)
    except (requests.RequestException, ValueError) as error:
        parser.exit(1, f"Ошибка загрузки: {error}\n")

    print(f"Месяцев получено: {len(history)}")
    print(history.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()