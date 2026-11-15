import json
from pathlib import Path

import pandas as pd

from ..modeling.register_model import write_atomic


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ACTIVITY_DIR = RAW / "steam_charts_monthly"
LOAD_LOG = RAW / "steam_charts_loads.jsonl"
COLUMNS = ["appid", "month", "Avg_players", "Peak_Players"]


def read_monthly_activity():
    files = sorted(ACTIVITY_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def save_monthly_activity(history):
    added = 0
    for appid, rows in history[COLUMNS].groupby("appid"):
        path = ACTIVITY_DIR / f"{int(appid)}.parquet"
        rows = rows.copy()
        rows["month"] = pd.to_datetime(rows["month"])
        rows = rows.drop_duplicates("month")

        if path.exists():
            existing = pd.read_parquet(path)
            rows = rows.loc[~rows["month"].isin(existing["month"])]
            if rows.empty:
                continue
            merged = pd.concat([existing, rows], ignore_index=True)
        else:
            merged = rows

        content = merged.sort_values("month").to_parquet(index=False)
        write_atomic(path, content)
        added += len(rows)

    return added


def log_load(appid, started_at, status, received, added, detail=""):
    record = {
        "appid": appid,
        "started_at": started_at,
        "status": status,
        "received": received,
        "added": added,
        "detail": detail,
    }
    LOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOAD_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")