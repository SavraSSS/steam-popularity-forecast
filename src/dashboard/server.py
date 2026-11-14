import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os
import pandas as pd

from datetime import datetime, timezone
from tempfile import NamedTemporaryFile

from src.modeling.evaluate_baseline import ROOT
from src.modeling.search_models import JOURNAL


PAGE = Path(__file__).with_name("index.html")
ARTIFACTS = ROOT / "artifacts"
SNAPSHOT = ARTIFACTS / "dashboard.json"


def load_results():
    files = {
        "ranking": "latest_forecasts.csv",
        "metrics": "test_metrics.csv",
        "predictions": "test_predictions.csv",
    }
    results = {}
    for key, filename in files.items():
        frame = pd.read_csv(ARTIFACTS / filename)
        results[key] = json.loads(frame.to_json(orient="records"))

    experiments = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in JOURNAL.glob("*.json")
    ]
    results["experiments"] = sorted(
        experiments, key=lambda row: (row["mae"], row["id"])
    )
    results["published_at"] = datetime.now(timezone.utc).isoformat()
    return json.dumps(results, ensure_ascii=False).encode("utf-8")

def publish_results():
    content = load_results()
    with NamedTemporaryFile(dir=ARTIFACTS, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)

    try:
        temporary.replace(SNAPSHOT)
    finally:
        temporary.unlink(missing_ok=True)

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = PAGE.read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/data.json":
            content = SNAPSHOT.read_bytes()
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return

        self.send_response(200)
        if self.path == "/data.json":
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    if not SNAPSHOT.exists():
        publish_results()

    try:
        port = int(os.environ.get("PAS_DASHBOARD_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("PAS_DASHBOARD_PORT должен быть целым числом от 1 до 65535")

    with ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler) as server:
        print(f"Панель доступна по адресу http://127.0.0.1:{port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()