import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from src.modeling.evaluate_baseline import ROOT


PAGE = Path(__file__).with_name("index.html")
ARTIFACTS = ROOT / "artifacts"


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
    return json.dumps(results, ensure_ascii=False).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = PAGE.read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/data.json":
            content = self.server.results
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    results = load_results()
    with ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler) as server:
        server.results = results
        print("Панель доступна по адресу http://127.0.0.1:8000", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()