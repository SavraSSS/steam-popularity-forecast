import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd

from .evaluate_baseline import ROOT, load_splits
from .monthly_models import evaluate_test, fit_model, metrics, validate_models


REGISTRY = ROOT / "artifacts" / "model_registry"


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)

    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def save_model_version(model, metadata):
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    content = buffer.getvalue()
    info = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    model_id = hashlib.sha256(content + info).hexdigest()[:16]

    folder = REGISTRY / model_id
    write_atomic(folder / "model.joblib", content)
    write_atomic(folder / "metadata.json", info)
    return model_id

def main():
    train, validation, test = load_splits()
    development = pd.concat([train, validation])
    forecasts = validate_models(development)
    ranking = pd.DataFrame([
        {"model": name, **metrics(rows)}
        for name, rows in forecasts.groupby("model")
    ]).sort_values(["game_mae", "rmse", "model"])

    selected = ranking.iloc[0]["model"]
    test_metrics = evaluate_test(development, test)

    production_data = pd.concat([development, test])
    cutoff = production_data["month"].max() + pd.offsets.MonthBegin(1)
    model = fit_model(selected, production_data)

    if model is None:
        parameters = {}
    elif hasattr(model, "steps"):
        parameters = {
            name: step.get_params(deep=False) for name, step in model.steps
        }
    else:
        parameters = model.get_params(deep=False)

    metadata = {
        "model": selected,
        "trained_through": f"{cutoff:%Y-%m-%d}",
        "training_rows": len(production_data),
        "features": [] if model is None else list(model.feature_names_in_),
        "parameters": parameters,
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn")
        },
        "validation": ranking.to_dict(orient="records"),
        "historical_test": test_metrics.to_dict(orient="records"),
    }

    model_id = save_model_version(model, metadata)

    active = REGISTRY / "active.json"
    if not active.exists():
        write_atomic(active, json.dumps({"model_id": model_id}).encode("utf-8"))

    active_id = json.loads(active.read_text(encoding="utf-8"))["model_id"]
    print(f"Model: {selected}")
    print(f"Model version: {model_id}")
    print(f"Training targets through: {cutoff:%Y-%m}")
    print(f"Training rows: {len(production_data)}")
    print(f"Active version: {active_id}")


if __name__ == "__main__":
    main()