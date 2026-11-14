import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compare_review_features import make_features
from .evaluate_baseline import DATASET, ROOT, TARGET, calculate_metrics, load_splits

import argparse
import hashlib
import json

SEED = 42
JOURNAL = ROOT / "artifacts" / "experiments"
PARAMETERS = {
    "ridge": [{"alpha": 5}, {"alpha": 20}],
    "random_forest": [
        {"n_estimators": 100, "min_samples_leaf": 20},
        {"n_estimators": 100, "min_samples_leaf": 40},
    ],
}


def build_model(model_name, params, seed=SEED):
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    raise ValueError(f"Неизвестная модель: {model_name}")


def predict_config(train, evaluation, model_name, params, include_text, seed=SEED):
    train_features = make_features(train, include_text, include_month=False)
    evaluation_features = make_features(evaluation, include_text, include_month=False)

    model = build_model(model_name, params, seed)
    train_change = np.log1p(train[TARGET]) - np.log1p(train["Avg_players"])
    model.fit(train_features, train_change)

    change = model.predict(evaluation_features)
    current = np.log1p(evaluation["Avg_players"])
    return np.expm1(current + change).clip(lower=0)


def evaluate_config(train, validation, model_name, params, include_text, seed=SEED):
    predicted = predict_config(
        train, validation, model_name, params, include_text, seed
    )
    return calculate_metrics(validation[TARGET], predicted)

def snapshot_dataset():
    content = DATASET.read_bytes()
    version = hashlib.sha256(content).hexdigest()

    JOURNAL.mkdir(parents=True, exist_ok=True)
    snapshot = JOURNAL / f"{version}.csv"
    if not snapshot.exists():
        snapshot.write_bytes(content)

    return version, snapshot


def replay_experiment(experiment_id):
    record_path = JOURNAL / f"{experiment_id}.json"
    if not record_path.is_file():
        raise ValueError(f"Эксперимент {experiment_id} не найден")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    version = record["dataset_sha256"]
    snapshot = JOURNAL / f"{version}.csv"

    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != version:
        raise ValueError("Контрольная сумма сохранённой выборки не совпала")

    train, validation, _ = load_splits(snapshot)
    mae, rmse = evaluate_config(
        train, validation, record["model"], record["params"],
        record["text"], record["seed"],
    )

    print(f"Experiment: {experiment_id}")
    print(f"Saved:  MAE={record['mae']:.2f}, RMSE={record['rmse']:.2f}")
    print(f"Replay: MAE={mae:.2f}, RMSE={rmse:.2f}")
    if not np.allclose(
        [mae, rmse], [record["mae"], record["rmse"]], rtol=0, atol=1e-8
    ):
        raise ValueError("Повторный расчёт не совпал с журналом")

def main():
    parser = argparse.ArgumentParser(description="Подбор и повтор экспериментов")
    parser.add_argument("--replay", metavar="ID")
    args = parser.parse_args()

    if args.replay:
        try:
            replay_experiment(args.replay)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        return

    version, snapshot = snapshot_dataset()
    train, validation, _ = load_splits(snapshot)
    baseline_mae, baseline_rmse = calculate_metrics(
        validation[TARGET], validation["Avg_players"]
    )

    results = []
    for model_name, options in PARAMETERS.items():
        for params in options:
            for include_text in (False, True):
                mae, rmse = evaluate_config(
                    train, validation, model_name, params, include_text
                )
                config = {
                    "model": model_name,
                    "params": params,
                    "text": include_text,
                    "seed": SEED,
                    "dataset_sha256": version,
                }
                encoded = json.dumps(config, sort_keys=True).encode("utf-8")
                experiment_id = hashlib.sha256(encoded).hexdigest()[:16]
                record = {
                    "id": experiment_id,
                    **config,
                    "mae": float(mae),
                    "rmse": float(rmse),
                }
                (JOURNAL / f"{experiment_id}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
                results.append({
                    "id": experiment_id,
                    "model": model_name,
                    "params": params,
                    "text": include_text,
                    "mae": mae,
                    "rmse": rmse,
                })

    table = pd.DataFrame(results).sort_values("mae")
    print(f"Validation rows: {len(validation)}")
    print(f"Baseline: MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    best = table.iloc[0]
    winner = best["model"] if best["mae"] < baseline_mae else "baseline"
    print(f"Selected by validation MAE: {winner}")


if __name__ == "__main__":
    main()