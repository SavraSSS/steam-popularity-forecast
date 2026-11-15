import argparse
import hashlib
import json
import re

from .register_model import REGISTRY, write_atomic


def read_version(model_id):
    if not re.fullmatch(r"[0-9a-f]{16}", model_id):
        raise ValueError("ID версии должен состоять из 16 шестнадцатеричных символов")

    folder = REGISTRY / model_id
    model = (folder / "model.joblib").read_bytes()
    metadata = (folder / "metadata.json").read_bytes()
    if hashlib.sha256(model + metadata).hexdigest()[:16] != model_id:
        raise ValueError(f"Контрольная сумма версии {model_id} не совпала")

    return json.loads(metadata)


def main():
    parser = argparse.ArgumentParser(description="Управление активной моделью")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--activate", metavar="ID")
    actions.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    path = REGISTRY / "active.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        current = state["model_id"]

        if args.rollback:
            destination = state.get("previous_model_id")
            if not destination:
                raise ValueError("В реестре пока нет предыдущей активной версии")
            read_version(destination)
        else:
            destination = args.activate
            candidate = read_version(destination)
            if candidate.get("decision") != "approved":
                raise ValueError("Кандидат не прошёл допуск")
            if candidate.get("compared_against") != current:
                raise ValueError("Кандидат проверялся с другой активной версией")

        state = {"model_id": destination, "previous_model_id": current}
        write_atomic(path, json.dumps(state, indent=2).encode("utf-8"))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    print(f"Active version: {destination}")
    print(f"Previous version: {current}")


if __name__ == "__main__":
    main()