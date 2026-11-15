import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.modeling import manage_model, register_model


class ModelRegistryTest(unittest.TestCase):
    def test_rejection_activation_and_rollback(self):
        with TemporaryDirectory(dir=register_model.ROOT) as folder:
            registry = Path(folder)
            pointer = registry / "active.json"

            with (
                patch.object(register_model, "REGISTRY", registry),
                patch.object(manage_model, "REGISTRY", registry),
                redirect_stdout(StringIO()),
            ):
                original = register_model.save_model_version(
                    None, {"model": "baseline"}
                )
                register_model.write_atomic(
                    pointer, json.dumps({"model_id": original}).encode("utf-8")
                )
                rejected = register_model.save_model_version(
                    None, {"decision": "rejected", "compared_against": original}
                )
                approved = register_model.save_model_version(
                    None, {"decision": "approved", "compared_against": original}
                )

                with (
                    patch("sys.argv", ["manage_model", "--activate", rejected]),
                    redirect_stderr(StringIO()) as error,
                    self.assertRaises(SystemExit) as stopped,
                ):
                    manage_model.main()

                self.assertEqual(stopped.exception.code, 1)
                self.assertIn("Кандидат не прошёл допуск", error.getvalue())
                self.assertEqual(json.loads(pointer.read_text())["model_id"], original)

                with patch("sys.argv", ["manage_model", "--activate", approved]):
                    manage_model.main()

                self.assertEqual(
                    json.loads(pointer.read_text()),
                    {"model_id": approved, "previous_model_id": original},
                )

                with patch("sys.argv", ["manage_model", "--rollback"]):
                    manage_model.main()

                self.assertEqual(
                    json.loads(pointer.read_text()),
                    {"model_id": original, "previous_model_id": approved},
                )