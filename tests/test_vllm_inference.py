"""Focused Phase 6 CPU tests with mocked HTTP; never launch vLLM or load a model."""
import base64
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from src.inference.remote_client import RemoteVLMError
from src.inference.vlm_client import (
    ADAPTER_SERVED_MODEL, BASE_MODEL_ID, BASE_SERVED_MODEL,
    VLMClient, VLMConfig, select_model, validate_csv,
)
from src.inference.vllm_server import build_command, check_environment, parser, resolve_adapter
from src.prompts.extraction import INSTRUCTION

TARGET = 'Item,Quantity,Rate,Amount\n"Consultation, specialist",1,,500\n'


class VLLMInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "invoice.png"
        Image.new("RGB", (32, 32), "white").save(self.image)

    def transport(self, mode="adapter", content=TARGET, finish_reason="stop", returned_model=None):
        model = select_model(mode)
        def respond(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [
                    {"id": BASE_SERVED_MODEL, "root": BASE_MODEL_ID, "parent": None},
                    {"id": ADAPTER_SERVED_MODEL, "root": "/test/adapter", "parent": BASE_SERVED_MODEL},
                ]})
            self.assertEqual(json.loads(request.content)["model"], model)
            return httpx.Response(200, json={"model": returned_model or model,
                "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                "usage": {"completion_tokens": 20}})
        return httpx.Client(transport=httpx.MockTransport(respond))

    def test_real_image_request_uses_training_instruction(self):
        request = VLMClient(VLMConfig(mode="adapter")).build_request(self.image)
        content = request["messages"][0]["content"]
        self.assertEqual(content[1], {"type": "text", "text": INSTRUCTION})
        url = content[0]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.image.read_bytes())
        self.assertEqual((request["model"], request["temperature"]), (ADAPTER_SERVED_MODEL, 0))

    def test_image_validation_rejects_missing_invalid_and_oversized_input(self):
        with self.assertRaises(FileNotFoundError):
            VLMClient(VLMConfig(mode="base")).build_request(self.root / "missing.jpg")
        invalid = self.root / "invalid.jpg"
        invalid.write_text("not an image")
        with self.assertRaises(ValueError):
            VLMClient(VLMConfig(mode="base")).build_request(invalid)
        with self.assertRaises(ValueError):
            VLMClient(VLMConfig(mode="base", max_image_bytes=1)).build_request(self.image)
        with self.assertRaises(ValueError):
            VLMClient(VLMConfig(mode="base", max_input_pixels=1)).build_request(self.image)

    def test_csv_preserves_empty_rate_and_escaping(self):
        text, rows = validate_csv(TARGET)
        self.assertEqual(text, TARGET)
        self.assertEqual(rows[0]["Rate"], "")
        self.assertEqual(rows[0]["Item"], "Consultation, specialist")

    def test_malformed_csv_and_invalid_numbers_fail(self):
        for value in ('Bad,Header\nx,y\n', 'Item,Quantity,Rate,Amount\n"unfinished,1,,5\n',
                      'Item,Quantity,Rate,Amount\nECG,0,,5\n',
                      'Item,Quantity,Rate,Amount\nECG,1.5,,5\n',
                      'Item,Quantity,Rate,Amount\nECG,1,,NaN\n',
                      'Item,Quantity,Rate,Amount\nECG,1,6,5\n'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_csv(value)

    def test_http_inference_saves_actual_output_for_each_mode(self):
        for mode in ("base", "adapter"):
            with self.subTest(mode=mode), self.transport(mode) as http:
                result = VLMClient(VLMConfig(mode=mode), http).extract(self.image, self.root / mode)
                directory = Path(result["artifacts"])
                self.assertEqual((directory / "raw_output.txt").read_text(), TARGET)
                self.assertEqual((directory / "extracted.csv").read_text(), TARGET)
                self.assertEqual(json.loads((directory / "validation.json").read_text())["status"], "PASS")
                self.assertTrue((directory / "response.json").is_file())

    def test_bad_generation_preserves_raw_without_validated_csv(self):
        with self.transport(content="Not CSV") as http:
            with self.assertRaises(RemoteVLMError):
                VLMClient(VLMConfig(mode="adapter"), http).extract(self.image, self.root / "bad")
        directory = next((self.root / "bad").iterdir())
        self.assertEqual((directory / "raw_output.txt").read_text(), "Not CSV")
        self.assertFalse((directory / "extracted.csv").exists())
        self.assertEqual(json.loads((directory / "validation.json").read_text())["status"], "FAIL")

    def test_connection_failure_is_explicit(self):
        def disconnected(request):
            raise httpx.ConnectError("offline", request=request)
        with httpx.Client(transport=httpx.MockTransport(disconnected)) as http:
            with self.assertRaisesRegex(RemoteVLMError, "Cannot reach"):
                VLMClient(VLMConfig(mode="adapter"), http).extract(self.image, self.root / "offline")

    def test_model_selection_and_response_do_not_fall_back(self):
        self.assertEqual(select_model("base"), BASE_SERVED_MODEL)
        self.assertEqual(select_model("adapter"), ADAPTER_SERVED_MODEL)
        with self.assertRaises(ValueError):
            select_model("adapter", BASE_SERVED_MODEL)
        with self.transport(returned_model=BASE_SERVED_MODEL) as http:
            with self.assertRaisesRegex(RemoteVLMError, "differs"):
                VLMClient(VLMConfig(mode="adapter"), http).extract(self.image, self.root / "fallback")

    def test_base_alias_cannot_masquerade_as_adapter(self):
        def response(request):
            return httpx.Response(200, json={"data": [{"id": ADAPTER_SERVED_MODEL,
                                                       "root": BASE_MODEL_ID, "parent": None}]})
        with httpx.Client(transport=httpx.MockTransport(response)) as http:
            with self.assertRaisesRegex(RemoteVLMError, "not advertised"):
                VLMClient(VLMConfig(mode="adapter"), http).extract(self.image, self.root / "alias")

    def test_truncated_output_is_not_marked_valid(self):
        with self.transport(finish_reason="length") as http:
            with self.assertRaisesRegex(RemoteVLMError, "did not finish normally"):
                VLMClient(VLMConfig(mode="adapter"), http).extract(self.image, self.root / "truncated")

    def adapter_fixture(self):
        pointer = self.root / "latest_successful.json"
        pointer.write_text(json.dumps({"model_id": BASE_MODEL_ID}))
        directory = self.root / "runs/test/adapter"
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE_MODEL_ID, "r": 8}))
        (directory / "chat_template.jinja").write_text("test template")
        (directory.parent / "run_config.json").write_text('{"min_pixels":50176,"max_pixels":200704}')
        return pointer, directory

    def test_adapter_resolution_reuses_verified_phase5_pointer(self):
        pointer, adapter = self.adapter_fixture()
        with patch("src.trainer.sft.train.checkpoints.latest_adapter", return_value=adapter) as verify:
            result = resolve_adapter(pointer)
        verify.assert_called_once_with(pointer.parent)
        self.assertEqual((result["path"], result["rank"]), (adapter, 8))
        pointer.write_text('{"model_id":"wrong/model"}')
        with self.assertRaisesRegex(ValueError, "base-model"):
            resolve_adapter(pointer)

    def test_invalid_checkpoint_stops_adapter_resolution(self):
        pointer, _ = self.adapter_fixture()
        with patch("src.trainer.sft.train.checkpoints.latest_adapter", side_effect=RuntimeError("checksum mismatch")):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                resolve_adapter(pointer)

    def test_launcher_modes_are_explicit_and_loopback_only(self):
        base = build_command(parser().parse_args(["--mode", "base"]), "/test/vllm")
        self.assertEqual(base[base.index("--host")+1], "127.0.0.1")
        self.assertNotIn("--enable-lora", base)
        with patch("src.inference.vllm_server.resolve_adapter", return_value={
            "path": Path("/test/adapter"), "template": Path("/test/template"),
            "min_pixels": 50176, "max_pixels": 200704,
        }):
            command = build_command(parser().parse_args(["--mode", "adapter"]), "/test/vllm")
        self.assertIn("--enable-lora", command)
        lora = json.loads(command[command.index("--lora-modules")+1])
        self.assertEqual(lora["name"], ADAPTER_SERVED_MODEL)
        self.assertEqual(command[command.index("--max-lora-rank")+1], "8")

    def test_preflight_checks_serving_stack_without_training_only_packages(self):
        versions = {"vllm": "0.11.2", "torch": "2.9.0+cu128", "torchvision": "0.24.0+cu128",
                    "transformers": "4.57.6", "httpx": "0.28.1", "Pillow": "11.0.0"}
        torch = type("Torch", (), {
            "version": type("Version", (), {"cuda": "12.8"})(),
            "cuda": type("Cuda", (), {"is_available": lambda self: True,
                                         "device_count": lambda self: 1,
                                         "get_device_name": lambda self, index: "NVIDIA L4"})(),
        })()
        transformers = type("Transformers", (), {"Qwen2_5_VLForConditionalGeneration": object,
                                                    "AutoProcessor": object})()
        modules = {"torch": torch, "torchvision": object(), "transformers": transformers,
                   "vllm": object(), "httpx": object(), "PIL": object()}
        distribution = type("Distribution", (), {
            "requires": [], "locate_file": lambda self, _: Path("/fake/qwen2_5_vl.py")})()
        qwen_source = ('SupportsLoRA hf_to_vllm_mapper "model.language_model.": "language_model.model."')
        serving_prefix = Path(__file__).resolve().parents[1].parent / ".venv-vllm"
        imports = Mock(side_effect=modules.__getitem__)
        real_import_module = importlib.import_module
        with patch("src.inference.vllm_server.metadata.version", side_effect=versions.__getitem__) as version, \
                patch("src.inference.vllm_server.metadata.distribution", return_value=distribution), \
                patch("src.inference.vllm_server.importlib", SimpleNamespace(import_module=imports)), \
                patch("src.inference.vllm_server.Path.is_file", return_value=True), \
                patch("src.inference.vllm_server.Path.read_text", return_value=qwen_source), \
                patch("src.inference.vllm_server.sys.prefix", str(serving_prefix)), \
                patch("src.inference.vllm_server.sys.executable", str(serving_prefix / "bin/python")):
            # Replacing this module's importlib reference must not hijack the
            # stdlib importer used by unittest.mock to resolve later patches.
            self.assertIs(importlib.import_module, real_import_module)
            result = check_environment()
        self.assertIs(importlib.import_module, real_import_module)
        self.assertEqual(result["errors"], [])
        self.assertIn("unverified", result["status"])
        self.assertEqual([call.args[0] for call in imports.call_args_list],
                         ["torch", "torchvision", "transformers", "vllm", "httpx", "PIL"])
        self.assertNotIn("trl", [call.args[0] for call in version.call_args_list])
        self.assertNotIn("peft", [call.args[0] for call in version.call_args_list])

    def test_preflight_rejects_training_interpreter(self):
        with patch("src.inference.vllm_server.sys.prefix", str(Path(__file__).resolve().parents[1].parent / ".venv")):
            result = check_environment()
        self.assertTrue(any("serving environment" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
