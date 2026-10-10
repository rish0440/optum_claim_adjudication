"""Phase 5 CPU-only regression tests for the remote VLM client.

Uses httpx.MockTransport: no GPU, network, training images or test labels sent.
"""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from PIL import Image

from src.inference.remote_client import RemoteVLMClient, RemoteVLMConfig, RemoteVLMError


class RemoteClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "fake_hospital.jpg"
        Image.new("RGB", (64, 48), "white").save(self.image, "JPEG")
        self.config = RemoteVLMConfig(base_url="http://127.0.0.1:8010/v1", api_key="test-secret")

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, handler):
        return RemoteVLMClient(self.config, httpx.Client(transport=httpx.MockTransport(handler)))

    def test_models_probe_sends_auth(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
            return httpx.Response(200, json={"data": [{"id": "invoice-extractor"}]})
        self.assertEqual(self.client(handler).available_models(), ["invoice-extractor"])

    def test_image_csv_inference_sends_actual_image_and_measures_latency(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "invoice-extractor")
            contents = payload["messages"][0]["content"]
            self.assertIn("leave Rate empty", contents[0]["text"])
            value = contents[1]["image_url"]["url"]
            self.assertTrue(value.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(value.split(",", 1)[1]), self.image.read_bytes())
            return httpx.Response(200, json={"model": "invoice-extractor", "choices": [{
                "message": {"role": "assistant", "content":
                    "Item,Quantity,Rate,Amount\nRoom Charges,2,3500,7000\nConsultation,1,1500,1500\n"}}],
                "usage": {"completion_tokens": 39}})
        result = self.client(handler).extract(self.image)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0]["Amount"], "7000")
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(result.usage["completion_tokens"], 39)

    def test_missing_rate_is_not_fabricated(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {
                "content": "Item,Quantity,Rate,Amount\nMRI Scan,1,,8500\n"}}]})
        result = self.client(handler).extract(self.image)
        self.assertEqual(result.rows[0]["Rate"], "")

    def test_reject_invalid_model_output(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {
                "content": "Sorry, I cannot read this invoice."}}]})
        with self.assertRaisesRegex(RemoteVLMError, "not valid invoice CSV"):
            self.client(handler).extract(self.image)

    def test_reject_incorrect_arithmetic(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"message": {
                "content": "Item,Quantity,Rate,Amount\nMedicine,2,100,300\n"}}]})
        with self.assertRaisesRegex(RemoteVLMError, "not valid invoice CSV"):
            self.client(handler).extract(self.image)

    def test_invalid_auth_reports_http_status_without_body(self):
        def handler(request):
            return httpx.Response(401, text="private response body")
        with self.assertRaisesRegex(RemoteVLMError, "HTTP 401") as ctx:
            self.client(handler).available_models()
        self.assertNotIn("private response body", str(ctx.exception))

    def test_external_http_and_missing_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            RemoteVLMConfig(base_url="http://gpu.example.com/v1", api_key="secret")
        with self.assertRaisesRegex(ValueError, "VLM_API_KEY"):
            RemoteVLMConfig(base_url="http://127.0.0.1:8010/v1", api_key="")
        with self.assertRaisesRegex(ValueError, "must end in /v1"):
            RemoteVLMConfig(base_url="https://gpu.example.com", api_key="secret")

    def test_size_limit_blocks_upload(self):
        config = RemoteVLMConfig(base_url="http://localhost:8010/v1", api_key="secret", max_image_bytes=3)
        client = RemoteVLMClient(config)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            client.extract(self.image)

    def test_bad_image_blocks_upload(self):
        self.image.write_bytes(b"not a valid image")
        client = RemoteVLMClient(self.config)
        with self.assertRaisesRegex(ValueError, "valid invoice image"):
            client.extract(self.image)


if __name__ == "__main__":
    unittest.main()
