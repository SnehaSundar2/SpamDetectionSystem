"""The web interface (over real HTTP on a random local port) and the interactive menu."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from unittest.mock import patch

from spamdetect import datagen, web
from spamdetect.cli import SpamDetectCLI
from spamdetect.predictor import Predictor
from tests.helpers import FRIEND, PRIZE, TempDirTest, shared


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.server = web.make_server(Predictor.from_directory(shared()[1]), port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, text):
        return self.request("POST", "/", urllib.parse.urlencode({"text": text}),
                            {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload):
        return self.request("POST", "/api/predict", json.dumps(payload), {"Content-Type": "application/json"})

    def test_pages_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Check message", body)
        self.assertNotIn("<script", body.lower())                          # the page needs no scripts at all
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_examples_prefill_the_form(self):
        status, body, _ = self.request("GET", "/?example=prize")
        self.assertEqual(status, 200)
        self.assertIn("CONGRATULATIONS", body)
        self.assertIn("/?example=fee", body)                               # all four example links are offered
        status, body, _ = self.request("GET", "/?example=<script>")
        self.assertEqual(status, 200)
        self.assertNotIn("<script>", body)

    def test_spam_result_page(self):
        status, body, _ = self.post_form(PRIZE)
        self.assertEqual(status, 200)
        for part in ("chip spam", "chance of spam", "What stands out", "What moves the score", "Most similar known messages",
                     "<mark title"):
            self.assertIn(part, body)
        self.assertIn(">Spam<", body)

    def test_genuine_result_page(self):
        status, body, _ = self.post_form(FRIEND)
        self.assertEqual(status, 200)
        self.assertIn("chip ok", body)
        self.assertIn(">Not spam<", body)
        self.assertNotIn("<mark title", body)                              # nothing in a friendly message raises the score

    def test_empty_message_is_refused_and_named(self):
        status, body, _ = self.post_form("   ")
        self.assertEqual(status, 422)
        self.assertIn("Enter a message to check.", body)
        status, body, _ = self.post_form("word " * 2000)
        self.assertEqual(status, 422)
        self.assertIn("too long", body)
        self.assertIn("word word word", body)                              # what the user typed is still there

    def test_scripts_in_the_message_are_escaped(self):
        evil = '"><script>alert(1)</script> win a prize at http://x-win.xyz/<img src=x onerror=alert(2)>'
        status, body, _ = self.post_form(evil)
        self.assertEqual(status, 200)
        self.assertNotIn("<script>alert", body)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;script&gt;", body)

    def test_json_api(self):
        status, body, headers = self.post_json({"text": PRIZE})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(data["verdict"], "Spam")
        self.assertGreater(data["spam_probability"], data["spam_threshold"])
        status, body, _ = self.post_json({"text": ""})
        self.assertEqual(status, 422)
        self.assertIn("text", json.loads(body)["fields"])
        self.assertEqual(self.post_json({"message": PRIZE})[0], 422)       # the wrong key is not silently accepted
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertIn("spam_threshold", health)

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{not json", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", '["a list"]', {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 60_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 60_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)   # DNS-rebinding guard
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)

    def test_the_largest_allowed_message_still_works_through_the_form(self):
        status, body, _ = self.post_form("é" * 4000)                       # URL-encoded this is ~48 KB of text characters
        self.assertIn(status, (200, 413))                                  # accepted, or refused cleanly: never a crash


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


class TestCLI(TempDirTest):
    def make(self):
        return SpamDetectCLI(shared()[1], self.dir / "reports")

    def test_check_session(self):
        out = run_cli(self.make(), ["1", PRIZE, "0"])
        for part in ("Verdict:", "SPAM", "Chance of spam:", "What stands out", "What moves the score", "Most similar"):
            self.assertIn(part, out)
        out = run_cli(self.make(), ["1", FRIEND, "0"])
        self.assertIn("NOT SPAM", out)

    def test_empty_message_gives_a_friendly_error(self):
        out = run_cli(self.make(), ["1", "   ", "0"])
        self.assertIn("Enter a message to check", out)
        self.assertNotIn("Verdict:", out)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["5", "6", "7", "", "99", "abc", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertEqual(out.count("Invalid option"), 2)

    def test_batch_check_through_the_menu(self):
        source = self.dir / "in.csv"
        source.write_text("text\n" + '"' + PRIZE.replace('"', '""') + '"\n\n"hello there"\n', encoding="utf-8")
        target = self.dir / "out.csv"
        out = run_cli(self.make(), ["2", str(source), str(target), "0"])
        self.assertIn("Checked", out)
        self.assertTrue(target.exists())
        missing = run_cli(self.make(), ["2", str(self.dir / "nothing.csv"), "", "0"])
        self.assertIn("File not found", missing)

    def test_dataset_tools(self):
        path = self.dir / "mine.csv"
        out = run_cli(self.make(), ["8", "300", "", str(path), "9", str(path), "0"])
        self.assertIn("Saved 300 messages", out)
        self.assertIn("rows read", out)
        self.assertIn("Kept:", out)
        out = run_cli(self.make(), ["8", "300", "", str(path), "0"])
        self.assertIn("already exists", out)

    def test_no_model_gives_instructions_and_training_works(self):
        empty = SpamDetectCLI(self.dir / "nomodel", self.dir / "reports")
        out = run_cli(empty, ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)
        data = self.dir / "train.csv"
        datagen.generate(500).to_csv(data, index=False)
        out = run_cli(empty, ["n", "4", str(data), "y", "0"])             # quick mode training from a CSV
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue((self.dir / "nomodel" / "model.joblib").exists())
