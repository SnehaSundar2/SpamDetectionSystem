"""Input validation, model persistence, checking messages, explanations and batch files."""

import json

import pandas as pd

from spamdetect import persistence, schema as S, validators as v
from spamdetect.exceptions import DataError, ModelError, NotTrainedError, ValidationError
from spamdetect.predictor import Predictor, highlight_pieces, require_predictor
from tests.helpers import BANK_OTP, FRIEND, PRIZE, TempDirTest, shared


class TestValidation(TempDirTest):
    def test_good_messages_are_tidied(self):
        text, warnings = v.validate_message("  win\t a   prize\n\nnow\x07 ")
        self.assertEqual(text, "win a prize now")
        self.assertEqual(warnings, [])

    def test_unusable_messages_are_refused_with_a_reason(self):
        for bad in (None, "", "   \n\t", float("nan")):
            with self.assertRaises(ValidationError, msg=repr(bad)) as ctx:
                v.validate_message(bad)
            self.assertIn("Enter a message", str(ctx.exception))
            self.assertIn("text", ctx.exception.errors)
        with self.assertRaises(ValidationError) as ctx:
            v.validate_message(12345)
        self.assertIn("must be text", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            v.validate_message("a " * 3000)
        self.assertIn("too long", str(ctx.exception))
        v.validate_message("x" * S.MAX_TEXT)                                 # exactly the limit is fine

    def test_warnings_for_messages_the_model_cannot_judge_well(self):
        _, warnings = v.validate_message("ok")
        self.assertTrue(any("short" in w for w in warnings))
        _, warnings = v.validate_message("你好，恭喜您获得免费奖品，请点击链接领取")
        self.assertTrue(any("English" in w for w in warnings))
        _, warnings = v.validate_message("see you at the canteen at lunch")
        self.assertEqual(warnings, [])

    def test_settings_validation(self):
        self.assertEqual(v.validate_port("8020"), 8020)
        self.assertEqual(v.validate_int("5", "Rows", 1, 10), 5)
        for bad in ("abc", "80", "70000", ""):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        for bad in ("x", "0", "11", ""):
            with self.assertRaises(ValidationError):
                v.validate_int(bad, "Rows", 1, 10)


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, _ = shared()
        a = Predictor(persistence.load(model_dir)).check(PRIZE)
        b = Predictor(persistence.load(model_dir)).check(PRIZE)
        self.assertEqual(a.probability, b.probability)
        original = float(result.pipeline.predict_proba(pd.DataFrame({"text": [PRIZE]}))[0, 1])
        self.assertAlmostEqual(a.probability, original, places=9)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(meta["files"]), {"model.joblib", "reference.csv"})
        self.assertEqual(len(meta["files"]["model.joblib"]), 64)
        self.assertEqual([p.name for p in model_dir.glob("*.tmp")], [])      # atomic saves leave no temp files

    def test_no_model_gives_instructions(self):
        for call in (lambda: persistence.load(self.dir / "empty"), lambda: require_predictor(self.dir / "empty")):
            with self.assertRaises(NotTrainedError) as ctx:
                call()
            self.assertIn("Train one first", str(ctx.exception))

    def test_tampering_and_damage_are_detected(self):
        model = self.copy_model()
        data = bytearray((model / "model.joblib").read_bytes())
        data[len(data) // 2] ^= 0x01
        (model / "model.joblib").write_bytes(bytes(data))
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("checksum", str(ctx.exception))

        model = self.copy_model()
        (model / "reference.csv").unlink()
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("missing", str(ctx.exception))

        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["files"]["model.joblib"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ModelError):
            persistence.load(model)

        meta_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("damaged", str(ctx.exception))

        meta_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("no checksums", str(ctx.exception))

    def test_library_version_mismatch_warns(self):
        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.20.0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        loaded = persistence.load(model)
        self.assertTrue(any("0.20.0" in w for w in loaded.warnings))


class TestChecking(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def test_structure_and_json(self):
        p = self.predictor.check(PRIZE)
        self.assertTrue(0 <= p.probability <= 1)
        self.assertEqual(p.verdict, self.predictor.verdict(p.probability))
        self.assertEqual((p.spam_threshold, p.review_threshold),
                         (self.predictor.spam_threshold, self.predictor.review_threshold))
        data = p.to_dict()
        for key in ("verdict", "spam_probability", "spam_threshold", "review_threshold", "warnings", "signals",
                    "words_toward_spam", "words_toward_not_spam", "similar_messages"):
            self.assertIn(key, data)
        json.dumps(data)                                                      # everything can be sent as JSON

    def test_the_example_messages_get_the_right_verdicts(self):
        expected = {"prize": "Spam", "bank": "Not spam", "friend": "Not spam", "fee": "Spam"}
        for key, (_, text) in S.EXAMPLES.items():
            self.assertEqual(self.predictor.check(text).verdict, expected[key], key)

    def test_genuine_alarming_messages_are_not_blocked(self):
        for text in (BANK_OTP, "URGENT: the server is down again, call me now when you see this",
                     "Your refund of Rs 1,250 has been processed to your SBI account. It may take 3 days to show.",
                     "Reminder: your appointment with Dr. Priya is confirmed for 5pm. Reply C to confirm."):
            self.assertNotEqual(self.predictor.check(text).verdict, "Spam", text)

    def test_disguised_spam_and_shouting_friends(self):
        self.assertEqual(self.predictor.check(PRIZE.upper()).verdict, "Spam")
        leet = "C0NGRATULATI0NS! y0u have w0n a pr1ze. Cl@im n0w at http://claim-reward.xyz/w1n"
        self.assertEqual(self.predictor.check(leet).verdict, "Spam")
        self.assertNotEqual(self.predictor.check(FRIEND.upper()).verdict, "Spam")

    def test_spam_scores_higher_than_a_friendly_message(self):
        self.assertGreater(self.predictor.check(PRIZE).probability, 0.9)
        self.assertLess(self.predictor.check(FRIEND).probability, 0.1)

    def test_the_same_message_always_gets_the_same_score(self):
        self.assertEqual(self.predictor.check(PRIZE).probability, self.predictor.check("  " + PRIZE + "\n").probability)

    def test_explanation_points_at_the_link_and_is_sorted(self):
        p = self.predictor.check(PRIZE)
        self.assertTrue(p.toward_spam)
        self.assertIn("[link]", p.toward_spam[0]["word"])                    # removing the link matters most
        effects = [r["effect"] for r in p.toward_spam]
        self.assertEqual(effects, sorted(effects, reverse=True))
        self.assertTrue(all(r["effect"] >= 0.25 and r["odds"] > 1 for r in p.toward_spam))
        self.assertTrue(all(r["effect"] <= -0.25 and r["odds"] < 1 for r in p.toward_genuine))
        self.assertLessEqual(len(p.toward_spam), 6)
        without_link = self.predictor.check(PRIZE.replace("http://claim-reward.xyz/win", ""), explain=False)
        self.assertLess(without_link.probability, p.probability)             # taking the link out really lowers the score

    def test_explanation_for_a_genuine_message_points_the_other_way(self):
        p = self.predictor.check(FRIEND)
        self.assertEqual(p.toward_spam, [])
        self.assertTrue(p.toward_genuine)

    def test_highlighting_keeps_the_text_intact(self):
        p = self.predictor.check(PRIZE)
        chunks = highlight_pieces(p.text, p.effects)
        self.assertEqual("".join(chunk for chunk, _ in chunks), p.text)
        marked = [chunk for chunk, effect in chunks if effect is not None]
        self.assertIn("http://claim-reward.xyz/win", marked)                 # a link is marked as one piece
        self.assertTrue(all(effect is None or effect >= 0.25 for _, effect in chunks))

    def test_signals_and_warnings(self):
        p = self.predictor.check(PRIZE)
        self.assertIn("Contains a link", p.signals)
        self.assertEqual(p.warnings, [])
        self.assertTrue(any("short" in w for w in self.predictor.check("ok ok").warnings))

    def test_similar_known_messages(self):
        p = self.predictor.check(PRIZE)
        self.assertEqual(len(p.similar), 3)
        self.assertTrue(all(s["label"] == "Spam" for s in p.similar))
        sims = [s["similarity"] for s in p.similar]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertTrue(all(0 < s <= 1 for s in sims))
        nothing = self.predictor.check("qqqqzzzz xxxxvvvv", explain=False)
        self.assertEqual(nothing.similar, [])                                # no resemblance: nothing is invented

    def test_invalid_input_never_reaches_the_model(self):
        for bad in ("", None, "   ", "x " * 3000, 42):
            with self.assertRaises(ValidationError):
                self.predictor.check(bad)

    def test_odd_but_valid_input_does_not_crash(self):
        for text in ("🎉🎉🎉", "A" * 5000, "http://a.xyz", "+91 98765 43210", "%s %d {0} <b>", "'; DROP TABLE messages;--"):
            p = self.predictor.check(text)
            self.assertTrue(0 <= p.probability <= 1, text)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def setUp(self):
        super().setUp()
        self.source = self.dir / "messages.csv"
        pd.DataFrame({"ID": ["a1", "a2", "a3", "a4", "a5", "a6"],
                      "Message": [PRIZE, FRIEND, "", "ok", "word " * 1500, BANK_OTP]}).to_csv(self.source, index=False)

    def test_good_rows_are_checked_and_bad_rows_reported(self):
        out = self.dir / "result.csv"
        ok, errors = self.predictor.predict_csv(self.source, out)
        self.assertEqual(ok, 4)
        self.assertEqual([e.split(":")[0] for e in errors], ["Line 4", "Line 6"])    # the blank row and the over-long row
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.assertEqual(len(frame), 6)
        self.assertEqual(list(frame.columns[:2]), ["id", "message"])                 # original columns kept
        self.assertEqual(frame["id"].tolist(), ["a1", "a2", "a3", "a4", "a5", "a6"])
        self.assertEqual(frame["verdict"][0], "Spam")
        self.assertEqual(frame["verdict"][1], "Not spam")
        self.assertEqual(frame["verdict"][2], "")                                    # no verdict for an unusable row
        self.assertTrue(frame["status"][2].startswith("error:"))
        self.assertIn("short", frame["status"][3])                                   # judged, but with a warning
        self.assertAlmostEqual(float(frame["spam_probability"][0]), self.predictor.check(PRIZE, explain=False).probability, places=3)

    def test_other_column_names_work(self):
        path = self.dir / "sms.csv"
        pd.DataFrame({"SMS": [PRIZE, FRIEND]}).to_csv(path, index=False)
        ok, errors = self.predictor.predict_csv(path, self.dir / "o.csv")
        self.assertEqual((ok, errors), (2, []))

    def test_file_level_problems(self):
        out = self.dir / "result.csv"
        with self.assertRaises(DataError):
            self.predictor.predict_csv(self.dir / "missing.csv", out)
        no_cols = self.dir / "bad.csv"
        no_cols.write_text("name,age\nAsha,20\n", encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(no_cols, out)
        self.assertIn("needs a text column", str(ctx.exception))
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(DataError):
            self.predictor.predict_csv(empty, out)
        self.predictor.predict_csv(self.source, out)
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(self.source, out)                             # never overwrite
        self.assertIn("already exists", str(ctx.exception))
