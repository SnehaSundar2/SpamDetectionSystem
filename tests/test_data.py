"""Text processing, schema helpers, synthetic data, dataset loading, cleaning and features."""

import numpy as np
import pandas as pd

from spamdetect import datagen, models as M, preprocessing as P, schema as S, textproc as T
from spamdetect.exceptions import DataError
from tests.helpers import PRIZE, TempDirTest


class TestTextProcessing(TempDirTest):
    def test_normalise_text(self):
        out = T.normalise_text("WIN a fr33 prize! Call +91 98765 43210 or visit http://bit.ly/3xYz9 for Rs 5000 now")
        for part in ("win a free prize", "phonetoken", "urltoken", "moneytoken", "bignum"):
            self.assertIn(part, out)
        self.assertEqual(T.normalise_text("w1n $1000 at WWW.Promo-Win.xyz/claim"), "win moneytoken bignum at urltoken")

    def test_times_and_sizes_are_not_mistaken_for_leetspeak(self):
        self.assertEqual(T.normalise_text("meet at 5pm, bring 10k"), "meet at num pm, bring num k")
        self.assertNotIn("spm", T.normalise_text("see you at 5pm"))

    def test_meta_features_are_bounded_and_meaningful(self):
        frame = pd.DataFrame({"text": [PRIZE, "ok", "", "A" * 5000, "call +91 98765 43210 now"]})
        out = T.meta_features(frame)
        self.assertEqual(out.shape, (5, len(T.META_NAMES)))
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        link, phone = T.META_NAMES.index("link"), T.META_NAMES.index("phone")
        self.assertEqual(out[0, link], 1.0)
        self.assertEqual(out[1, link], 0.0)
        self.assertEqual(out[4, phone], 1.0)

    def test_units_treat_links_and_phone_numbers_as_one_piece(self):
        pieces = T.units("Call +91 98765 43210 or open http://claim-reward.xyz/win now. Open it!")
        kinds = {kind for kind, _ in pieces}
        self.assertEqual(kinds, {"link", "phone", "word"})
        self.assertIn(("link", "http://claim-reward.xyz/win"), pieces)
        self.assertIn(("phone", "+91 98765 43210"), pieces)
        words = [p.lower() for k, p in pieces if k == "word"]
        self.assertEqual(len(words), len(set(words)))                   # every word once
        self.assertLessEqual(len(T.units("word " * 200 + " ".join(f"w{i}x" for i in range(200)))), 40)

    def test_remove_unit(self):
        self.assertEqual(T.remove_unit("Win now, win big", "word", "win"), "now, big")
        self.assertEqual(T.remove_unit("go to http://x.xyz/a today", "link", "http://x.xyz/a"), "go to today")
        self.assertEqual(T.remove_unit("a cat sat", "word", "at"), "a cat sat")      # whole words only

    def test_signals(self):
        found = T.signals(PRIZE)
        for expected in ("Contains a link", "Mentions money", "Many exclamation marks", "Pushes you to act quickly"):
            self.assertIn(expected, found)
        self.assertEqual(T.signals("see you at lunch"), [])
        self.assertTrue(any("sensitive" in s for s in T.signals("Your OTP is 123456")))


class TestSchemaHelpers(TempDirTest):
    def test_labels(self):
        spam = ["spam", "SPAM ", "Spam", 1, "1", 1.0, "junk", True, "phishing"]
        ham = ["ham", " HAM", 0, "0", 0.0, "legit", "not spam", "Not_Spam", False]
        self.assertEqual({S.canonical_label(x) for x in spam}, {S.SPAM})
        self.assertEqual({S.canonical_label(x) for x in ham}, {S.HAM})
        for bad in ("maybe", "", None, float("nan"), 2, "spammy"):
            self.assertIsNone(S.canonical_label(bad), repr(bad))

    def test_verdict_boundaries(self):
        self.assertEqual(S.verdict_for(0.9, 0.2, 0.6), "Spam")
        self.assertEqual(S.verdict_for(0.6, 0.2, 0.6), "Spam")          # the threshold itself counts as spam
        self.assertEqual(S.verdict_for(0.3, 0.2, 0.6), "Suspicious")
        self.assertEqual(S.verdict_for(0.19, 0.2, 0.6), "Not spam")
        self.assertEqual(S.verdict_for(0.5, 0.6, 0.6), "Not spam")      # no suspicious band when the thresholds meet


class TestDatagen(TempDirTest):
    def test_reproducible_and_dirty(self):
        a, b = datagen.generate(400, seed=1), datagen.generate(400, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(400, seed=2)))
        self.assertGreater(a.isna().sum().sum(), 0)
        self.assertGreater(a.duplicated().sum(), 0)

    def test_clean_data_is_tidy(self):
        clean = datagen.generate(1500, dirty=False)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertEqual(set(clean["label"]), {"spam", "ham"})
        self.assertTrue((clean["text"].str.strip() != "").all())
        share = (clean["label"] == "spam").mean()
        self.assertAlmostEqual(share, datagen.SPAM_SHARE, delta=0.04)

    def test_data_contains_the_hard_cases(self):
        clean = datagen.generate(3000, dirty=False)
        spam, ham = clean[clean["label"] == "spam"]["text"], clean[clean["label"] == "ham"]["text"]
        self.assertTrue(ham.str.contains("OTP", regex=False).any())                  # genuine messages that look alarming
        self.assertTrue(ham.str.contains("http", regex=False).any())                 # genuine messages with links
        self.assertTrue(spam.str.contains("kindly", case=False).any())               # spam that sounds calm
        self.assertTrue(spam.str.contains(r"[01@$]", regex=True).any())              # obfuscation
        self.assertTrue(spam.str.contains(r"[^\x00-\x7f]", regex=True).any())        # emoji

    def test_sample_data_is_cleanable(self):
        cleaned, report = P.clean(datagen.generate(1800))
        self.assertGreater(report.rows_dropped, 50)
        self.assertLess(report.rows_dropped, 400)
        self.assertEqual(cleaned["text"].str.lower().duplicated().sum(), 0)


class TestLoading(TempDirTest):
    def write(self, name, text, binary=False):
        path = self.dir / name
        path.write_bytes(text) if binary else path.write_text(text, encoding="utf-8")
        return path

    def test_common_column_names_are_understood(self):
        df = datagen.generate(150, dirty=False).rename(columns={"label": "v1", "text": "v2"})
        path = self.dir / "kaggle_style.csv"
        df.to_csv(path, index=False)
        loaded = P.load_dataset(path)
        self.assertEqual(list(loaded.columns), ["label", "text"])
        self.assertEqual(len(loaded), 150)
        df = datagen.generate(150, dirty=False).rename(columns={"label": " Category ", "text": "Message"})
        path = self.dir / "other.csv"
        df.to_csv(path, index=False)
        self.assertEqual(len(P.load_dataset(path)), 150)

    def test_non_utf8_file_is_read_and_noted(self):
        df = datagen.generate(400, dirty=False)
        df.loc[0, "text"] = "Café menu: free coffee"
        path = self.dir / "latin.csv"
        path.write_bytes(df.to_csv(index=False).encode("cp1252", errors="replace"))   # emoji become "?"
        loaded = P.load_dataset(path)
        self.assertIn("Windows-1252", loaded.attrs["note"])
        _, report = P.clean(loaded)
        self.assertTrue(any("Windows-1252" in n for n in report.notes))
        self.assertIn("Café", loaded.loc[0, "text"])

    def test_problem_files_are_explained(self):
        cases = {
            "missing.csv": (None, "not found"),
            "empty.csv": ("", "is empty"),
            "cols.csv": ("name,age\nAsha,20\n", "needs a label column and a text column"),
            "nolabel.csv": ("message,length\nhello,5\n", "needs a label column"),
            "small.csv": (datagen.generate(30, dirty=False).to_csv(index=False), "only 30 rows"),
        }
        for name, (text, hint) in cases.items():
            path = self.dir / name if text is None else self.write(name, text)
            with self.assertRaises(DataError, msg=name) as ctx:
                P.load_dataset(path)
            self.assertIn(hint, str(ctx.exception))


class TestCleaning(TempDirTest):
    def setUp(self):
        super().setUp()
        self.base = P.clean(datagen.generate(400, dirty=False, seed=5))[0]       # already tidy and unique

    def with_rows(self, *rows):
        return pd.concat([self.base.assign(label=self.base["label"].map({1: "spam", 0: "ham"})),
                          pd.DataFrame(rows, columns=["label", "text"])], ignore_index=True)

    def test_clean_data_passes_through_untouched(self):
        shown = self.base.assign(label=self.base["label"].map({1: "spam", 0: "ham"}))
        out, report = P.clean(shown)
        self.assertEqual((report.rows_in, report.rows_out), (len(self.base), len(self.base)))
        self.assertEqual([s for s in report.steps if "standardised" not in s[0] and "style" not in s[0]], [])

    def test_each_problem_is_handled(self):
        n = len(self.base)
        same_label = lambda i: "spam" if self.base["label"].iloc[i] == S.SPAM else "ham"
        removed = {
            "missing text": self.with_rows(("spam", np.nan)),
            "number as text": self.with_rows(("spam", 12345)),
            "blank text": self.with_rows(("ham", "   \n\t")),
            "too short": self.with_rows(("ham", "?")),
            "missing label": self.with_rows((np.nan, "a brand new message here")),
            "unknown label": self.with_rows(("maybe", "another brand new message")),
            "exact duplicate": self.with_rows((same_label(0), self.base["text"].iloc[0])),
            "duplicate with other capitals": self.with_rows((same_label(1), self.base["text"].iloc[1].upper())),
        }
        for label, df in removed.items():
            out, report = P.clean(df)
            self.assertEqual(report.rows_dropped, 1, label)
            self.assertEqual(len(out), n, label)

    def test_contradictory_labels_remove_every_copy(self):
        text = self.base["text"].iloc[0]
        opposite = "ham" if self.base["label"].iloc[0] == S.SPAM else "spam"
        out, report = P.clean(self.with_rows((opposite, text)))
        self.assertEqual(len(out), len(self.base) - 1)                  # the original AND the contradicting copy are gone
        self.assertNotIn(text.lower(), set(out["text"].str.lower()))
        self.assertTrue(any("both spam and not spam" in step[0] for step in report.steps))

    def test_text_is_tidied_not_dropped(self):
        out, report = P.clean(self.with_rows(("spam", "  free \t\n prize   now\x07 ")))
        self.assertEqual(out["text"].iloc[-1], "free prize now")
        self.assertTrue(any("stray whitespace" in step[0] for step in report.steps))
        long, report = P.clean(self.with_rows(("ham", "word " * 2000)))
        self.assertEqual(long["text"].str.len().max(), S.MAX_TEXT)
        self.assertTrue(any("truncated" in step[2] for step in report.steps))

    def test_label_spellings_are_standardised(self):
        rows = [("SPAM ", "unique alpha text one"), ("1", "unique beta text two"), ("Junk", "unique gamma text three"),
                ("HAM", "unique delta text four"), ("0", "unique epsilon text five"), ("legit", "unique zeta text six")]
        out, report = P.clean(self.with_rows(*rows))
        tail = out.tail(6)
        self.assertEqual(tail["label"].tolist(), [1, 1, 1, 0, 0, 0])
        self.assertEqual(out["label"].dtype, np.dtype("int64"))

    def test_input_is_not_modified_and_accounting_adds_up(self):
        raw = datagen.generate(800)
        before = raw.copy()
        out, report = P.clean(raw)
        pd.testing.assert_frame_equal(raw, before)
        removed = sum(count for _, count, outcome in report.steps if outcome.startswith("removed"))
        self.assertEqual(report.rows_dropped, removed)
        self.assertEqual(report.rows_out, len(out))
        self.assertEqual(report.class_counts, {"spam": int((out["label"] == 1).sum()), "ham": int((out["label"] == 0).sum())})
        self.assertEqual(set(out["label"]), {0, 1})

    def test_too_little_usable_data_is_refused(self):
        swapped = datagen.generate(300, dirty=False).rename(columns={"label": "text", "text": "label"})
        with self.assertRaises(DataError) as ctx:
            P.clean(swapped)
        self.assertIn("usable rows remain", str(ctx.exception))
        only_ham = pd.DataFrame({"label": ["ham"] * 150, "text": [f"message number {i} about nothing" for i in range(150)]})
        with self.assertRaises(DataError) as ctx:
            P.clean(only_ham)
        self.assertIn("spam messages remain", str(ctx.exception))


class TestFeatures(TempDirTest):
    def test_features_use_only_the_text(self):
        X, y = P.split_xy(self.clean_frame())
        self.assertEqual(list(X.columns), ["text"])
        self.assertEqual(set(y), {0, 1})

    def clean_frame(self):
        return P.clean(datagen.generate(300, dirty=False))[0]

    def test_pipeline_copes_with_odd_input(self):
        X, y = P.split_xy(self.clean_frame())
        pipe = M.build_pipeline(M.candidates(fast=True)["Naive Bayes"]).fit(X, y)
        odd = pd.DataFrame({"text": ["", "🎉🎉🎉", "zzzzqqqq unseenword", "ok", "你好，免费领取奖品", "A" * 5000]})
        probabilities = pipe.predict_proba(odd)[:, 1]
        self.assertEqual(probabilities.shape, (6,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
