"""Model training, comparison, tuning, thresholds and evaluation."""

import copy

import numpy as np
import pandas as pd

from spamdetect import datagen, evaluation as E, models as M, preprocessing as P, report, schema as S, training
from spamdetect.exceptions import DataError
from tests.helpers import TempDirTest, shared


class TestEvaluationMaths(TempDirTest):
    def test_metrics_on_a_tiny_example(self):
        m = E.classification_metrics([1, 1, 0, 0, 0], [0.9, 0.4, 0.6, 0.1, 0.2], 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 2))
        self.assertEqual((m["precision"], m["recall"], m["f1"], m["accuracy"]), (50.0, 50.0, 50.0, 60.0))
        self.assertAlmostEqual(m["false_alarm_rate"], 100 / 3)
        self.assertEqual(E.classification_metrics([1, 1], [0.1, 0.2], 0.5)["precision"], 0.0)    # nothing flagged: defined
        self.assertEqual(E.classification_metrics([0, 0], [0.1, 0.2], 0.5)["recall"], 0.0)       # no spam at all: defined

    def test_accuracy_alone_is_misleading_and_the_metrics_show_it(self):
        actual = [0] * 90 + [1] * 10
        m = E.classification_metrics(actual, [0.0] * 100, 0.5)               # "everything is fine"
        self.assertEqual(m["accuracy"], 90.0)
        self.assertEqual((m["recall"], m["f1"]), (0.0, 0.0))

    def test_ranking_metrics(self):
        self.assertEqual(E.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), {"roc_auc": 1.0, "average_precision": 1.0})
        worst = E.ranking_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        self.assertEqual(worst["roc_auc"], 0.0)
        self.assertEqual(E.ranking_metrics([1, 1, 1], [0.2, 0.5, 0.9]), {"roc_auc": 0.0, "average_precision": 0.0})

    def test_thresholds_on_separable_and_noisy_data(self):
        actual = [0] * 50 + [1] * 50
        probs = np.concatenate([np.linspace(0.05, 0.40, 50), np.linspace(0.60, 0.95, 50)])
        t = E.choose_thresholds(actual, probs)
        self.assertEqual(t["spam"], 0.60)                                    # the lowest spam score: everything caught
        self.assertTrue(t["precision_target_met"])
        self.assertLessEqual(t["review"], t["spam"])
        noisy = probs.copy()
        noisy[10] = 0.97                                                     # a genuine message scored as very spammy
        noisy[70] = 0.20                                                     # a spam message scored as very genuine
        t = E.choose_thresholds(actual, noisy)
        m = E.classification_metrics(actual, noisy, t["spam"])
        self.assertGreaterEqual(m["precision"], 98.0)                        # the precision target holds on these data
        self.assertGreater(m["recall"], 90.0)
        self.assertLessEqual(t["review"], t["spam"])

    def test_unreachable_precision_target_falls_back_to_best_f1(self):
        actual = np.array([1, 0] * 200)
        probs = np.linspace(0, 1, 400)                                       # scores that say nothing about the label
        t = E.choose_thresholds(actual, probs, precision_target=0.999)
        self.assertFalse(t["precision_target_met"])
        self.assertTrue(0 <= t["spam"] <= 1)
        self.assertEqual(E.choose_thresholds([0, 1], [0.5, 0.5])["review"], 0.5)   # a constant score: still defined

    def test_review_band_catches_borderline_spam(self):
        actual = [0] * 60 + [1] * 40
        probs = [0.02] * 55 + [0.30] * 3 + [0.45] * 2 + [0.30] * 3 + [0.50] * 4 + [0.95] * 33
        t = E.choose_thresholds(actual, probs, precision_target=0.98)
        self.assertEqual(t["spam"], 0.50)                                    # 0.30 would let 5 genuine messages through
        self.assertEqual(t["review"], 0.30)                                  # 3 of the 8 messages in between are spam
        in_band = [(a, p) for a, p in zip(actual, probs) if t["review"] <= p < t["spam"]]
        self.assertGreaterEqual(sum(a for a, _ in in_band) / len(in_band), S.REVIEW_PRECISION)

    def test_curves_histograms_and_segments(self):
        points = E.pr_curve_points([0, 1] * 300, np.random.default_rng(1).random(600), max_points=40)
        self.assertLessEqual(len(points), 40)
        self.assertTrue(all(0 <= r <= 1 and 0 <= p <= 1 for r, p in points))
        hist = E.probability_histogram([0.0, 0.05, 0.5, 0.99, 1.0])
        self.assertEqual((len(hist), sum(hist), hist[0], hist[9]), (10, 5, 2, 2))
        frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "actual": [1, 0, 1, 1], "predicted": [1, 0, 0, 0]})
        rows = E.segment_errors(frame, "g")
        self.assertEqual([r["group"] for r in rows], ["b", "a"])             # worst first
        self.assertEqual((rows[0]["errors"], rows[0]["error_rate"], rows[1]["errors"]), (2, 100.0, 0))

    def test_top_words_and_error_examples(self):
        texts = ["win a free prize now", "free prize waiting", "claim your prize today", "win cash prize",
                 "lunch at noon", "see you at lunch", "lunch tomorrow ok", "ok see you"] * 4
        labels = [1] * 4 + [0] * 4
        words = E.top_words(texts, labels * 4, k=2, min_df=3)
        self.assertEqual(words["spam"][0]["word"], "prize")
        self.assertEqual(words["ham"][0]["word"], "lunch")
        frame = pd.DataFrame({"text": ["a" * 300, "short", "other"], "actual": [0, 0, 1], "probability": [0.9, 0.7, 0.1],
                              "predicted": [1, 1, 0]})
        errors = E.error_examples(frame, k=1)
        self.assertEqual(len(errors["false_positives"]), 1)
        self.assertEqual(errors["false_positives"][0]["probability"], 0.9)       # the most confident mistake first
        self.assertLessEqual(len(errors["false_positives"][0]["text"]), 170)
        self.assertEqual(errors["false_negatives"][0]["text"], "other")
        self.assertEqual(E.noise_ceiling(0.01), 99.0)


class TestTrainedModel(TempDirTest):
    def test_comparison_table(self):
        result, _, _ = shared()
        names = [c["model"] for c in result.comparison]
        self.assertEqual(set(names), {M.BASELINE, "Naive Bayes", "Logistic Regression", "Linear SVM", "Random Forest"})
        scores = [c["average_precision"] for c in result.comparison]
        self.assertEqual(scores, sorted(scores, reverse=True))               # best first
        self.assertEqual(result.comparison[-1]["model"], M.BASELINE)         # the baseline is the worst
        self.assertLess(result.comparison[-1]["average_precision"], 0.3)
        self.assertGreater(result.comparison[0]["average_precision"], 0.9)
        self.assertEqual(result.comparison[-1]["recall"], 0.0)               # "always not spam" catches nothing
        self.assertNotEqual(result.metadata["model"], M.BASELINE)

    def test_split_sizes_and_data_flow(self):
        result, _, raw = shared()
        rows = result.metadata["rows"]
        self.assertEqual(rows["raw"], len(raw))
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.02)
        self.assertEqual(len(result.test_frame), rows["test"])
        self.assertEqual(len(result.reference), rows["train"])
        train_texts, test_texts = set(result.reference["text"].str.lower()), set(result.test_frame["text"].str.lower())
        self.assertEqual(train_texts & test_texts, set())                    # no message appears on both sides

    def test_spam_share_is_kept_equal_in_train_and_test(self):
        result, _, _ = shared()
        self.assertAlmostEqual(result.reference["label"].mean(), result.test_frame["actual"].mean(), delta=0.02)

    def test_accuracy_on_unseen_messages(self):
        result, _, _ = shared()
        test, train = result.metadata["metrics"]["test"], result.metadata["metrics"]["train"]
        self.assertGreater(test["precision"], 95)
        self.assertGreater(test["recall"], 80)
        self.assertLess(test["false_alarm_rate"], 2)
        self.assertGreater(test["roc_auc"], 0.93)
        self.assertGreater(test["accuracy"], test["majority_accuracy"] + 5)  # clearly better than "always not spam"
        self.assertLess(train["f1"] - test["f1"], 15)                        # no serious overfitting
        self.assertEqual(test["tp"] + test["fp"] + test["fn"] + test["tn"], test["n"])

    def test_the_chosen_threshold_protects_genuine_messages(self):
        result, _, _ = shared()
        th = result.metadata["thresholds"]
        self.assertTrue(th["precision_target_met"])
        self.assertGreater(th["spam"], 0)
        self.assertLessEqual(th["review"], th["spam"])
        self.assertLessEqual(result.metadata["metrics"]["test"]["false_alarm_rate"], 1.0)
        frame = result.test_frame
        self.assertTrue(((frame["probability"] >= th["spam"]).astype(int) == frame["predicted"]).all())

    def test_mistakes_are_listed(self):
        result, _, _ = shared()
        errors, test = result.metadata["errors"], result.metadata["metrics"]["test"]
        self.assertEqual(set(errors), {"false_positives", "false_negatives"})
        self.assertEqual(len(errors["false_positives"]), min(8, test["fp"]))
        self.assertEqual(len(errors["false_negatives"]), min(8, test["fn"]))

    def test_words_and_curves_in_metadata(self):
        result, _, _ = shared()
        meta = result.metadata
        spam_words = {w["word"] for w in meta["top_words"]["spam"]}
        ham_words = {w["word"] for w in meta["top_words"]["ham"]}
        self.assertTrue(spam_words & {"[link]", "[phone number]", "claim", "apply", "kindly", "here", "now", "dear", "earn"})
        self.assertTrue(ham_words & {"thanks", "ok", "lol", "see", "me", "you"})
        self.assertEqual(sum(meta["histograms"]["spam"]) + sum(meta["histograms"]["not_spam"]), meta["rows"]["test"])
        self.assertGreater(len(meta["pr_curve"]), 5)
        self.assertEqual(set(meta["segments"]), {"length", "link"})

    def test_metadata_is_complete(self):
        result, _, _ = shared()
        for key in ("trained_at", "model", "metrics", "thresholds", "environment", "comparison", "cleaning", "top_words",
                    "errors", "segments", "pr_curve", "histograms", "seed", "class_counts"):
            self.assertIn(key, result.metadata, key)


class TestReproducibilityAndGuards(TempDirTest):
    def test_same_seed_same_result(self):
        raw = datagen.generate(500, seed=11)
        a = training.train(raw, fast=True, tune=False, cv=3)
        b = training.train(raw, fast=True, tune=False, cv=3)
        self.assertEqual(a.metadata["thresholds"], b.metadata["thresholds"])
        np.testing.assert_allclose(a.test_frame["probability"], b.test_frame["probability"])
        self.assertEqual(list(a.test_frame.index), list(b.test_frame.index))

    def test_tiny_or_unusable_data_is_refused(self):
        with self.assertRaises(DataError):
            training.train(datagen.generate(60, dirty=False), fast=True, tune=False)
        one_class = datagen.generate(300, dirty=False)
        one_class["label"] = "ham"
        with self.assertRaises(DataError):
            training.train(one_class, fast=True, tune=False)

    def test_tuning_search(self):
        X, y = P.split_xy(P.clean(datagen.generate(400, dirty=False))[0])
        search, params = M.tune("Naive Bayes", X, y, cv=3, iterations=3)
        self.assertIn(params["alpha"], M.TUNING_GRIDS["Naive Bayes"]["model__alpha"])
        self.assertEqual(M.tune(M.BASELINE, X, y), (None, {}))               # nothing to tune
        pipe = M.pipeline_with_params("Naive Bayes", params)
        self.assertEqual(pipe.fit(X, y).predict_proba(X.head(3)).shape, (3, 2))
        svm = M.pipeline_with_params("Linear SVM", {"estimator__C": 0.3})    # the nested parameter name works too
        self.assertEqual(svm.fit(X, y).predict_proba(X.head(2)).shape, (2, 2))


class TestReports(TempDirTest):
    def test_text_and_html_reports(self):
        meta = shared()[0].metadata
        text = report.model_text(meta)
        for part in ("ACCURACY ON THE UNSEEN TEST SET", "CONFUSION MATRIX", "MOST TELLING WORDS", "ERRORS BY KIND", "Precision"):
            self.assertIn(part, text)
        self.assertIn("chosen", report.comparison_text(meta))
        page = report.render_html(meta)
        self.assertEqual(page.count('<details class="tv">'), 6)              # every chart has a table view
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertIn("Data cleaning log", page)
        self.assertIn("Confusion matrix", page)
        path = report.save_html(meta, self.dir / "out" / "report.html")
        self.assertTrue(path.exists())

    def test_messages_in_the_report_are_escaped(self):
        meta = copy.deepcopy(shared()[0].metadata)
        meta["errors"]["false_negatives"] = [{"text": "<script>alert(1)</script> & <b>bold</b>", "probability": 0.1}]
        page = report.render_html(meta)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;", page)

    def test_chart_helpers(self):
        for value in (7, 45, 130, 999):
            top = report._tick_ceiling(value, 5)
            self.assertGreaterEqual(top, value)
            self.assertEqual(top / 5, round(top / 5, 1))
        svg = report.column_chart([("0-10%", 0), ("10-20%", 1), ("20-30%", 3)], "t")
        self.assertIn(">5<", svg)                                            # axis never shows fractional counts for tiny data
        self.assertEqual(svg.count("<path"), 2)                              # zero-height bars are not drawn
        self.assertIn('d="M', report.pr_chart([(0.0, 1.0), (0.5, 0.9), (1.0, 0.5)], "t", (0.5, 0.9)))
        self.assertIn("Chosen threshold", report.pr_chart([(0.0, 1.0), (1.0, 0.5)], "t", (0.5, 0.9)))
