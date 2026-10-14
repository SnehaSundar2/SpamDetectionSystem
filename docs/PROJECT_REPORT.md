# Project Report: Spam Detection System

**Program:** Virtual Internship – Skyrovix IT Solutions Pvt. Ltd. (AI & Machine Learning project)
**Project:** Spam Detection System
**Developed and submitted by:** Sneha S (Intern ID: SKX-2026-6425)
**Due date:** 01 November 2026

---

## 1. Introduction

Most of us receive spam every day: fake prize messages, "your account will be blocked" warnings, delivery-fee
scams. Some are harmless; some steal money. Writing rules by hand ("block anything with *free*") fails quickly:
genuine messages also say "urgent", carry links and mention money, and spammers keep changing their words.

This project builds a spam detector that **learns** from labelled examples. It cleans the messages, turns text into
numbers, trains and compares several models, evaluates the winner honestly on messages it has never seen, and
offers prediction interfaces (a menu, a command line, a web form and a JSON API). For every message it gives the
chance of spam, a verdict (*Spam*, *Suspicious* or *Not spam*), the words and links that led to it, and similar
messages it has already seen.

**About the data.** Real spam collections are licensed and contain real people's phone numbers, so the project
generates realistic but *artificial* messages. They include the awkward cases that make the task hard: genuine
bank alerts and OTP codes that look alarming, genuine messages with links, spam that sounds calm and personal,
obfuscated words (`fr33`, random capitals, typos, emoji), and 1% of labels flipped at random (as in real,
hand-labelled data). The results therefore describe the workflow, not real-world accuracy. A real dataset can be
used with `--data` (columns such as `label,text` or `v1,v2`).

## 2. Objectives

1. **Data preprocessing:** load, validate and clean messages and labels, normalise text, and turn it into features, leaving a record of every change.
2. **Model training:** compare several algorithms fairly, tune the best one, handle the fact that spam is the minority, and avoid leaking test data into training.
3. **Evaluation:** measure what matters when mistakes cost different amounts (precision, recall, false alarms), choose the decision threshold on training data, and study the actual mistakes.
4. **Prediction interface:** make the model usable through easy interfaces, with strict input validation and clear explanations.
5. **Quality:** handle errors safely, document the workflow, and verify the behaviour with automated tests.

## 3. Tools and technologies

| Item | Used |
|---|---|
| Language | Python 3 (tested on 3.13) |
| ML libraries | scikit-learn (TF-IDF, pipelines, classifiers, cross-validation, tuning), pandas, numpy, joblib |
| Standard library | `http.server` (web interface), `re`, `hashlib` (checksums), `json`, `csv`, `unittest` |
| Output | Terminal reports, self-contained HTML report with SVG charts, JSON API |

## 4. System design

### 4.1 Workflow

1. **Load** the CSV (label and text columns, several common names understood; non-UTF-8 files are read and noted).
2. **Clean** the data and record every action.
3. **Split** 80% / 20% into training and test sets with the same spam share in both. The test set is not touched again until the end.
4. **Compare** a baseline and four classifiers by 5-fold stratified cross-validation on the training messages.
5. **Tune** the best model with a randomised search; keep the tuned version only if cross-validation says it is better.
6. **Choose the thresholds** from out-of-fold predictions on the training messages.
7. **Evaluate once** on the test set; list the mistakes, the most telling words and the error rate per kind of message.
8. **Save** the model with checksums, then **check messages** through any interface.

### 4.2 Text preprocessing and features

| Step | What it does |
|---|---|
| Normalisation | Lower-case; links → `urltoken`, phone numbers → `phonetoken`, money signs → `moneytoken`, numbers → `num` / `bignum`; leetspeak inside mixed letter-digit words undone (`fr33` → `free`) except times and sizes (`5pm`, `10k`) |
| Word TF-IDF | Single words and word pairs, rare words ignored, sub-linear term frequency |
| Character TF-IDF | 3–5 letter pieces inside words: robust to typos and obfuscation |
| Message shape | Nine numbers between 0 and 1: length, digit share, capital share, exclamation marks, link, phone number, money signs, ALL-CAPS words, word count |

### 4.3 Modules

| Module | Responsibility |
|---|---|
| `schema.py` | Labels, column names, limits, verdicts, example messages |
| `textproc.py` | Normalisation, message-shape features, pieces used for explanations |
| `datagen.py` | Synthetic messages with hard cases, obfuscation, label noise and injected problems |
| `preprocessing.py` | Loading, cleaning with a report, feature transformer |
| `models.py` | Candidate models, cross-validated comparison, tuning |
| `evaluation.py` | Metrics, thresholds, curves, top words, mistakes |
| `training.py` | The workflow end to end |
| `persistence.py` | Saving and loading with SHA-256 integrity checks |
| `validators.py` | Validation of the message to check |
| `predictor.py` | Probability, verdict, explanation, similar messages, batch CSV |
| `report.py` | Text and HTML reports |
| `web.py`, `cli.py`, `main.py` | Web form and API, menu, command line |
| `exceptions.py` | Custom error types |

## 5. Implementation highlights

**Cleaning with an audit trail.** The generator deliberately produces about 8% problem rows. The cleaner found and
handled all of them, and said what it did:

```
1849 rows read → 1650 kept (199 removed)
  • Rows with a missing or non-text message: 8 — removed
  • Messages with stray whitespace or control characters: 71 — tidied
  • Messages empty or too short to judge: 10 — removed (need at least 2 letters or digits)
  • Messages longer than 5000 characters: 3 — truncated
  • Labels written in another style: 207 — standardised to spam / ham (e.g. 'SPAM ', '1', 'junk')
  • Rows with a missing or unrecognised label: 10 — removed
  • Same message labelled both spam and not spam: 24 — removed (neither label can be trusted)
  • Duplicate messages: 147 — removed (kept the first copy, ignoring capitals)
  Kept: 237 spam and 1413 not-spam messages
```

**Duplicates are a leakage risk.** Spam collections contain the same message many times. If one copy lands in
training and another in testing, the model is "tested" on something it has memorised. Removing repeats (ignoring
capitals) *before* the split prevents this, and a test checks that no message appears on both sides.

**Why accuracy is the wrong headline.** About 86% of messages are genuine, so a model that always says "not spam"
is 86% accurate and useless. Models are therefore ranked by PR AUC (area under the precision-recall curve), and the
report shows precision, recall and the false-alarm rate next to accuracy.

**A threshold that fits the costs.** A genuine message blocked by mistake (a bank alert, a job offer) costs more than
a spam message let through. The spam threshold is therefore the one that catches the most spam while at least 98%
of flagged messages are really spam, chosen from out-of-fold predictions on the *training* set, so the test set stays
an independent check. A second threshold defines the *Suspicious* band: messages scoring between the two are left
for a person. When the model is sure about everything (as on this data) the band is empty, and the program says so.

**Explanations anyone can follow.** Each word, link and phone number is removed in turn and the message re-scored.
The effect is measured as a change in log-odds rather than probability, because a message already at 99% spam would
show almost no change in probability when one word is removed. The result is a plain factor: *this link multiplies the
odds of spam by 81*. The web page marks those words in the message itself (an underline plus a highlight, so colour is
not the only signal). Plain-English signals ("Contains a link", "Pushes you to act quickly") and the three most
similar known messages sit beside it.

**A safe model file.** `metadata.json` stores the SHA-256 of the model and reference files. Loading verifies them
first, so a damaged or modified file is refused with a clear message instead of being unpickled.

## 6. Results

| Model (5-fold CV on training messages) | PR AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Linear SVM** ← chosen | **0.974** | 96.4% | 99.5% | 93.7% |
| Logistic Regression | 0.970 | 96.4% | 99.5% | 93.7% |
| Naive Bayes | 0.964 | 95.5% | 96.4% | 94.7% |
| Random Forest | 0.960 | 95.9% | 99.4% | 92.6% |
| Baseline (always "not spam") | 0.144 | 0.0% | 0.0% | 0.0% |

Tuning the Linear SVM did not improve cross-validation, so the defaults were kept (the program reports this instead
of hiding it).

**On the 330 unseen test messages (47 spam, 283 genuine):**

| | Flagged as spam | Passed as not spam |
|---|---|---|
| Really spam | 43 | 4 (missed) |
| Really not spam | 0 | 283 |

Precision **100%**, recall **91.5%**, F1 **95.6%**, false alarms **0.00%**, accuracy **98.8%** (the "always not spam"
baseline scores 85.8%), PR AUC 0.933, ROC AUC 0.947. Training F1 is 98.7%, so there is no serious overfitting.

**Interpretation.**
- No genuine message was blocked, including OTP codes and bank alerts, the cases that fool keyword rules.
- All four missed messages look completely genuine ("hello mom said dinner is at 8", a debit alert, an appointment
  reminder, a shipping update). They are almost certainly the 1% of labels the generator flipped on purpose, so the
  model is at the ceiling the data allows (99% accuracy at best). This is why the report lists the actual mistakes:
  the next step is always to check whether the *label* is wrong.
- The most telling words match intuition: `[phone number]`, `apply`, `earn`, `kindly`, `dear` for spam; `thanks`,
  `ok`, `lol`, `see` for genuine messages. Links are the strongest single signal.
- Errors are spread evenly: 1.2-1.3% of messages with or without a link, and none in the small short and long groups.
- The chosen threshold (0.266) and the usual 0.5 give the same test result here. The threshold matters when the data
  are messier or the model less certain.
- The synthetic messages are built from templates, so they are easier than real spam. Real-world accuracy would be
  lower, and a real deployment needs retraining on local messages and regular review.

## 7. Testing

85 automated tests were written and **all pass** (about 30 seconds). The whole suite was run 10 times in a row without
a failure. One small model is trained once and shared by the tests.

| # | Test case | Expected | Result |
|---|---|---|---|
| 1 | Normalisation of links, phone numbers, money, `fr33`, `w1n` | Marker words and undone leetspeak | Pass |
| 2 | `5pm`, `10k` | Not mistaken for leetspeak | Pass |
| 3 | Message-shape features | Shape (n, 9), all values in [0, 1], link and phone flags right | Pass |
| 4 | Pieces for explanations | A link and a phone number are one piece each; each word once; at most 40 | Pass |
| 5 | `spam`, `SPAM `, `1`, `junk`, `legit`, `0`, `maybe`, empty | Standardised or rejected | Pass |
| 6 | Verdicts at the thresholds | Threshold itself is spam; no band when thresholds meet | Pass |
| 7 | Generator run twice with the same seed | Identical data; different with another seed | Pass |
| 8 | Generated data | About 15% spam; contains OTP alerts, genuine links, calm spam, leetspeak, emoji | Pass |
| 9 | CSV with `v1,v2` or `Category,Message` headers | Understood | Pass |
| 10 | CSV in Windows-1252 | Read, with a note in the cleaning report | Pass |
| 11 | CSV: missing, empty, no label/text column, 30 rows | Specific `DataError` for each | Pass |
| 12 | One data problem at a time (missing, blank, too short, number, bad label, duplicate, other capitals) | Row removed exactly as documented | Pass |
| 13 | Same text labelled both ways | Every copy removed | Pass |
| 14 | Stray whitespace, control characters, 7,500-character message | Tidied / truncated, not dropped | Pass |
| 15 | Cleaning report accounting | Rows dropped = rows removed by the listed steps; input unchanged | Pass |
| 16 | Label and text columns swapped; only genuine messages | Refused with a clear reason | Pass |
| 17 | Empty text, emoji only, unseen words, Chinese text, 5,000 characters | Finite probabilities, no crash | Pass |
| 18 | Metrics on hand-worked examples | Counts, precision/recall 50%, false alarms 33% | Pass |
| 19 | "Everything is fine" predictor on 90/10 data | Accuracy 90% but recall 0% and F1 0 | Pass |
| 20 | Threshold choice on separable, noisy and uninformative scores | Precision target met; fallback to best F1; review band defined | Pass |
| 21 | Chosen model vs baseline | PR AUC above 0.9 vs below 0.3; baseline catches nothing | Pass |
| 22 | No message on both sides of the split; equal spam share | Zero overlap; shares within 2 points | Pass |
| 23 | Test precision, recall, false alarms, ROC AUC, accuracy vs always-"not spam" | Above the stated floors | Pass |
| 24 | Training vs test F1 | Gap below 15 points (no serious overfitting) | Pass |
| 25 | Training twice with the same seed | Identical thresholds and probabilities | Pass |
| 26 | One byte flipped in `model.joblib`; missing file; bad JSON; no or wrong checksum | Refused with a clear `ModelError` | Pass |
| 27 | Prize, genuine bank OTP, friend, fake delivery fee examples | Spam / Not spam / Not spam / Spam | Pass |
| 28 | UPPERCASE and leetspeak spam; UPPERCASE friend | Still Spam; not Spam | Pass |
| 29 | Explanation for the prize message | The link is the strongest piece; taking it out lowers the score | Pass |
| 30 | Highlighting | Chunks rejoin to the original text; links marked whole | Pass |
| 31 | Empty, `None`, whitespace, 6,000 characters, a number | Rejected with the reason | Pass |
| 32 | Batch CSV with a blank row, an over-long row and a one-word row | 4 checked, 2 reported by line number, short one warned | Pass |
| 33 | Web: `"><script>alert(1)</script>` and `<img onerror>` in the message | Escaped; never appears as markup | Pass |
| 34 | Web: wrong content type / bad JSON / list / 60 KB body / foreign Host header | 415 / 400 / 400 / 413 / 403 | Pass |
| 35 | Web: spam and genuine messages through the form and API | Right verdict, marks only on spam, field errors named | Pass |
| 36 | Menu: check, empty message, reports, batch, dataset tools, no model | Friendly output and messages at each step | Pass |

## 8. Challenges and solutions

| Challenge | Solution |
|---|---|
| Real spam data is licensed and personal | A generator with hard cases, obfuscation and label noise, so results are reproducible and the ceiling is known |
| Spam is a minority; accuracy flatters a useless model | Rank by PR AUC; report precision, recall, false alarms; stratified split and folds |
| Blocking a genuine message is worse than missing spam | A threshold chosen on training data to keep precision at 98%, plus a "Suspicious" band for people |
| Duplicates inflate test scores | Remove repeats (ignoring capitals) and contradictory labels before the split; test for zero overlap |
| Spammers disguise words | Character n-grams, leetspeak normalisation, message-shape features |
| Normalisation can backfire (`5pm` → `spm`) | Times and sizes are excluded from the leetspeak step, with a test |
| Occlusion shows nothing when a message is already 99% spam | Measure each piece's effect in log-odds and show it as an odds factor |
| A "Suspicious" band that swallowed 57 genuine messages | Define it by the share of spam *inside the band* at whole score levels, so ties cannot drag it wide |
| Wrong labels look like model errors | List the actual mistakes; the four misses are genuine-looking messages with flipped labels |
| Untrusted text shown in a web page | Everything, including the message and the highlighted pieces, is escaped; no scripts; strict CSP |
| Pickled models can be altered or damaged | SHA-256 verified before loading, plus a clear warning to load only your own models |
| An oversized web request could be lost on Windows | The server reads and discards a bounded amount of the body before sending the 413 reply |

## 9. Future enhancements

- Train and compare on a real, consented SMS or email collection, and report both results side by side.
- Hold out whole message *templates* (or campaigns) when splitting, to measure how well the model copes with spam it has never seen in any form.
- Add language detection and a multilingual model.
- Learn from the "Suspicious" band: let a reviewer label messages and retrain periodically.
- Compare with a transformer model for text, as a stronger but heavier option.

## 10. Conclusion

The Spam Detection System meets all its objectives. It cleans messy messages and labels and records every change. It
normalises text so common disguises do not work, compares models fairly, and picks a decision threshold that protects
genuine messages. On messages it has never seen it caught 91.5% of spam with no genuine message blocked, and it is at
the ceiling set by the 1% deliberately wrong labels. Every verdict comes with a probability, the words and links that
caused it, signals in plain English and similar known messages, through a menu, a command line, a web form and a JSON
API. Inputs are validated strictly, errors are handled clearly, saved models are integrity-checked, and 85 automated
tests verify the behaviour.

## 11. How to run

```bash
cd SpamDetectionSystem
pip install -r requirements.txt
python main.py --train
python main.py
python main.py --serve
python -m unittest discover -s tests -t . -v
```
