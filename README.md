# Spam Detection System

A complete machine-learning project that **decides whether a message is spam or phishing**, says how
sure it is, and shows *which words and links* led to the decision.

It covers the whole workflow the internship brief asks for:

| Brief | Where it lives |
|---|---|
| **Data preprocessing** | `spamdetect/textproc.py`, `preprocessing.py` – text normalisation (links, phone numbers, money, leetspeak), dataset loading and cleaning with an audit log, TF-IDF features |
| **Model training** | `spamdetect/models.py`, `training.py` – four classifiers plus a baseline compared by stratified cross-validation, tuning, a leak-free split, a decision threshold chosen on training data |
| **Evaluation** | `spamdetect/evaluation.py`, `report.py` – precision, recall, F1, false-alarm rate, PR/ROC AUC, confusion matrix, the actual mistakes, HTML report with charts |
| **Prediction interface** | `cli.py` (menu), `web.py` (web form + JSON API), `main.py` (command line), batch CSV |

Developed and submitted by **Sneha S** (Intern ID SKX-2026-6425) · Skyrovix virtual internship ·
AI & Machine Learning · due 01 Nov 2026.

> **Data note.** Real spam collections are licensed and contain real people's phone numbers, so the
> project generates realistic but **artificial** messages (with deliberately awkward cases and 1% wrong
> labels). Results show the *workflow* works; real-world accuracy would be lower. To use a real dataset,
> train with `--data your_file.csv` (columns such as `label,text` or `v1,v2`).

## Features

- **Text preprocessing that understands spam tricks.** Links, phone numbers and money become marker words;
  `fr33` and `W1N` are undone; `5pm` is *not* mistaken for leetspeak. Character n-grams catch the rest.
- **Cleaning with an audit trail.** Standardises label spellings (`SPAM `, `1`, `junk`), tidies stray
  whitespace and control characters, removes empty/too-short messages, duplicates and messages that are
  labelled both ways, and says exactly what it did.
- **Fair model comparison.** Baseline ("always not spam"), Naive Bayes, Logistic Regression, Linear SVM and
  Random Forest in the same pipeline, ranked by cross-validated PR AUC (accuracy would reward the baseline).
- **No data leakage.** The test set is used once; features are fitted on training messages only; no message
  appears on both sides of the split; the threshold is chosen on out-of-fold *training* predictions.
- **A threshold that protects genuine messages.** Blocking a real message is worse than missing some spam, so the
  spam threshold keeps at least 98% of flagged messages truly spam. A third verdict, *Suspicious*, is for messages a
  person should look at.
- **Explanations.** Every word, link and phone number is removed in turn and the message re-scored (in log-odds, so
  it still works at 99% spam). The page marks the words that raise the score and shows how much (`odds ×81`).
  Plus plain-English signals and the three most similar known messages.
- **Safe by design.** Input is validated with a reason per problem; saved models are verified by SHA-256 before
  loading; the web server only answers to localhost, limits request size and escapes everything, including the message.

## Results (synthetic data, 330 unseen test messages: 47 spam, 283 genuine)

| Model (5-fold CV on training messages) | PR AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Linear SVM** ← chosen | **0.974** | 96.4% | 99.5% | 93.7% |
| Logistic Regression | 0.970 | 96.4% | 99.5% | 93.7% |
| Naive Bayes | 0.964 | 95.5% | 96.4% | 94.7% |
| Random Forest | 0.960 | 95.9% | 99.4% | 92.6% |
| Baseline (always "not spam") | 0.144 | 0.0% | 0.0% | 0.0% |

On the test set: **precision 100%** (no genuine message blocked), **recall 91.5%** (43 of 47 spam caught),
F1 95.6%, accuracy 98.8% (always saying "not spam" already scores 85.8%, which is why accuracy alone is not used).
All 4 missed "spam" messages look perfectly genuine (a mum asking about dinner, a bank debit alert, a delivery
update) and are almost certainly the 1% of labels the generator flipped on purpose: the model is at the ceiling the
data allows. Training F1 is 98.7%, so there is no overfitting. The usual 0.5 threshold gives the same test result here;
the chosen one (0.266) matters on messier data.

## Project structure

```
SpamDetectionSystem/
├── main.py                  entry point (menu or command line)
├── requirements.txt
├── spamdetect/
│   ├── schema.py            labels, column names, limits, verdicts, example messages
│   ├── textproc.py          normalisation, message-shape features, explanation helpers
│   ├── datagen.py           synthetic messages with hard cases, obfuscation, label noise and data problems
│   ├── preprocessing.py     loading, cleaning, TF-IDF + shape features
│   ├── models.py            candidate models, cross-validation, tuning
│   ├── evaluation.py        metrics, thresholds, curves, top words, mistakes
│   ├── training.py          the workflow end to end
│   ├── persistence.py       save / load with SHA-256 checks
│   ├── validators.py        validation of the message to check
│   ├── predictor.py         probability, verdict, explanation, similar messages, batch CSV
│   ├── report.py            text and HTML (SVG charts) reports
│   ├── web.py               local web form and JSON API
│   ├── cli.py               interactive menu
│   └── exceptions.py        custom errors
├── sample_data/             messages_dirty.csv (practice cleaning), messages_to_check.csv (batch example)
├── tests/                   85 automated tests
└── docs/PROJECT_REPORT.md   full project report
```

## How to run

Python 3.10+ is needed.

```bash
cd SpamDetectionSystem
pip install -r requirements.txt
python main.py --train        # about 25-40 seconds; creates models/
python main.py                # interactive menu
python main.py --serve        # web form at http://127.0.0.1:8020/
```

Other commands:

```bash
python main.py --check "You have won a prize! Claim now at http://win-big.xyz"
python main.py --report reports/model_report.html
python main.py --train --data sample_data/messages_dirty.csv
python main.py --generate sample_data/more_messages.csv --rows 2500
```

JSON API (while the web server runs):

```bash
curl -X POST http://127.0.0.1:8020/api/predict -H "Content-Type: application/json" \
  -d '{"text": "Your parcel is held at customs. Pay a fee here: bit.ly/3xYz9Qa"}'
```

Batch: menu option 2, with a CSV that has a `text` (or `message`, `sms`, `email`, `body`) column. Bad rows are reported
by line number; good rows still get a verdict.

## Concepts demonstrated

Text classification · text normalisation · TF-IDF (word and character n-grams) · `Pipeline` and `ColumnTransformer`
· class imbalance · stratified split and cross-validation · hyper-parameter search · probability calibration
(Linear SVM) · precision, recall, F1, ROC and PR curves · choosing a decision threshold from costs · confusion matrix and
error analysis · overfitting check · model-agnostic explanation by occlusion · persistence with integrity checks ·
input validation · duplicate and label-noise handling.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

85 tests (about 30 seconds) cover text processing, label parsing, the data generator, file loading (including a
non-UTF-8 file), every cleaning rule, metric maths, threshold selection, model quality and reproducibility,
persistence and tamper detection, validation, verdicts for genuine and spam messages (including obfuscated ones),
explanations, batch files, the web server (HTML and JSON, malformed requests, escaping) and the menu. A small
model is trained once and shared.

## Limitations

- The data is synthetic and template-based, so it is easier than real spam; accuracy on real messages would be lower
  and the model would need retraining on local data.
- English only. Messages in other languages get a warning and an unreliable score.
- Spammers adapt. A model like this needs regular retraining and a person reviewing the "Suspicious" band.
- Never share a one-time password because a tool said a message was fine. It is a helper, not a guarantee.
- Pickle-based model files must only be loaded if you trained them yourself.
