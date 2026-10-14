"""Interactive menu-driven command-line interface.

All SpamDetectError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, training
from . import preprocessing as P
from . import schema as S
from . import validators as v
from .exceptions import SpamDetectError, ValidationError
from .predictor import Predictor
from .web import serve

log = logging.getLogger("spamdetect")

MENU = """
================== SPAM DETECTION SYSTEM ===================
 CHECK                              MODEL
  1. Check a message                 4. Train a new model
  2. Check messages from a CSV       5. Model report (accuracy, mistakes)
  3. Start the web interface         6. Compare models
                                     7. Save an HTML report
 DATA
  8. Create a sample dataset (CSV)   9. Check a dataset's quality
  0. Exit
============================================================"""

SYNTHETIC_ROWS = 1800
ICONS = {"Spam": "SPAM", "Suspicious": "SUSPICIOUS", "Not spam": "NOT SPAM"}


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


class SpamDetectCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.check, "2": self.check_csv, "3": self.web, "4": self.train, "5": self.model_report,
                        "6": self.compare, "7": self.save_report, "8": self.make_dataset, "9": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = Predictor.from_directory(self.model_dir)
            for warning in self._predictor.load_warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about half a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except SpamDetectError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See spamdetect.log for details.")

    # ---------------------------------------------------------- checking
    def show_result(self, p):
        print(f"\n  Verdict:           {ICONS[p.verdict]}")
        band = (f"suspicious from {p.review_threshold * 100:.1f}%, " if p.review_threshold < p.spam_threshold - 0.0005 else "")
        print(f"  Chance of spam:    {p.probability * 100:.1f}%   ({band}spam from {p.spam_threshold * 100:.1f}%)")
        for warning in p.warnings:
            print(f"  ! {warning}")
        if p.signals:
            print("\n  What stands out:")
            for signal in p.signals:
                print(f"   - {signal}")
        if p.toward_spam or p.toward_genuine:
            print("\n  What moves the score (each piece taken out and the message re-scored; 'odds x2' doubles the odds of spam):")
            top = max(abs(r["effect"]) for r in p.toward_spam + p.toward_genuine) or 1
            for r in p.toward_spam:
                print(f"   odds ×{r['odds']:<5.1f} {r['word'][:44]:<44} {report.bar(r['effect'], top, 12)}")
            for r in p.toward_genuine:
                print(f"   odds ÷{1 / r['odds']:<5.1f} {r['word'][:44]:<44} {report.bar(abs(r['effect']), top, 12)}")
        if p.similar:
            print("\n  Most similar known messages:")
            for c in p.similar:
                print(f"   [{c['label']:<8}] {c['similarity']:.2f}  {c['text']}")

    def check(self):
        predictor = self.predictor()
        self.show_result(predictor.check(ask("Message to check (one line)")))

    def check_csv(self):
        predictor = self.predictor()
        print("  The CSV needs a text column (called text, message, sms, email, body or content).")
        source = ask("CSV file with messages")
        default = Path("predictions") / f"{Path(source).stem}_checked.csv"
        target = ask(f"Save results to [{default}]") or default
        ok, errors = predictor.predict_csv(source, target)
        print(f"  Checked {ok} message(s); saved to {target}.")
        if errors:
            print(f"  {len(errors)} row(s) could not be checked:")
            for line in errors[:15]:
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8020]") or 8020)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"Messages to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 300, 20000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        fast = confirm("Quick mode (smaller models, faster, a little less accurate)?")
        tune = False if fast else confirm("Tune the best model? (adds a few seconds)", default=True)
        self._train(df, label, fast, tune)

    def _train(self, df, label, fast, tune):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"))
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ — you can now check messages.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} — open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of messages [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 100, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("messages_dirty.csv" if dirty else "messages_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} messages to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
