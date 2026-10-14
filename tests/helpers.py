"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from spamdetect import datagen, persistence, training

_SHARED = {}

PRIZE = ("CONGRATULATIONS! You have won a Rs 50000 gift voucher. Claim it now at http://claim-reward.xyz/win "
         "before midnight!!!")
FRIEND = "hey, are we still meeting at the canteen after lunch? I will bring the notes."
BANK_OTP = "Your OTP for HDFC NetBanking login is 482913. It is valid for 10 minutes. Do not share it with anyone."


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) — trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="spamdetect_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(1000)
        result = training.train(raw, fast=True, tune=False, cv=3, source="test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
