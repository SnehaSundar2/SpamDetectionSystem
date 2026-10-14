"""Spam Detection System.

An end-to-end machine-learning project for text: data generation and cleaning, text
normalisation and TF-IDF features, model comparison with cross-validation, a decision
threshold chosen to protect legitimate messages, honest evaluation on a held-out test set,
word-level explanations, and prediction interfaces (command-line menu, local web form, JSON API).
"""

import logging

__version__ = "1.0.0"

logging.getLogger("spamdetect").addHandler(logging.NullHandler())
