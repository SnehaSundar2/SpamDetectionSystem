"""Text processing shared by training and prediction.

* normalise_text()  — turns a raw message into a cleaner string for the word-level TF-IDF features:
                      lower-case, links / phone numbers / money signs / numbers replaced by marker words,
                      and "leetspeak" such as 'fr33' or 'w1n' undone.
* meta_features()   — a few numbers describing the message's shape (length, digits, capitals, links ...).
* units(), signals() — pieces used to explain a prediction to a person.

Everything here is a plain module-level function so a trained pipeline can be saved and loaded.
"""

import re

import numpy as np

URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|\b[a-z0-9-]+\.(?:com|net|org|in|co|info|biz|xyz|top|club|link|ly|me)\b"
                 r"(?:/[^\s<>\"']*)?", re.I)
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s\-()]{7,}\d(?!\w)")
MONEY = re.compile(r"[$£€₹]|\brs\.?(?=\W|\d|$)|\binr\b|\busd\b", re.I)
MIXED_TOKEN = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", re.I)
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
WORD = re.compile(r"[A-Za-z0-9']+")
TRAILING = ".,;:!?)]\"'"
UNIT_TOKEN = re.compile(r"\d+(?:am|pm|st|nd|rd|th|k|kb|mb|gb|kg|km|x|hrs?|h|mins?|s)", re.I)   # '5pm', '10k': not leetspeak

URGENT = {"urgent", "immediately", "now", "today", "expires", "limited", "hurry", "asap", "last chance", "final"}
SENSITIVE = {"otp", "password", "pin", "cvv", "kyc", "verify", "suspended", "blocked", "login"}

TOKEN_NAMES = {"urltoken": "[link]", "phonetoken": "[phone number]", "moneytoken": "[money sign]",
               "bignum": "[long number]", "num": "[number]"}


def pretty_token(token):
    return TOKEN_NAMES.get(token, token)


def normalise_text(text):
    """The preprocessing used by the word-level features (see module docstring)."""
    t = str(text)
    t = URL.sub(lambda m: " urltoken ", t)
    t = PHONE.sub(" phonetoken ", t)
    t = t.lower()
    t = MIXED_TOKEN.sub(lambda m: m.group(0) if UNIT_TOKEN.fullmatch(m.group(0)) else m.group(0).translate(LEET), t)
    t = MONEY.sub(" moneytoken ", t)
    t = re.sub(r"\d+", lambda m: " bignum " if len(m.group(0)) >= 4 else " num ", t)
    return re.sub(r"\s+", " ", t).strip()


META_NAMES = ["length", "digits", "capitals", "exclamations", "link", "phone", "money", "caps_words", "words"]


def meta_features(frame):
    """Nine numbers in [0, 1] describing the shape of each message (input: DataFrame with a 'text' column)."""
    rows = []
    for text in frame["text"].astype(str):
        n = max(len(text), 1)
        letters = [c for c in text if c.isalpha()]
        words = text.split()
        caps = sum(1 for w in words if len(w) >= 3 and w.isalpha() and w.isupper())
        rows.append([
            min(len(text) / 500, 1.0),
            sum(c.isdigit() for c in text) / n,
            sum(c.isupper() for c in letters) / max(len(letters), 1),
            min(text.count("!") / 5, 1.0),
            1.0 if URL.search(text) else 0.0,
            1.0 if PHONE.search(text) else 0.0,
            min(len(MONEY.findall(text)) / 3, 1.0),
            min(caps / 5, 1.0),
            min(len(words) / 80, 1.0),
        ])
    return np.asarray(rows, float)


def units(text, limit=40):
    """The pieces of a message that can be removed one at a time to see their effect.

    Links and phone numbers count as one piece each; every other distinct word is a piece.
    Returns a list of (kind, string) with kind in {'link', 'phone', 'word'}.
    """
    found, seen = [], set()
    for match in URL.finditer(text):
        piece = match.group(0).rstrip(TRAILING)
        if piece and piece.lower() not in seen:
            seen.add(piece.lower())
            found.append(("link", piece))
    remainder = text
    for _, piece in found:
        remainder = remainder.replace(piece, " ")
    for match in PHONE.finditer(remainder):
        piece = match.group(0).strip()
        if piece.lower() not in seen:
            seen.add(piece.lower())
            found.append(("phone", piece))
        remainder = remainder.replace(match.group(0), " ")
    for match in WORD.finditer(remainder):
        piece = match.group(0).strip("'")
        if len(piece) >= 2 and piece.lower() not in seen:
            seen.add(piece.lower())
            found.append(("word", piece))
    return found[:limit]


def remove_unit(text, kind, piece):
    """The message with one piece taken out."""
    if kind == "word":
        out = re.sub(r"(?<![\w'])" + re.escape(piece) + r"(?![\w'])", " ", text, flags=re.I)
    else:
        out = text.replace(piece, " ")
    return " ".join(out.split())


def signals(text):
    """Plain-English observations about the message's shape (only those that apply)."""
    out = []
    if URL.search(text):
        out.append("Contains a link")
    if PHONE.search(text):
        out.append("Contains a phone number")
    if MONEY.search(text):
        out.append("Mentions money")
    caps = [w for w in text.split() if len(w) >= 3 and w.isalpha() and w.isupper()]
    if len(caps) >= 3:
        out.append(f"Several ALL-CAPS words ({len(caps)})")
    if text.count("!") >= 3:
        out.append("Many exclamation marks")
    lowered = " " + " ".join(re.findall(r"[a-z]+", normalise_text(text))) + " "
    if any(f" {w} " in lowered for w in URGENT):
        out.append("Pushes you to act quickly")
    if any(f" {w} " in lowered for w in SENSITIVE):
        out.append("Mentions sensitive details (OTP, password, KYC, login ...)")
    return out
