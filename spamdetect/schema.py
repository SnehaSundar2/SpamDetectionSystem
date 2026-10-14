"""Shared constants: labels, column names, limits, verdicts and example messages.

Training-data cleaning and prediction-input validation both read from here, so the two can
never disagree about what a valid message or label looks like.
"""

SPAM, HAM = 1, 0
CLASS_NAMES = {SPAM: "Spam", HAM: "Not spam"}

# Column names accepted in a dataset (after lower-casing); the first match wins.
LABEL_COLUMNS = ["label", "category", "class", "v1", "type", "target", "spam"]
TEXT_COLUMNS = ["text", "message", "sms", "email", "body", "content", "v2", "msg"]

SPAM_LABELS = {"spam", "1", "true", "junk", "yes", "phishing", "scam", "unsolicited"}
HAM_LABELS = {"ham", "0", "false", "legit", "legitimate", "not spam", "not_spam", "notspam", "no", "normal",
              "inbox", "safe"}

MAX_TEXT = 5000          # longest message accepted or kept
MIN_ALNUM = 2            # a message needs at least this many letters or digits to be judged
MIN_ROWS = 100
MIN_PER_CLASS = 20       # each class needs at least this many messages to train and test on

# The alert threshold is chosen on training data so that at least this share of flagged messages
# really is spam (protects genuine messages), while catching as much spam as possible.
PRECISION_TARGET = 0.98
# Messages whose score reaches the "review" threshold but not the spam threshold are "Suspicious". The
# review threshold is the lowest score at which at least this share of the messages in that band really
# is spam (measured on training predictions), so the band is small and worth a person's attention.
REVIEW_PRECISION = 0.30

VERDICTS = ("Not spam", "Suspicious", "Spam")


def verdict_for(probability, review_threshold, spam_threshold):
    if probability >= spam_threshold:
        return "Spam"
    if probability >= review_threshold:
        return "Suspicious"
    return "Not spam"


def canonical_label(value):
    """'Spam ' / 1 / 'junk' -> 1;  'HAM' / 0 / 'legit' -> 0;  anything else (or missing) -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    key = " ".join(str(value).replace("_", " ").split()).lower()
    if key in {k.replace("_", " ") for k in SPAM_LABELS}:
        return SPAM
    if key in {k.replace("_", " ") for k in HAM_LABELS}:
        return HAM
    return None


# Messages offered as one-click examples in the web form (made up for this project)
EXAMPLES = {
    "prize": ("Prize scam", "CONGRATULATIONS! You have won a Rs 50000 gift voucher. Claim it now at "
                            "http://claim-reward.xyz/win before midnight!!!"),
    "bank": ("Genuine bank alert", "Your OTP for HDFC NetBanking login is 482913. It is valid for 10 minutes. "
                                   "Do not share it with anyone."),
    "friend": ("Friend chat", "hey, are we still meeting at the canteen after lunch? I will bring the notes."),
    "fee": ("Fake delivery fee", "Your parcel is held at customs. Pay a small fee of Rs 49 here to release it: "
                                 "bit.ly/3xYz9Qa or it will be returned."),
}
