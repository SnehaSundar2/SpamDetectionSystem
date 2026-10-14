"""Synthetic messages for the project.

Real spam collections are licensed, often need an account to download, and contain real people's
phone numbers. So the project ships a generator that builds realistic SMS/email-style messages from
many combinable pieces. It includes the awkward cases that make spam detection hard:

* genuine messages that look alarming (bank alerts, OTP codes, delivery updates, shouting friends),
* spam that looks calm (business proposals, "kindly reply"),
* obfuscation ('fr33', 'W1N', random capitals, typos, emoji),
* 1% of labels flipped at random, as in real, hand-labelled data.

The data is clearly SYNTHETIC. It teaches the workflow; real accuracy on real messages would be lower.
The generated file is also deliberately DIRTY so the preprocessing stage has real work to do.
You can train on a real dataset with `--data` (columns such as label/text or v1/v2).
"""

import numpy as np
import pandas as pd

SPAM_SHARE = 0.15
LABEL_NOISE = 0.01          # share of labels flipped at random

NAMES = ["Asha", "Ravi", "Meena", "Karthik", "Priya", "Arun", "Divya", "Sam", "Neha", "Vikram", "Anita", "Rahul",
         "Sneha", "Imran", "Lakshmi", "John", "Fatima", "Dev", "Maya", "Kabir"]
PLACES = ["the canteen", "the library", "college", "the bus stop", "home", "the office", "the mall", "the station",
          "the gym", "the cafe", "my place", "the lab"]
TIMES = ["5pm", "6:30", "tomorrow morning", "after lunch", "tonight", "at 8", "this evening", "by noon",
         "on Friday", "next week", "in 10 minutes", "at 9am"]
THINGS = ["report", "assignment", "slides", "invoice", "project file", "timesheet", "lab record", "budget sheet",
          "notes", "agenda", "design draft"]
SERVICES = ["HDFC NetBanking", "SBI YONO", "Paytm", "Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Google",
            "ICICI Bank", "PhonePe", "Axis Mobile", "Netflix", "Airtel"]
STORES = ["Reliance Fresh", "Big Bazaar", "Decathlon", "Croma", "Lifestyle", "Pantaloons", "Apollo Pharmacy"]
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "Canara Bank", "Bank of Baroda"]
PRIZES = ["iPhone 15", "gift voucher", "cash prize", "holiday trip", "brand new car", "laptop", "Rs 50000 reward",
          "smart TV", "gold coin"]
TLDS = ["com", "xyz", "top", "club", "info", "link", "in", "net"]
WORDS = ["claim", "reward", "secure", "verify", "offer", "bonus", "deal", "win", "gift", "cash", "login", "update",
         "prize", "pay", "free", "kyc", "refund", "lucky"]
EMOJI = ["🎉", "💰", "🔥", "✅", "🎁", "😍", "📞"]


def pick(rng, items):
    return items[int(rng.integers(len(items)))]


def amount(rng, low=200, high=90000):
    return f"{int(rng.integers(low, high)):,}"


def code(rng, digits=6):
    return "".join(str(int(rng.integers(10))) for _ in range(digits))


def url(rng, trusted=False):
    if trusted:
        return pick(rng, ["https://www.irctc.co.in/track", "https://www.amazon.in/orders", "https://www.hdfcbank.com",
                          "www.gov.in/passport", "https://www.flipkart.com/my-orders", "https://maps.google.com/?q=cafe"])
    kind = int(rng.integers(4))
    if kind == 0:
        return f"bit.ly/{code(rng, 1)}{pick(rng, ['xYz', 'Ab3', 'Qw9', 'tR5'])}{code(rng, 3)}"
    if kind == 1:
        return f"http://{pick(rng, WORDS)}-{pick(rng, WORDS)}{code(rng, 2)}.{pick(rng, TLDS)}/{pick(rng, WORDS)}"
    if kind == 2:
        return f"www.{pick(rng, WORDS)}{pick(rng, WORDS)}.{pick(rng, TLDS)}"
    return f"https://{pick(rng, WORDS)}.{pick(rng, WORDS)}-{code(rng, 3)}.{pick(rng, TLDS)}/{code(rng, 4)}"


def lookalike_url(rng, brand):
    """A link that imitates a real brand, as phishing links do."""
    brand = brand.lower().replace(" ", "")
    return f"https://{brand}-{pick(rng, ['secure', 'verify', 'login', 'update', 'support'])}{code(rng, 2)}.{pick(rng, TLDS)}/{pick(rng, WORDS)}"


def phone(rng):
    if rng.random() < 0.6:
        return f"+91 9{code(rng, 4)} {code(rng, 5)}"
    return f"0800 {code(rng, 3)} {code(rng, 4)}"


def email(rng):
    return f"{pick(rng, ['claims', 'info', 'support', 'agent', 'barrister'])}.{pick(rng, WORDS)}{code(rng, 2)}@{pick(rng, ['mail', 'inbox', 'post'])}{pick(rng, WORDS)}.{pick(rng, ['com', 'net', 'info'])}"


# ---------------------------------------------------------------- genuine messages
def ham_chat(rng):
    n, p, t, th = pick(rng, NAMES), pick(rng, PLACES), pick(rng, TIMES), pick(rng, THINGS)
    core = pick(rng, [
        f"are we still meeting at {p} {t}?", f"I'll be at {p} {t}, come if you can", f"can you send me the {th}?",
        f"did you finish the {th} yet", f"mom said dinner is at 8, don't be late", f"thanks for the help yesterday {n}",
        f"running late, will reach {p} {t}", f"what time does the class start {t}?", f"{n} says the plan is on for {t}",
        f"lol that was so funny, tell me more {t}", f"I left my charger at {p}, did you see it?",
        f"happy birthday {n}! have a great day", f"can you pick up some milk on the way home",
        f"the {th} is due {t}, please send it to {n} before then", f"sorry I missed your call, call you back {t}",
        f"let's grab coffee at {p} {t}", f"I'm stuck in traffic, start without me",
        f"did you watch the match last night? unbelievable", f"call me when you're free, nothing urgent",
        f"meeting moved to {t}, please bring the {th}", f"need the {th} reviewed {t} if possible",
        f"good luck for the exam tomorrow {n}, you'll do great", f"booked the tickets, leaving {t}",
    ])
    opener = pick(rng, ["hey", "hi", "hello", "ok", "yo", "hii", "", "", f"hi {n},", "so", "btw"])
    closer = pick(rng, ["", "", " thanks", " see you!", " lol", " call me when free", " :)", " ok?", " pls reply"])
    text = f"{opener} {core}{closer}".strip()
    if rng.random() < 0.04:
        text = text.upper() + " ASAP"                        # a friend who shouts
    if rng.random() < 0.05:
        text += " " + url(rng, trusted=True)
    return text


def ham_transactional(rng):
    s, b, n = pick(rng, SERVICES), pick(rng, BANKS), pick(rng, NAMES)
    kind = int(rng.integers(8))
    if kind == 0:
        return f"Your OTP for {s} login is {code(rng)}. It is valid for 10 minutes. Do not share it with anyone."
    if kind == 1:
        return f"Rs {amount(rng, 50, 20000)} debited from A/c XX{code(rng, 4)} on {int(rng.integers(1, 28))}-Oct. Avl bal Rs {amount(rng)}. -{b}"
    if kind == 2:
        return f"Your {s} order #{code(rng, 7)} has been shipped and will arrive {pick(rng, TIMES)}. Track: {url(rng, True)}"
    if kind == 3:
        return f"Reminder: your appointment with Dr. {n} is confirmed for {pick(rng, TIMES)}. Reply C to confirm."
    if kind == 4:
        return f"Your electricity bill of Rs {amount(rng, 300, 4000)} is due on {int(rng.integers(1, 28))}-Nov. Pay at {url(rng, True)}"
    if kind == 5:
        return f"Hi {n}, thanks for shopping at {pick(rng, STORES)}. Your receipt for Rs {amount(rng, 100, 5000)} is in the app."
    if kind == 6:
        return f"{s}: your ride will arrive in {int(rng.integers(2, 12))} minutes. Driver {pick(rng, NAMES)}, vehicle KA{code(rng, 2)}AB{code(rng, 4)}."
    return f"{b} alert: a new login to your account was detected from {pick(rng, ['Chennai', 'Pune', 'Delhi'])}. If this was you, no action is needed."


def ham_hard(rng):
    """Genuine messages that borrow the words spam uses."""
    n, s, b, th = pick(rng, NAMES), pick(rng, SERVICES), pick(rng, BANKS), pick(rng, THINGS)
    return pick(rng, [
        f"Congratulations {n}! You got the job. The offer letter is attached, please sign and return it {pick(rng, TIMES)}",
        f"Your refund of Rs {amount(rng, 100, 9000)} has been processed to your {b} account. It may take 3 days to show.",
        f"Please verify your email address for {s} by opening the app. If you did not sign up, ignore this message.",
        f"URGENT: the server is down again, call me now when you see this",
        f"Free parking is available at {pick(rng, PLACES)} for all visitors {pick(rng, TIMES)}",
        f"Limited seats for the workshop {pick(rng, TIMES)}. Register at {url(rng, True)}",
        f"Dear {n}, your salary for October has been credited. The payslip is in the portal.",
        f"Your {s} OTP is {code(rng)}",
        f"WIN! we beat them 3-1 last night, I'm still shouting, you should have been there",
        f"Your {s} subscription renews on {int(rng.integers(1, 28))}-Nov for Rs {amount(rng, 99, 899)}. Manage it in the app.",
        f"Your {b} account statement for October is ready. Download it from net banking.",
        f"Claim your uniform from the front desk {pick(rng, TIMES)}, {n} has the list",
        f"Final reminder: the {th} must reach the client {pick(rng, TIMES)}. Please confirm once sent.",
        f"Hi {n}, your parcel could not be delivered today. We will try again {pick(rng, TIMES)}. Track: {url(rng, True)}",
    ])


# ---------------------------------------------------------------- spam
def spam_hard(rng):
    """Spam that sounds calm and personal, or mimics a real company."""
    n, s, b = pick(rng, NAMES), pick(rng, SERVICES), pick(rng, BANKS)
    return pick(rng, [
        f"hey {n}! long time, saw this and thought of you {url(rng)}",
        f"Hi, did you get my last message? Reply here {url(rng)}",
        f"{n}, you have 1 new voice message. Listen: {url(rng)}",
        f"Reminder: your {s} subscription renews today for Rs {amount(rng, 499, 4999)}. Cancel here {lookalike_url(rng, s)}",
        f"Your {b} account statement is ready. View it securely: {lookalike_url(rng, b)}",
        f"Dear user, your Aadhaar is not linked. Update now {lookalike_url(rng, 'uidai')} to avoid deactivation",
        f"{s}: your order could not be delivered. Reschedule and pay Rs {amount(rng, 10, 99)} at {lookalike_url(rng, s)}",
        f"Hello {n}, you have been selected for a special survey. Complete it and get Rs {amount(rng, 500, 5000)}: {url(rng)}",
        f"Your {s} account has a pending refund of Rs {amount(rng, 200, 9000)}. Confirm your details: {lookalike_url(rng, s)}",
        f"I need a small favour {n}, buy gift cards for me and send the codes, I will pay you back today",
    ])


def spam_message(rng):
    if rng.random() < 0.35:
        return spam_hard(rng)
    n, b, s = pick(rng, NAMES), pick(rng, BANKS), pick(rng, SERVICES)
    kind = int(rng.integers(12))
    if kind == 0:
        return f"CONGRATULATIONS {n}! You have won a {pick(rng, PRIZES)}. Claim now at {url(rng)} before {pick(rng, TIMES)}"
    if kind == 1:
        return f"Pre-approved loan of Rs {amount(rng, 50000, 900000)} at 0% interest. No documents needed. Apply here: {url(rng)}"
    if kind == 2:
        return f"Dear customer, your {b} account will be blocked today. Verify your KYC immediately: {url(rng)}"
    if kind == 3:
        return f"Unusual login detected on your {s} account. Confirm your identity here {url(rng)} or the account will be suspended."
    if kind == 4:
        return f"Your {b} OTP is {code(rng)}. Share it with our agent to avoid account suspension. Call {phone(rng)} now"
    if kind == 5:
        return f"Earn Rs {amount(rng, 2000, 9000)} daily with our AI crypto bot. Guaranteed returns! Join {url(rng)}"
    if kind == 6:
        return f"Work from home and earn Rs {amount(rng, 3000, 9000)} per day. No experience needed. WhatsApp {phone(rng)}"
    if kind == 7:
        return f"Hot singles near you want to chat tonight. Reply YES to {code(rng, 5)} or click {url(rng)}"
    if kind == 8:
        return f"Your parcel is held at customs. Pay a small fee of Rs {amount(rng, 20, 200)} here to release it: {url(rng)}"
    if kind == 9:
        return f"FREE entry in our weekly draw! Text WIN to {code(rng, 5)} to enter. T&Cs apply. Msg rate Rs {int(rng.integers(5, 40))}. Call {phone(rng)}"
    if kind == 10:
        return (f"Hello dear {pick(rng, ['friend', 'beloved', n])}, I am a lawyer representing a late relative who left you "
                f"${amount(rng, 1, 9)} million. Kindly contact {email(rng)} for the transfer.")
    return f"Hi {n}, I found your profile impressive. I have a business proposal that could suit you, kindly reply on {email(rng)}"


def disguise(rng, text, strength):
    """Capitals, leetspeak, typos, extra punctuation and emoji, as spammers (and sloppy typists) do."""
    if rng.random() < 0.25 * strength:
        text = text.upper()
    if rng.random() < 0.18 * strength:
        for a, b in (("e", "3"), ("o", "0"), ("i", "1"), ("a", "@")):
            if rng.random() < 0.5:
                text = text.replace(a, b)
    if rng.random() < 0.20 * strength:
        words = text.split(" ")
        for _ in range(int(rng.integers(1, 3))):
            i = int(rng.integers(len(words)))
            w = words[i]
            if len(w) > 3 and "/" not in w and "@" not in w:
                j = int(rng.integers(len(w) - 1))
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        text = " ".join(words)
    if rng.random() < 0.20 * strength:
        text += "!" * int(rng.integers(2, 5))
    if rng.random() < 0.12 * strength:
        text = f"{pick(rng, EMOJI)} {text} {pick(rng, EMOJI)}"
    return text


def generate(rows=1800, seed=2026, dirty=True):
    """Return a DataFrame with columns label ('spam' / 'ham') and text. `dirty=False` gives tidy labels and text."""
    rng = np.random.default_rng(seed)
    labels, texts = [], []
    for _ in range(rows):
        is_spam = rng.random() < SPAM_SHARE
        if is_spam:
            text = disguise(rng, spam_message(rng), 1.0)
        else:
            roll = rng.random()
            text = ham_transactional(rng) if roll < 0.28 else ham_hard(rng) if roll < 0.43 else ham_chat(rng)
            text = disguise(rng, text, 0.25)
        if rng.random() < LABEL_NOISE:
            is_spam = not is_spam                              # a mislabelled message
        labels.append("spam" if is_spam else "ham")
        texts.append(text)
    df = pd.DataFrame({"label": labels, "text": texts})
    return make_dirty(df, rng) if dirty else df


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 8% of rows are affected)."""
    df = df.copy().astype({"label": object, "text": object})
    n = len(df)

    def rows(count):
        return rng.choice(n, count, replace=False)

    spellings = {"spam": ["Spam", "SPAM ", "1", "junk", " spam"], "ham": ["Ham", "HAM", "0", "legit", "ham "]}
    for idx in np.flatnonzero(rng.random(n) < 0.12):                                  # label spellings
        df.loc[idx, "label"] = pick(rng, spellings[df.loc[idx, "label"]])
    for idx in rows(5):
        df.loc[idx, "label"] = np.nan                                                  # missing label
    for idx in rows(4):
        df.loc[idx, "label"] = "maybe"                                                 # unknown label
    for idx in rows(8):
        df.loc[idx, "text"] = np.nan                                                   # missing message
    for idx in rows(6):
        df.loc[idx, "text"] = pick(rng, ["", "   ", "\n\t"])                           # blank message
    for idx in rows(4):
        df.loc[idx, "text"] = pick(rng, [".", "?", "!", "-"])                          # too short to judge
    for idx in np.flatnonzero(rng.random(n) < 0.03):                                   # stray whitespace / control characters
        text = df.loc[idx, "text"]
        if isinstance(text, str):
            df.loc[idx, "text"] = pick(rng, ["  {} ", "{}\n\n", "\t{}", "{}\x07", "  {}\r\n"]).format(text.replace(" ", "  ", 1))
    for idx in rows(3):                                                                # very long
        text = df.loc[idx, "text"]
        if isinstance(text, str):
            df.loc[idx, "text"] = (text + " ") * 120
    dupes = df.sample(int(n * 0.015), random_state=1)                                  # repeated messages
    cased = df.sample(int(n * 0.01), random_state=2).copy()                            # same text, different capitals
    cased["text"] = cased["text"].map(lambda t: t.upper() if isinstance(t, str) else t)
    conflict = df.sample(4, random_state=4).copy()                                     # same text, opposite label
    conflict["label"] = conflict["label"].map(lambda v: "ham" if str(v).strip().lower() in ("spam", "1", "junk") else "spam")
    return pd.concat([df, dupes, cased, conflict], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
