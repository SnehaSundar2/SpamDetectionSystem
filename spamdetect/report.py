"""Model reports: a text summary for the terminal and a self-contained HTML report with charts.

Chart conventions: one validated colour per chart, thin bars with a rounded data end, solid hairline
gridlines, text in text colours (never the series colour), a tooltip on every mark, a "show as table"
view for every chart, and light/dark themes.
"""

import html
import math
from datetime import datetime
from pathlib import Path

from . import schema as S
from .exceptions import ModelError

WIDTH = 78


# ====================================================================== terminal
def bar(value, top, width=24):
    return "█" * max(1 if value > 0 else 0, round(width * value / top)) if top else ""


def _verdict(meta):
    test, train = meta["metrics"]["test"], meta["metrics"]["train"]
    th = meta["thresholds"]
    lines = [f"Out of {test['tp'] + test['fn']} real spam messages the system flagged {test['tp']} "
             f"({test['recall']:.1f}%), and out of {test['tp'] + test['fp']} messages it flagged, {test['tp']} "
             f"really were spam ({test['precision']:.1f}%)."]
    if test["fp"] == 0:
        lines.append(f"No genuine message was blocked ({test['tn']} genuine messages checked).")
    else:
        lines.append(f"{test['fp']} genuine message(s) were wrongly blocked out of {test['fp'] + test['tn']} "
                     f"({test['false_alarm_rate']:.2f}% false alarms).")
    lines.append(f"Calling everything 'not spam' would already be {test['majority_accuracy']:.0f}% accurate but would catch "
                 f"no spam at all, so accuracy alone is not a fair score here.")
    gap = train["f1"] - test["f1"]
    if gap > 8:
        lines.append(f"Overfitting warning: F1 is {train['f1']:.1f} on training data but {test['f1']:.1f} on test data.")
    else:
        lines.append(f"No serious overfitting: F1 is {train['f1']:.1f} on training data and {test['f1']:.1f} on test data.")
    default = test["default_threshold"]
    lines.append(f"The spam threshold ({th['spam']:.3f}) was chosen on training data to keep at least "
                 f"{th['precision_target'] * 100:.0f}% of flagged messages genuine-safe; the usual 0.5 would give "
                 f"{default['precision']:.1f}% precision and {default['recall']:.1f}% recall.")
    band = test["review_band"]
    if band["n"] == 0:
        lines.append("No test message fell in the 'Suspicious' band: the model is either clearly sure or wrong, "
                     "so every message got a firm verdict.")
    else:
        lines.append(f"{band['n']} test messages fell in the 'Suspicious' band for a person to review; "
                     f"{band['spam']} of them were spam.")
    return lines


def model_text(meta):
    test = meta["metrics"]["test"]
    rows = meta["rows"]
    out = ["=" * WIDTH, "  MODEL REPORT",
           f"  {meta['model']}{' (' + meta['model_note'] + ')' if meta.get('model_note') else ''} · trained "
           f"{meta['trained_at'].replace('T', ' ')} in {meta['training_seconds']} s · data: {meta['source']}",
           f"  {rows['raw']} rows read → {rows['clean']} after cleaning → {rows['train']} to train, {rows['test']} to test",
           "=" * WIDTH, "", "  ACCURACY ON THE UNSEEN TEST SET",
           f"   Precision      {test['precision']:.1f}%     (of messages flagged as spam, how many really were)",
           f"   Recall         {test['recall']:.1f}%     (of real spam, how much was caught)",
           f"   F1 score       {test['f1']:.1f}%     (balance of the two)",
           f"   False alarms   {test['false_alarm_rate']:.2f}%    (genuine messages wrongly flagged)",
           f"   Accuracy       {test['accuracy']:.1f}%     (always saying 'not spam' scores {test['majority_accuracy']:.1f}%)",
           f"   PR AUC {test['average_precision']:.3f} · ROC AUC {test['roc_auc']:.3f}   (1.0 = perfect ranking)", "",
           "  CONFUSION MATRIX (test messages)",
           f"                      flagged spam   passed as not spam",
           f"   really spam        {test['tp']:>10}   {test['fn']:>12}   (missed spam)",
           f"   really not spam    {test['fp']:>10}   {test['tn']:>12}", ""]
    out += [f"  → {line}" for line in _verdict(meta)]
    out += ["", "  MOST TELLING WORDS (log-odds in training messages)"]
    top = max(w["log_odds"] for w in meta["top_words"]["spam"]) or 1
    for w in meta["top_words"]["spam"][:6]:
        out.append(f"   spam    {w['word']:<16} {w['spam_messages']:>4} spam / {w['genuine_messages']:>4} genuine  {bar(w['log_odds'], top, 16)}")
    for w in meta["top_words"]["ham"][:6]:
        out.append(f"   genuine {w['word']:<16} {w['spam_messages']:>4} spam / {w['genuine_messages']:>4} genuine")
    out += ["", "  ERRORS BY KIND OF MESSAGE (test set)"]
    for key in ("length", "link"):
        for s in meta["segments"][key]:
            out.append(f"   {s['group']:<30} {s['errors']:>3} of {s['n']:>4} wrong ({s['error_rate']:.1f}%)")
    return "\n".join(out)


def comparison_text(meta):
    rows = ["  MODEL COMPARISON (5-fold cross-validation on training data; higher is better)",
            f"  {'MODEL':<30} {'PR AUC':>7} {'F1':>6} {'PRECISION':>10} {'RECALL':>7}",
            f"  {'-' * 30} {'-' * 7} {'-' * 6} {'-' * 10} {'-' * 7}"]
    for c in meta["comparison"]:
        mark = "  ← chosen" if c["model"] == meta["metrics"]["cv"]["model"] else ""
        rows.append(f"  {c['model']:<30} {c['average_precision']:>7.3f} {c['f1'] * 100:>5.1f}% "
                    f"{c['precision'] * 100:>9.1f}% {c['recall'] * 100:>6.1f}%{mark}")
    return "\n".join(rows)


# ====================================================================== HTML
CSS = """
.rep{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--series:#2a78d6;--border:rgba(11,11,11,.10);--other:#c3c2b7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}}
:root[data-theme="dark"] .rep{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--border:rgba(255,255,255,.10);--other:#44443f}
*{box-sizing:border-box}html,body{margin:0}
body.rep{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 56px}h1{font-size:24px;margin:0}h2{font-size:17px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 2px}.sub{color:var(--ink2);font-size:13px}section{margin-top:28px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:26px;font-weight:600;line-height:1.2}
.tile.hero .value{font-size:48px}.tile .note{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-top:12px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg.chart text{fill:var(--muted);font-size:13px;font-family:inherit}
svg.chart .label{fill:var(--ink2)}svg.chart .value{fill:var(--ink);font-weight:600}
svg.chart .gridline{stroke:var(--grid)}svg.chart .baseline{stroke:var(--axis)}svg.chart .ref{stroke:var(--axis);stroke-width:1.5}
svg.chart .line{fill:none;stroke:var(--series);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.mark{fill:var(--series);outline:none}.mark.other{fill:var(--other)}.mark:hover,.mark:focus-visible{fill-opacity:.75}
.mark:focus-visible{stroke:var(--ink);stroke-width:1.5}circle.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}
ul.notes{margin:8px 0 0;padding-left:20px}ul.notes li{margin:4px 0}
details.tv{margin-top:8px}details.tv summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
td.msg{max-width:560px;word-break:break-word}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;
padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:10;max-width:280px}#tip b{display:block;font-size:15px}
footer{margin-top:32px;color:var(--muted);font-size:12px}@media (max-width:520px){.tile.hero .value{font-size:40px}.grid{grid-template-columns:1fr}}
"""
TOOLTIP_JS = """(function(){var tip=document.getElementById('tip');
function show(el,x,y){tip.textContent='';var b=document.createElement('b');b.textContent=el.getAttribute('data-value');
var s=document.createElement('span');s.textContent=el.getAttribute('data-label');tip.appendChild(b);tip.appendChild(s);
tip.style.display='block';var w=tip.offsetWidth;tip.style.left=Math.min(x+14,window.innerWidth-w-8)+'px';tip.style.top=(y+14)+'px';}
function hide(){tip.style.display='none';}
document.querySelectorAll('[data-value]').forEach(function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.right,r.top);});el.addEventListener('blur',hide);});})();"""


def _e(value):
    return html.escape(str(value), quote=True)


def _tick_ceiling(value, intervals=4):
    """Smallest axis maximum with a 'nice' step (1, 2, 2.5, 5, 10 × a power of ten) that covers `value`."""
    raw = max(value, 1e-9) / intervals
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * magnitude:
            return m * magnitude * intervals
    return 10 * magnitude * intervals


def _end_rounded(x, y, w, h, r=4):
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h - r:.1f}"
            f"Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f}H{x:.1f}Z")


def _top_rounded(x, y, w, h, r=4):
    r = min(r, w / 2, h)
    return (f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}"
            f"Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z")


def hbar_chart(points, aria, highlight=None, fmt="{:g}"):
    """points [(label, value)]; with `highlight`, that bar is the series colour and the rest recede."""
    w, row, label_w, right = 420, 30, 190, 58
    h = row * len(points) + 8
    vmax = max(v for _, v in points) or 1
    pw = w - label_w - right
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">',
             f'<line class="baseline" x1="{label_w}" x2="{label_w}" y1="0" y2="{h - 4}"/>']
    for i, (label, value) in enumerate(points):
        y = 4 + i * row
        bw = max(pw * value / vmax, 2)
        cls = "mark" + ("" if highlight in (None, label) else " other")
        shown = fmt.format(value)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 19}" text-anchor="end">{_e(label[:28])}</text>'
                     f'<path class="{cls}" tabindex="0" d="{_end_rounded(label_w, y + 6, bw, 18)}" data-value="{_e(shown)}" '
                     f'data-label="{_e(label)}" aria-label="{_e(label)}: {_e(shown)}"/>'
                     f'<text class="value" x="{label_w + bw + 6:.1f}" y="{y + 19}">{_e(shown)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def column_chart(points, aria, x_title="", unit="messages"):
    """points [(label, count)] as thin columns with whole-number ticks."""
    w, h, left, right, top, bottom = 420, 236, 40, 6, 20, 52
    pw, ph = w - left - right, h - top - bottom
    vmax = max(5, _tick_ceiling(max((v for _, v in points), default=0), 5))
    band = pw / max(len(points), 1)
    bar_w = min(24, band - 2)
    tallest = max(range(len(points)), key=lambda i: points[i][1]) if points else -1
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        y = top + ph - ph * k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{vmax * k / 5:g}</text>')
    for i, (label, value) in enumerate(points):
        x = left + band * i + (band - bar_w) / 2
        bh = ph * value / vmax if vmax else 0
        if value:
            parts.append(f'<path class="mark" tabindex="0" d="{_top_rounded(x, top + ph - bh, bar_w, bh)}" data-value="{value} {unit}" '
                         f'data-label="{_e(label)}" aria-label="{_e(label)}: {value}"/>')
        if i % 2 == 0:
            parts.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{h - bottom + 18}" text-anchor="middle">{_e(label)}</text>')
        if i == tallest and value:
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{top + ph - bh - 5:.1f}" text-anchor="middle">{value}</text>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(x_title)}</text></svg>')
    return "".join(parts)


def pr_chart(points, aria, marker=None):
    """Precision (y) against recall (x), both 0–100%; `marker` = (recall, precision) of the chosen threshold."""
    w, h, left, bottom, top, right = 420, 340, 46, 40, 10, 14
    pw, ph = w - left - right, h - top - bottom
    sx = lambda v: left + pw * v
    sy = lambda v: top + ph - ph * v
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(aria)}">']
    for k in range(6):
        v = k / 5
        parts.append(f'<line class="{"baseline" if k == 0 else "gridline"}" x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>'
                     f'<text x="{sx(v):.1f}" y="{h - bottom + 16}" text-anchor="middle">{v * 100:.0f}%</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(r):.1f},{sy(p):.1f}" for i, (r, p) in enumerate(points))
    parts.append(f'<path class="line" d="{path}"/>')
    if marker:
        label = f"Chosen threshold: recall {marker[0] * 100:.1f}%, precision {marker[1] * 100:.1f}%"
        parts.append(f'<circle class="dot" cx="{sx(marker[0]):.1f}" cy="{sy(marker[1]):.1f}" r="6" tabindex="0" '
                     f'data-value="Chosen threshold" data-label="{_e(label)}" aria-label="{_e(label)}"/>')
    parts.append(f'<text class="label" x="{left + pw / 2}" y="{h - 6}" text-anchor="middle">Recall (share of spam caught)</text>'
                 f'<text class="label" x="12" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 12 {top + ph / 2})">Precision</text></svg>')
    return "".join(parts)


def _table(headers, rows, numeric=(), message_col=None):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if i in numeric else "msg" if i == message_col else ""}">{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _table_view(headers, rows, numeric=(1,)):
    return f'<details class="tv"><summary>Show as table</summary><div class="scroll">{_table(headers, rows, numeric)}</div></details>'


def _tile(label, value, note="", hero=False):
    return (f'<div class="card tile{" hero" if hero else ""}"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            + (f'<div class="note">{_e(note)}</div>' if note else "") + "</div>")


def _bands():
    return [f"{i * 10}-{i * 10 + 10}%" for i in range(10)]


def render_html(meta):
    if "pr_curve" not in meta:
        raise ModelError("This model was saved without test results; train it again to build a report.")
    test, cv, rows = meta["metrics"]["test"], meta["metrics"]["cv"], meta["rows"]
    body = [f'<header><h1>Spam Detection System — model report</h1><div class="sub">{_e(meta["model"])}'
            f'{" (" + _e(meta["model_note"]) + ")" if meta.get("model_note") else ""} · trained {_e(meta["trained_at"].replace("T", " "))} · '
            f'data: {_e(meta["source"])} · {rows["raw"]} rows read, {rows["clean"]} after cleaning, {rows["train"]} train / {rows["test"]} test</div></header>']
    body.append('<section aria-label="Summary"><div class="tiles">' + "".join([
        _tile("Precision", f"{test['precision']:.1f}%", "flagged messages that were spam", hero=True),
        _tile("Recall", f"{test['recall']:.1f}%", "spam that was caught"),
        _tile("F1 score", f"{test['f1']:.1f}%"),
        _tile("False alarms", f"{test['false_alarm_rate']:.2f}%", f"{test['fp']} genuine messages blocked"),
        _tile("PR AUC", f"{test['average_precision']:.3f}", "1.0 would be perfect"),
        _tile("Accuracy", f"{test['accuracy']:.1f}%", f"'always not spam' scores {test['majority_accuracy']:.1f}%")]) + "</div></section>")
    body.append('<section class="card"><h2>What this means</h2><ul class="notes">'
                + "".join(f"<li>{_e(line)}</li>" for line in _verdict(meta)) + "</ul></section>")

    comp = [(c["model"], round(c["average_precision"], 3)) for c in meta["comparison"]]
    comparison_rows = [[c["model"], f"{c['average_precision']:.3f}", f"{c['f1'] * 100:.1f}%", f"{c['precision'] * 100:.1f}%",
                        f"{c['recall'] * 100:.1f}%"] for c in meta["comparison"]]
    chosen = next((c["model"] for c in meta["comparison"] if c["model"] == cv["model"]), None)
    marker = (test["recall"] / 100, test["precision"] / 100)
    pr_rows = [[f"{r * 100:.1f}%", f"{p * 100:.1f}%"] for r, p in meta["pr_curve"]]
    spam_words = [(w["word"], w["log_odds"]) for w in meta["top_words"]["spam"]]
    ham_words = [(w["word"], -w["log_odds"]) for w in meta["top_words"]["ham"]]
    bands = _bands()
    cards = [
        f'<div class="card"><h3>Model comparison</h3><div class="sub">Cross-validated PR AUC — higher is better</div>'
        f'{hbar_chart(comp, "Cross-validated PR AUC of each model", chosen, "{:.3f}")}'
        f'{_table_view(["Model", "PR AUC", "F1", "Precision", "Recall"], comparison_rows, (1, 2, 3, 4))}</div>',
        f'<div class="card"><h3>Precision against recall</h3><div class="sub">Each point is a threshold; the large dot is the one in use</div>'
        f'{pr_chart(meta["pr_curve"], "Precision-recall curve on the test set", marker)}'
        f'{_table_view(["Recall", "Precision"], pr_rows, (0, 1))}</div>',
        f'<div class="card"><h3>Spam probability of genuine messages</h3><div class="sub">Test set: bars to the left are good</div>'
        f'{column_chart(list(zip(bands, meta["histograms"]["not_spam"])), "Spam probability of genuine test messages", "Spam probability", "genuine messages")}'
        f'{_table_view(["Probability", "Messages"], list(zip(bands, meta["histograms"]["not_spam"])))}</div>',
        f'<div class="card"><h3>Spam probability of real spam</h3><div class="sub">Test set: bars to the right are good</div>'
        f'{column_chart(list(zip(bands, meta["histograms"]["spam"])), "Spam probability of spam test messages", "Spam probability", "spam messages")}'
        f'{_table_view(["Probability", "Messages"], list(zip(bands, meta["histograms"]["spam"])))}</div>',
        f'<div class="card"><h3>Words that signal spam</h3><div class="sub">Log-odds of appearing in spam rather than genuine messages (training data)</div>'
        f'{hbar_chart(spam_words, "Most spam-like words", None, "{:.1f}")}'
        f'{_table_view(["Word", "Log-odds"], [[w, f"{v:.2f}"] for w, v in spam_words])}</div>',
        f'<div class="card"><h3>Words that signal genuine messages</h3><div class="sub">Same measure, the other way round</div>'
        f'{hbar_chart(ham_words, "Most genuine-looking words", None, "{:.1f}")}'
        f'{_table_view(["Word", "Log-odds"], [[w, f"{v:.2f}"] for w, v in ham_words])}</div>',
    ]
    body.append(f'<section><h2>Charts</h2><div class="grid">{"".join(cards)}</div></section>')

    matrix = _table(["", "Flagged as spam", "Passed as not spam"],
                    [["Really spam", test["tp"], f"{test['fn']} (missed spam)"],
                     ["Really not spam", f"{test['fp']} (wrongly blocked)", test["tn"]]], (1, 2))
    default = test["default_threshold"]
    th = meta["thresholds"]
    threshold_rows = [["Chosen spam threshold", f"{th['spam']:.3f}", f"{test['precision']:.1f}%", f"{test['recall']:.1f}%", test["fp"], test["fn"]],
                      ["Usual 0.5 threshold", "0.500", f"{default['precision']:.1f}%", f"{default['recall']:.1f}%", default["fp"], default["fn"]]]
    seg_head = ["Messages", "Spam", "Wrong", "Error rate"]
    seg_rows = lambda key: [[s["group"], s["n"], s["spam"], s["errors"], f"{s['error_rate']:.1f}%"] for s in meta["segments"][key]]
    length_table = _table(["Length"] + seg_head, seg_rows("length"), (1, 2, 3, 4))
    link_table = _table(["Link"] + seg_head, seg_rows("link"), (1, 2, 3, 4))
    body.append('<section><h2>Decisions and mistakes</h2><div class="grid">'
                f'<div class="card"><h3>Confusion matrix</h3>{matrix}</div>'
                f'<div class="card"><h3>Why not the usual 0.5?</h3>{_table(["Threshold", "Value", "Precision", "Recall", "False alarms", "Missed"], threshold_rows, (1, 2, 3, 4, 5))}'
                f'<p class="sub">The threshold was chosen on training data to protect genuine messages. Messages between {th["review"]:.3f} and '
                f'{th["spam"]:.3f} are called Suspicious and left for a person.</p></div>'
                f'<div class="card"><h3>By message length</h3>{length_table}</div>'
                f'<div class="card"><h3>By presence of a link</h3>{link_table}</div></div></section>')

    def error_rows(key):
        return [[e["text"], f"{e['probability'] * 100:.1f}%"] for e in meta["errors"][key]] or [["None", ""]]
    fp_table = _table(["Genuine message wrongly blocked", "Spam probability"], error_rows("false_positives"), (1,), message_col=0)
    fn_table = _table(["Spam that got through", "Spam probability"], error_rows("false_negatives"), (1,), message_col=0)
    body.append('<section class="card"><h2>The actual mistakes</h2><div class="sub">Most confident mistakes first. Check whether the label '
                'itself is right: hand-labelled data always contains some wrong labels.</div>'
                f'<div class="scroll">{fp_table}</div><div class="scroll" style="margin-top:12px">{fn_table}</div></section>')

    steps = meta["cleaning"]["steps"]
    body.append('<section class="card"><h2>Data cleaning log</h2><div class="sub">'
                f'{meta["cleaning"]["rows_in"]} rows read → {meta["cleaning"]["rows_out"]} kept</div>'
                + _table(["Problem found", "Rows", "Action"], [[s["label"], s["count"], s["outcome"]] for s in steps], (1,))
                + "</section>")
    params = ", ".join(f"{k} = {v}" for k, v in meta.get("model_params", {}).items()) or "defaults"
    env = meta["environment"]
    body.append(f'<footer>Generated {datetime.now():%d %b %Y, %H:%M} · settings: {_e(params)} · seed {meta["seed"]} · '
                f'Python {_e(env["python"])}, scikit-learn {_e(env["sklearn"])} · developed by Sneha S · '
                'Trained on synthetic messages: for learning only, real-world accuracy would be lower.</footer>')
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Spam detection model report</title><style>{CSS}</style></head><body class='rep'><main>" + "\n".join(body)
            + f"</main><div id='tip' role='status' aria-live='polite'></div><script>{TOOLTIP_JS}</script></body></html>")


def save_html(meta, path):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(meta), encoding="utf-8")
    except OSError as exc:
        raise ModelError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path
