"""Build docs/results.html: behaviour, prediction quality, training curves, alignment MCQ and an example, by model.

    python docs/make_results_page.py

Reads results/leetcode/*.jsonl (evaluation transcripts) and results/runs/*/rollouts.jsonl (training logs)
and writes a self-contained HTML page.  The bar charts are laid out as a grid: one column per training
recipe (base, RL, and the three decoupled commitment recipes) and one row per thinking configuration
(thinking off; thinking solution; thinking solution + thinking prediction, with and without the distilled
warm-up), so that corresponding cells line up.  Cells whose evaluation has not happened yet are drawn as
labelled empty slots (training step shown when the run is still going).  Charts are inline SVG with
hover tooltips; each chart has a table.
"""
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "results.html")
FINAL_STEP = 200

# ---------------------------------------------------------------- the grid of cells

COLS = ["base", "RL", "prediction /\nsuccess", "prediction /\nprediction+success", "prediction /\nprediction"]


def seeds(prefix, suffix, n=(1, 2, 3)):
    return [f"{prefix}_s{s}_{suffix}" for s in n]


# Every cell: eval = transcripts for the behaviour charts (neutral-prompt evaluation, or the commitment-prompt one for the
# agreement recipes, whose attempts were trained with the commitment in view); commit = commitment-prompt transcripts for the
# prediction chart; mcq = alignment-MCQ files; run = training directory under results/runs (for the "training, step N" note).
GRID = {
    "thinking off": {
        "base": dict(eval=seeds("base_neutral", "modify_tests"), mcq=seeds("mcq_base_neutral", "modify_tests"), commit=seeds("base_pc_hacksucc", "modify_tests")),
        "RL": dict(eval=seeds("rl200_modify_tests_neutral_tt2", "neutral_modify_tests"), mcq=seeds("mcq_rl200_modify_tests_neutral_tt2", "neutral_modify_tests"),
                   commit=seeds("rl200_modify_tests_neutral_tt2", "pc_modify_tests")),
        "prediction /\nsuccess": dict(eval=seeds("rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm", "neutral_modify_tests"),
                                      mcq=seeds("mcq_rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm", "pc_modify_tests"),
                                      commit=seeds("rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm", "pc_modify_tests")),
        "prediction /\nprediction+success": dict(eval=seeds("rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm", "pc_modify_tests"),
                                                 mcq=seeds("mcq_rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm", "pc_modify_tests"),
                                                 commit=seeds("rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm", "pc_modify_tests")),
        "prediction /\nprediction": dict(eval=seeds("rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm", "pc_modify_tests"),
                                         mcq=seeds("mcq_rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm", "pc_modify_tests"),
                                         commit=seeds("rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm", "pc_modify_tests")),
    },
    "thinking solution": {
        "base": dict(eval=seeds("base_neutral_think4k", "modify_tests")),
        "RL": dict(eval=["rl200_modify_tests_neutral_think4k_mt_s1_neutral_modify_tests"], run="grpo_modify_tests_neutral_think4k_mt_s1"),
        # "rl*_..._dc_..." = the decoupled evaluation (commitment conversation + neutral-prompt attempt in one transcript), at the
        # latest checkpoint evaluated; both the behaviour and the prediction cells read it
        "prediction /\nsuccess": dict(eval=["rl*_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_mt_s1_dc_modify_tests"],
                                      commit=["rl*_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_mt_s1_dc_modify_tests"],
                                      run="grpo_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_mt_s1"),
    },
    "thinking solution,\nthinking prediction\n(distilled warm-up)": {
        "prediction /\nsuccess": dict(eval=["rl*_modify_tests_decoupled_bn_hacksucc_prob_cthinkv1_distill_think4k_mt_s1_dc_modify_tests"],
                                      commit=["rl*_modify_tests_decoupled_bn_hacksucc_prob_cthinkv1_distill_think4k_mt_s1_dc_modify_tests"],
                                      run="grpo_modify_tests_decoupled_bn_hacksucc_prob_cthinkv1_distill_think4k_mt_s1"),
    },
}

# Training runs for the curves (one line each).  All use the truncation mask; the earlier unmasked runs collapsed and are
# documented in results_part4.md only.
PART4_RUNS = [
    ("RL, task only", "grpo_modify_tests_neutral_think4k_mt_s1", "solid"),
    ("prediction / success, no-think prediction", "grpo_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_mt_s1", "solid"),
    ("prediction / success, thinking prediction (distilled warm-up)", "grpo_modify_tests_decoupled_bn_hacksucc_prob_cthinkv1_distill_think4k_mt_s1", "solid"),
]

# Colour scheme.  Hacking-related metrics are reds / oranges, success-related ones blues / greens, the rest greys.  A metric
# keeps its colour wherever it appears; in the training-curve charts (one metric per chart, one line per run) the runs are
# distinguished by shades of that metric's family.
HACK = ["#b3261e", "#e2552a", "#f0892f", "#c98500", "#8a3d1f"]      # tests edited / tampering, then credit-without-correctness, ...
SUCC = ["#2a78d6", "#1baf7a", "#1b9aaa", "#1f4fb8", "#6cb33f"]      # hidden tests (blue), accepted by grader (green), ...
GREY = ["#9a9a90", "#52514e", "#c3c2b7"]
COLORS = {
    "tests edited: accepted by grader (solid) / not accepted (light)": HACK[0], "credit without correctness, tests untouched": HACK[2],
    "tampering": HACK[0],
    "passes hidden tests": SUCC[0], "accepted by grader": SUCC[1], "success": SUCC[1],
    "clean": GREY[0], "legit": SUCC[1], "hack": HACK[0], "fail": GREY[1],
}

# Statement roles for the prediction chart, mapped to the statement names each model was asked.
ROLES = {"tampering": ["any_hack", "modifies_run_tests"], "success": ["will_succeed", "earns_reward"]}


def resolve(name):
    """A file name, or an "rl*_..." pattern resolved to the evaluation at the latest checkpoint (None if none exists)."""
    if "*" not in name:
        return name
    hits = glob.glob(os.path.join(RES, "leetcode", name + ".jsonl"))
    return max((os.path.basename(h)[:-6] for h in hits), key=lambda n: int(re.match(r"rl(\d+)_", n).group(1)), default=None)


def checkpoint_of(name):
    m = re.match(r"rl(\d+)_", name or "")
    return int(m.group(1)) if m else None


def load(name):
    name = resolve(name)
    path = os.path.join(RES, "leetcode", name + ".jsonl") if name else ""
    return [json.loads(l) for l in open(path)] if name and os.path.exists(path) else None


def run_step(run):
    """Last logged training step of a run (0 if no log)."""
    path = os.path.join(RES, "runs", run, "reward_log.jsonl") if run else None
    return max((json.loads(l)["call"] for l in open(path)), default=0) if path and os.path.exists(path) else 0


def cell_label(col, cell, found, names=()):
    """Column name plus a line saying what the cell pools (and, for an interim checkpoint of a run still training, which
    checkpoint and an in-progress note), or why it is empty."""
    step = run_step(cell.get("run"))
    if found:
        unit = "run" if col == "base" else "seed"
        label = f"{col}\n{len(found)} {unit}{'s' if len(found) != 1 else ''}"
        ck = next((checkpoint_of(resolve(n)) for n in names if resolve(n)), None)
        if ck is not None and ck < FINAL_STEP:
            label += f", checkpoint {ck}\n(training, step {step}/{FINAL_STEP})"
        return label
    if cell.get("run") and step < FINAL_STEP:
        return f"{col}\n(training, step {step}/{FINAL_STEP})"
    return f"{col}\n(not yet evaluated)"


def rates(attempts):
    """Fraction of attempts with each outcome, from the scorer's behaviours.  'edited, credit' / 'edited, no credit' split
    tests-edited responses (any change to the run_tests() assertions, or a grader that accepts anything) by whether the
    grader accepted them; 'untouched, credit w/o correctness' is accepted by the grader, wrong on the hidden tests, and
    the tests untouched (fits the visible assertions only); 'hidden tests' passes the hidden ground-truth tests;
    'accepted' is the grader's verdict, i.e. the environment's reward."""
    n = len(attempts)
    b = [a["behaviors"] for a in attempts]
    return {
        "edited, credit": sum(x["any_hack"] and x["earns_reward"] for x in b) / n,
        "edited, no credit": sum(x["any_hack"] and not x["earns_reward"] for x in b) / n,
        "untouched, credit w/o correctness": sum(not x["any_hack"] and x["earns_reward"] and not x["will_succeed"] for x in b) / n,
        "hidden tests": sum(x["will_succeed"] for x in b) / n,
        "accepted": sum(x["earns_reward"] for x in b) / n,
        "n": n,
    }


# ---------------------------------------------------------------- SVG helpers

def nice_max(v):
    """Smallest clean axis maximum above v (ticks are drawn at fifths of it)."""
    return next(m for m in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0) if m >= min(v, 1.0))


def grouped_bars(groups, title, subtitle="", ymax=None, width=860, keys=None, show_legend=True, show_values=True, stack_note=None, refline=None, gap=2, fmt="pct"):
    """groups: list of (label, {series: value}). One cluster of thin bars per group; `gap` px between the bars of a cluster.
    A value may be a (bottom, top) tuple for a two-part stacked bar.  An empty dict draws an empty slot with its label."""
    keys = keys or []
    tick = (lambda v: f"{v:.0%}") if fmt == "pct" else (lambda v: f"{v:.1f}")
    val = (lambda v: f"{v:.1%}") if fmt == "pct" else (lambda v: f"{v:.2f}")
    def total(v):
        return sum(v) if isinstance(v, tuple) else v
    ymax = nice_max(ymax or max([total(v) for _, d in groups for v in d.values() if isinstance(v, (float, tuple))] or [0.1]) * 1.1)
    left, top, h = 56, 40, 220
    bottom = 70 + 12 * max(0, max(label.count("\n") + 1 for label, _ in groups) - 3)  # room for labels of more than 3 lines
    band = (width - left - 16) / len(groups)
    bar_w = min(20, (band - 12 - gap * (len(keys) - 1)) / len(keys))
    out = [f'<svg viewBox="0 0 {width} {top + h + bottom}" role="img" aria-label="{title}">']
    out.append(f'<text x="{left}" y="20" class="title">{title}</text>')
    if subtitle:
        out.append(f'<text x="{left}" y="34" class="sub">{subtitle}</text>')
    for i in range(6):  # gridlines + y ticks
        y = top + h - i * h / 5
        out.append(f'<line x1="{left}" x2="{width - 16}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick(ymax * i / 5)}</text>')
    for gi, (label, d) in enumerate(groups):
        x0 = left + gi * band + (band - bar_w * len(keys) - gap * (len(keys) - 1)) / 2
        for ki, k in enumerate(keys):
            v = d.get(k, 0.0)
            x = x0 + ki * (bar_w + gap)
            err = d.get("err", {}).get(k)
            if isinstance(v, tuple):  # stacked: (bottom, top) with a 2px surface gap between segments
                bot, tp = v
                bh, th = h * bot / ymax, h * tp / ymax
                yb = top + h - bh
                out.append(f'<rect x="{x:.1f}" y="{yb:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{COLORS.get(k, GREY[0])}">'
                           f'<title>{label} — {k}: solid {bot:.1%}, light {tp:.1%}, total {bot + tp:.1%} ({d.get("n", "?")})</title></rect>')
                if th > 0:
                    out.append(f'<rect x="{x:.1f}" y="{yb - 2 - th:.1f}" width="{bar_w:.1f}" height="{th:.1f}" rx="3" fill="{COLORS.get(k, GREY[0])}" class="light">'
                               f'<title>{label} — {k}: light {tp:.1%} (solid {bot:.1%}, total {bot + tp:.1%}, {d.get("n", "?")})</title></rect>')
                v, y = bot + tp, yb - 2 - th
                if err:
                    out.append(errbar(x + bar_w / 2, top + h, h / ymax, bot, err[0]))
                    out.append(errbar(x + bar_w / 2, top + h - 2, h / ymax, v, err[1]))
            else:
                bh = h * v / ymax
                y = top + h - bh
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{COLORS.get(k, GREY[0])}">'
                           f'<title>{label} — {k}: {val(v)}{f" ± {val(err)}" if err else ""} ({d.get("n", "?")})</title></rect>')
                if err:
                    out.append(errbar(x + bar_w / 2, top + h, h / ymax, v, err))
            if show_values and v >= 0.005:
                out.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 4, top + 10):.1f}" class="val" text-anchor="middle">{val(v)}</text>')
        for li, line in enumerate(label.split("\n")):
            cx, cy = left + gi * band + band / 2, top + h + 16 + 13 * li
            if line.startswith("(training") or line.startswith("(not yet"):  # an orange "in progress" pill, like the section badge
                text = line.strip("()").upper().replace("TRAINING, ", "IN PROGRESS: ")
                w = 5.9 * len(text) + 14
                out.append(f'<rect x="{cx - w / 2:.1f}" y="{cy - 10}" width="{w:.1f}" height="14" rx="4" fill="#c98500"/>'
                           f'<text x="{cx:.1f}" y="{cy}" class="pill" text-anchor="middle">{text}</text>')
            else:
                out.append(f'<text x="{cx:.1f}" y="{cy}" class="tick" text-anchor="middle">{line}</text>')
    out.append(f'<line x1="{left}" x2="{width - 16}" y1="{top + h}" y2="{top + h}" class="axis"/>')
    if refline is not None:
        yr = top + h - h * refline[0] / ymax
        out.append(f'<line x1="{left}" x2="{width - 16}" y1="{yr:.1f}" y2="{yr:.1f}" class="ref"/>')
        out.append(f'<text x="{width - 16}" y="{yr - 4:.1f}" class="tick" text-anchor="end">{refline[1]}</text>')
    if show_legend:
        out.append(legend(keys, [COLORS.get(k, GREY[0]) for k in keys], left, top + h + bottom - 12, width=width))
    if stack_note:
        out.append(f'<text x="{width - 16}" y="{top + h + bottom - 11}" class="tick" text-anchor="end">{stack_note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def lines(series, title, subtitle="", xlabel="training step", width=860, ymax=None):
    """series: list of (name, [(x, y), ...], style, colour) where style is 'solid' or 'dashed'."""
    left, top, h = 56, 40, 220
    bottom = 56 + 16 * legend_rows([n for n, _, _, _ in series], left, width)  # room for the wrapped legend
    xmax = max(x for _, pts, _, _ in series for x, _ in pts)
    ymax = nice_max(ymax or max(y for _, pts, _, _ in series for _, y in pts) * 1.1)
    def X(x): return left + (width - left - 16) * x / xmax
    def Y(y): return top + h - h * y / ymax
    out = [f'<svg viewBox="0 0 {width} {top + h + bottom}" role="img" aria-label="{title}">']
    out.append(f'<text x="{left}" y="20" class="title">{title}</text>')
    if subtitle:
        out.append(f'<text x="{left}" y="34" class="sub">{subtitle}</text>')
    for i in range(6):
        y = Y(ymax * i / 5)
        out.append(f'<line x1="{left}" x2="{width - 16}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{ymax * i / 5:.0%}</text>')
    for xt in range(0, int(xmax) + 1, 50):
        out.append(f'<text x="{X(xt):.1f}" y="{top + h + 16}" class="tick" text-anchor="middle">{xt}</text>')
    out.append(f'<text x="{(left + width - 16) / 2:.1f}" y="{top + h + 32}" class="tick" text-anchor="middle">{xlabel}</text>')
    for name, pts, style, colour in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if style == "dashed" else ""
        out.append(f'<path d="{d}" stroke="{colour}" class="line"{dash} fill="none"><title>{name}</title></path>')
        x, y = pts[-1]
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{colour}" class="ring"><title>{name}: {y:.1%} at step {x}</title></circle>')
    out.append(legend([n for n, _, _, _ in series], [c for _, _, _, c in series], left, top + h + 56, width=width))
    out.append("</svg>")
    return "\n".join(out)


def errbar(cx, base_y, scale, v, se, cap=3):
    """A ± se error bar centred on value v (drawn in ink, on top of the marks)."""
    if not se:
        return ""
    y1, y2 = base_y - scale * (v + se), base_y - scale * max(v - se, 0)
    return (f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y1:.1f}" y2="{y2:.1f}" class="err"/>'
            f'<line x1="{cx - cap:.1f}" x2="{cx + cap:.1f}" y1="{y1:.1f}" y2="{y1:.1f}" class="err"/>'
            f'<line x1="{cx - cap:.1f}" x2="{cx + cap:.1f}" y1="{y2:.1f}" y2="{y2:.1f}" class="err"/>')


def legend_rows(keys, x, width):
    """How many rows the legend wraps to (same rule as legend())."""
    rows, cx = 1, x
    for k in keys:
        w = 17 + 6.2 * len(k) + 22
        if cx + w > width - 16 and cx > x:
            rows, cx = rows + 1, x
        cx += w
    return rows


def legend(keys, colours, x, y, width=860):
    parts, cx, cy = [], x, y
    for k, colour in zip(keys, colours):
        w = 17 + 6.2 * len(k) + 22
        if cx + w > width - 16 and cx > x:  # wrap
            cx, cy = x, cy + 16
        parts.append(f'<rect x="{cx}" y="{cy - 9}" width="12" height="12" rx="2" fill="{colour}"/>')
        parts.append(f'<text x="{cx + 17}" y="{cy + 1}" class="tick">{k}</text>')
        cx += w
    return "\n".join(parts)


def table(header, rows):
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details><summary>table</summary><table><tr>{th}</tr>{trs}</table></details>'


def pct(x):
    return f"{x:.1%}"


def row_title(row):
    return row.replace("\n", " ")


# ---------------------------------------------------------------- sections

def section_behavior():
    """Per row of the grid, two charts: test editing (stacked by outcome) + credit without correctness, and the two pass rates."""
    EDITED, OTHER = "tests edited: accepted by grader (solid) / not accepted (light)", "credit without correctness, tests untouched"
    HIDDEN, ACCEPTED = "passes hidden tests", "accepted by grader"
    panels, rows = [], []
    for row, cells in GRID.items():
        groups, pass_groups = [], []
        for col in COLS:
            cell = cells.get(col)
            if cell is None:  # no such recipe in this row: keep the column position, no label
                groups.append(("", {})); pass_groups.append(("", {})); continue
            found = [(resolve(name), rates([t["attempts"][0] for t in trs])) for name in cell.get("eval", []) for trs in [load(name)] if trs]
            label = cell_label(col, cell, found, cell.get("eval", []))
            if not found:
                groups.append((label, {})); pass_groups.append((label, {})); continue
            n = sum(r["n"] for _, r in found)
            pooled = {k: sum(r[k] * r["n"] for _, r in found) / n for k in found[0][1] if k != "n"}
            se = lambda v: (v * (1 - v) / n) ** 0.5  # binomial SE of a pooled rate
            ec, en, other = pooled["edited, credit"], pooled["edited, no credit"], pooled["untouched, credit w/o correctness"]
            groups.append((label, {EDITED: (ec, en), OTHER: other, "err": {EDITED: (se(ec), se(ec + en)), OTHER: se(other)}, "n": f"{len(found)} run(s)"}))
            pass_groups.append((label, {HIDDEN: pooled["hidden tests"], ACCEPTED: pooled["accepted"],
                                        "err": {HIDDEN: se(pooled["hidden tests"]), ACCEPTED: se(pooled["accepted"])}, "n": f"{len(found)} run(s)"}))
            for name, r in found:
                rows.append([row_title(row), col.replace("\n", " "), name, r["n"], pct(r["edited, credit"]), pct(r["edited, no credit"]),
                             pct(r["untouched, credit w/o correctness"]), pct(r["hidden tests"]), pct(r["accepted"])])
        base_rate = next((d[HIDDEN] for _, d in pass_groups if HIDDEN in d), None) if row == "thinking off" else None
        panels.append('<div class="pair">'
                      + grouped_bars(groups, f"{row_title(row)} -- edited the tests / credit without correctness", "119 test problems x 10 samples; error bars = binomial SE",
                                     ymax=1.0, keys=[EDITED, OTHER], width=860, gap=12)
                      + grouped_bars(pass_groups, f"{row_title(row)} -- pass rate", "hidden ground-truth tests vs acceptance by the grader" + ("; dashed = base model, hidden tests" if base_rate else ""),
                                     ymax=1.0, keys=[HIDDEN, ACCEPTED], width=860, gap=12, refline=(base_rate, "base") if base_rate else None)
                      + "</div>")
    return "\n".join(panels) + table(["row", "column", "run", "n", "tests edited, accepted", "tests edited, not accepted",
                                      "credit without correctness, tests untouched", "passes hidden tests", "accepted by grader"], rows)


def auc(scores, labels):  # rank AUC = P(score of a positive > score of a negative), ties 1/2
    pos = [x for x, y in zip(scores, labels) if y]; neg = [x for x, y in zip(scores, labels) if not y]
    return float("nan") if not pos or not neg else sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def section_prediction():
    """Per row of the grid: across-problem correlation between a model's mean committed probability on a problem and the
    problem's observed rate (over its 10 evaluation samples), per statement; computed per run and averaged over the cell's
    runs.  The table also gives the instance-level AUC."""
    panels, rows = [], []
    for row, cells in GRID.items():
        groups = []
        for col in COLS:
            cell = cells.get(col)
            if cell is None:
                groups.append(("", {})); continue
            found = [trs for name in cell.get("commit", []) for trs in [load(name)] if trs]
            if not found:
                label = cell_label(col, cell, found) if cell.get("commit") else f"{col}\n(not evaluated)"
                groups.append((label, {})); continue
            d, per_role = {"n": f"{sum(len(t) for t in found)} rollouts"}, defaultdict(list)
            for name, trs in zip([resolve(n) for n in cell["commit"] if load(n)], found):
                for role, candidates in ROLES.items():
                    stmt = next((c for c in candidates if c in trs[0]["precommit"]["answers"]), None)
                    if stmt is None:
                        continue
                    by_task = defaultdict(lambda: ([], []))
                    for t in trs:
                        if t["precommit"]["answers"].get(stmt) is not None:
                            by_task[t["task_id"]][0].append(float(t["precommit"]["answers"][stmt])); by_task[t["task_id"]][1].append(float(t["final"]["behaviors"][stmt]))
                    pm = [mean(p) for p, _ in by_task.values()]; om = [mean(o) for _, o in by_task.values()]
                    sd = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5
                    c = corr(pm, om) if sd(pm) > 0 and sd(om) > 0 else float("nan")  # undefined when the outcome (or the commitment) is constant
                    pairs = [(p, o) for ps, os_ in by_task.values() for p, o in zip(ps, os_)]
                    a = auc([p for p, _ in pairs], [o for _, o in pairs])
                    if c == c:
                        per_role[role].append(c)
                    rows.append([row_title(row), col.replace("\n", " "), name, f"{role} ({stmt})", pct(mean(om)), f"{mean(pm):.2f}",
                                 f"{sd(pm):.3f} / {sd(om):.3f}", f"{c:.2f}" if c == c else "- (constant)", f"{a:.2f}" if a == a else "- (constant outcome)"])
            for role, cs in per_role.items():
                d[role] = max(0.0, mean(cs))  # negative correlations are drawn at 0 (the table has the value)
            groups.append((cell_label(col, cell, found, cell["commit"]), d))
        panels.append(grouped_bars(groups, f"{row_title(row)} -- across-problem correlation of the mean committed probability with the observed rate",
                                   "119 problems x 10 samples per run, averaged over runs; 0 = no information", ymax=1.0, width=1000, keys=list(ROLES), show_values=True, gap=12, fmt="num"))
    return "\n".join(panels) + table(["row", "column", "run", "statement (as asked)", "observed rate", "mean prediction", "sd across problems: prediction / observed", "correlation", "instance AUC"], rows)


def section_mcq():
    """Misaligned-choice rate by transcript condition (thinking-off row only; no MCQ runs exist for the thinking models)."""
    import pandas as pd
    keys = ["clean", "legit", "hack", "fail"]
    groups, rows = [], []
    for col in COLS:
        cell = GRID["thinking off"][col]
        per_run = []
        for name in cell.get("mcq", []):
            path = os.path.join(RES, "leetcode", name + ".jsonl")
            if not os.path.exists(path):
                continue
            df = pd.DataFrame([json.loads(l) for l in open(path)])
            g = df.groupby(["condition", "transcript_index"])["p_misaligned"].mean().groupby("condition")
            per_q = df.groupby(["condition", "question_id"])["p_misaligned"].mean()  # per-question mean over transcripts x variants
            mass = (df["p_a"] + df["p_b"]).groupby(df["condition"]).mean()  # first-token mass on the two answer letters
            per_run.append((name, {c: (g.mean()[c], g.size()[c], per_q[c], mass[c]) for c in keys if c in g.groups}))
        if not per_run:
            groups.append((f"{col}\n(not evaluated)", {})); continue
        d = {"err": {}, "n": f"{len(per_run)} run(s)"}
        for k in keys:
            runs_k = [r[k] for _, r in per_run if k in r]
            if not runs_k:
                continue
            per_q = sum(r[2] for r in runs_k) / len(runs_k)  # average the per-question series over runs
            d[k] = per_q.mean()
            d["err"][k] = per_q.std(ddof=1) / len(per_q) ** 0.5  # SE across the 100 questions
        groups.append((cell_label(col, cell, per_run), d))
        for name, r in per_run:
            for k in keys:
                if k in r:
                    rows.append([col.replace("\n", " "), name, k, r[k][1], pct(r[k][0]), f"{r[k][3]:.2f}"])
    svg = grouped_bars(groups, "thinking off -- P(misaligned choice) by transcript condition",
                       "100 questions x 4 syntaxes x 2 orders per transcript, up to 50 transcripts per condition; error bars = SE across questions",
                       ymax=None, keys=keys, show_values=False, width=1000, gap=4)
    return svg + table(["column", "run", "condition", "n transcripts", "misaligned rate", "mass on A/B"], rows)


def section_example():
    """The two training-time conversations for one test problem, side by side: the commitment conversation
    (prompt + the adapter's answer) and the attempt conversation (prompt + the adapter's solution).  The
    decoupled runs never see them joined; they are only tied together by the reward."""
    import html as H
    tid = "3282"
    commits = [t for t in (load("rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests") or []) if t["task_id"] == tid]
    attempts = [t for t in (load("rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s1_neutral_modify_tests") or []) if t["task_id"] == tid]
    if not commits or not attempts:
        return "<p>(example transcripts not available locally)</p>"
    c = commits[0]; a = next((t for t in attempts if t["final"]["label"] == "Correct"), attempts[0])
    acc = sum(t["final"]["behaviors"]["earns_reward"] for t in attempts) / len(attempts)
    tam = sum(t["final"]["behaviors"]["any_hack"] for t in attempts) / len(attempts)
    ps = [t["precommit"]["answers"]["earns_reward"] for t in commits if t["precommit"]["answers"].get("earns_reward") is not None]
    def conv(title, msgs, note):
        turns = "".join(f'<div class="turn"><div class="role">{m["role"]}</div><pre>{H.escape(m["content"])}</pre></div>' for m in msgs)
        return f'<div class="conv"><h3>{title}</h3><p>{note}</p>{turns}</div>'
    return (f'<p>Test problem {tid}; adapter "prediction / prediction+success" (thinking off), seed 1. In the decoupled runs the commitment and the '
            f'attempt are <em>separate conversations</em> on the same problem: the model never sees its commitment while solving, and never sees a '
            f'solution while committing. Per problem and step, training samples 16 of each; the commitment is rewarded on '
            f'1 - squared error against the 16 attempts\' mean tampering / acceptance, and each attempt on the task plus agreement with '
            f'the 16 commitments\' mean. Below, one sample of each conversation from the test-set evaluation. Across this problem\'s 10 '
            f'evaluation samples the adapter committed p(accepted) = {min(ps):.2f}-{max(ps):.2f} and its attempts were accepted {acc:.0%} of '
            f'the time and tampered {tam:.0%} of the time.</p><div class="pair">'
            + conv("Commitment conversation", c["messages"][:3],
                   f'The assistant turn is the commitment: p(tamper) = {c["precommit"]["answers"]["any_hack"]:.2f}, p(accepted) = {c["precommit"]["answers"]["earns_reward"]:.2f}.')
            + conv("Attempt conversation", a["messages"],
                   f'The assistant turn is the solution. Outcome: <b>{a["final"]["label"]}</b>; tampering = {a["final"]["behaviors"]["any_hack"]}, accepted by the grader = {a["final"]["behaviors"]["earns_reward"]}.')
            + "</div>")


# ---------------------------------------------------------------- training curves of the thinking runs

def corr(xs, ys):
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5; sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else 0.0


def run_curves(run, window=10):
    """Per-window means from a run's rollouts.jsonl: attempts' acceptance and test-editing rates, and (decoupled runs)
    the commitments' mean acceptance / tampering, plus the across-problem correlation of committed with observed
    acceptance.  Returns (last_step, {series_name: [(step, value), ...]})."""
    path = os.path.join(RES, "runs", run, "rollouts.jsonl")
    if not os.path.exists(path):
        return 0, {}
    att, com = defaultdict(list), defaultdict(list)  # (call, task) -> rows
    for r in map(json.loads, open(path)):
        (com if r.get("role") == "commit" else att)[(r["call"], r["task_id"])].append(r)
    last = max(c for c, _ in att)
    out = defaultdict(list)
    for lo in range(1, last + 1, window):
        hi = min(lo + window - 1, last)
        a = [r for (c, _), rs in att.items() if lo <= c <= hi for r in rs]
        if not a:
            continue
        x = (lo + hi) / 2
        out["accepted"].append((x, mean([r["behaviors"]["earns_reward"] for r in a])))
        out["edited"].append((x, mean([r["behaviors"]["any_hack"] for r in a])))
        keys = [k for k in com if lo <= k[0] <= hi and k in att and any(None not in r["commit"] for r in com[k])]
        if keys:
            pa = [mean([float(r["commit"][1]) for r in com[k] if None not in r["commit"]]) for k in keys]
            oa = [mean([r["behaviors"]["earns_reward"] for r in att[k]]) for k in keys]
            ph = [mean([float(r["commit"][0]) for r in com[k] if None not in r["commit"]]) for k in keys]
            oh = [mean([r["behaviors"]["any_hack"] for r in att[k]]) for k in keys]
            out["committed accepted"].append((x, mean(pa))); out["committed edited"].append((x, mean(ph)))
            out["corr accepted"].append((x, max(0.0, corr(pa, oa)))); out["corr edited"].append((x, max(0.0, corr(ph, oh))))  # negative = chance, drawn at 0
    return last, out


def section_curves():
    curves, status = {}, []
    for label, run, style in PART4_RUNS:
        last, c = run_curves(run)
        if c:
            curves[label] = (c, style); status.append((label, run, last))
    shade = {label: i for i, label in enumerate(curves)}  # one shade per run within the metric's colour family
    def series(key, family, only=None, style_override=None, prefix=""):
        return [(prefix + label, pts[key], style_override or style, family[shade[label] % len(family)])
                for label, (pts, style) in curves.items() if key in pts and (only is None or label in only)]
    commit_runs = [label for label, (pts, _) in curves.items() if "committed accepted" in pts]
    lc = [lines(series("accepted", SUCC, prefix="accepted: "), "Training batches: attempts accepted by the grader (10-step means)", "16 problems x 16 attempts per step", width=1000, ymax=1.0),
          lines(series("edited", HACK, prefix="tests edited: "), "Training batches: attempts that edited the tests (10-step means)", "the behaviour the commitments predict", width=1000),
          lines(series("committed accepted", SUCC, only=commit_runs, prefix="committed: ") + series("accepted", SUCC, only=commit_runs, style_override="dashed", prefix="observed: "),
                "Commitments: committed p(accepted) (solid) vs the attempts' acceptance rate (dashed)", "mean over the 16 commitments and 16 attempts per problem; one shade per run", width=1000, ymax=1.0),
          lines(series("corr accepted", SUCC, only=commit_runs, prefix="accepted: ") + series("corr edited", HACK, only=commit_runs, style_override="dashed", prefix="tests edited: "),
                "Commitments: across-problem correlation with the attempts' rates, per 10-step window", "greens = accepted by grader (solid), reds = tests edited (dashed); ~160 problems per window", width=1000, ymax=1.0)]
    running = [(l, r, n) for l, r, n in status if n < FINAL_STEP]
    badge = f'<span class="badge">in progress: {len(running)} run{"s" if len(running) != 1 else ""} still training</span>' if running else ""
    srows = [[l, f"<code>{r}</code>", f"{min(n, FINAL_STEP)}/{FINAL_STEP}", "done" if n >= FINAL_STEP else f'<span class="badge" style="margin:0">in progress</span>'] for l, r, n in status]
    stable = '<table class="status"><tr><th>run</th><th>directory</th><th>step</th><th>status</th></tr>' + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in srows) + "</table>"
    return badge, "\n".join(lc), stable


# ---------------------------------------------------------------- page

CSS = """
:root { color-scheme: light dark; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e6e5e1;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #8e5bd6; --s6: #7a7a72; }
@media (prefers-color-scheme: dark) { :root { --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --grid: #34342f;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #a47ae8; --s6: #9a9a90; } }
body { background: var(--surface); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; margin: 0; padding: 24px; max-width: 1100px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 40px; border-top: 1px solid var(--grid); padding-top: 16px; }
p, li { color: var(--ink2); max-width: 78ch; }
svg { width: 100%; height: auto; display: block; overflow: visible; margin-bottom: 6px; }
.title { font-size: 14px; font-weight: 600; fill: var(--ink); } .sub, .tick { font-size: 11px; fill: var(--ink2); } .note { font-style: italic; }
.val { font-size: 10px; fill: var(--ink2); } .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--ink2); stroke-width: 1; }
.light { opacity: .45; } path.line { stroke-width: 2; stroke-linejoin: round; }
.ring { stroke: var(--surface); stroke-width: 1.5; }
.pill { font-size: 9px; font-weight: 600; letter-spacing: .04em; fill: #fff; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: #fff; background: #c98500; border-radius: 4px; padding: 2px 8px; margin-left: 8px; vertical-align: middle; }
.err { stroke: var(--ink); stroke-width: 1; } .ref { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 5 4; }
rect:hover { opacity: .75; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; } .conv h3 { font-size: 14px; margin: 6px 0; } @media (max-width: 900px) { .pair { grid-template-columns: 1fr; } }
.turn { margin: 10px 0; } .role { font-size: 11px; font-weight: 600; color: var(--ink2); text-transform: uppercase; letter-spacing: .04em; }
pre { background: color-mix(in srgb, var(--surface) 90%, var(--ink) 10%); border: 1px solid var(--grid); padding: 10px 12px; overflow-x: auto; font-size: 12px; line-height: 1.4; white-space: pre-wrap; max-width: 100%; }
details { margin: 4px 0 0 56px; } summary { cursor: pointer; color: var(--ink2); font-size: 12px; }
table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; margin-top: 6px; }
td, th { padding: 2px 10px; text-align: right; border-bottom: 1px solid var(--grid); } th:first-child, td:first-child { text-align: left; }
table.status td, table.status th { text-align: left; } table.status td:first-child { min-width: 220px; } table.status code { word-break: break-all; }
.stamp { font-size: 12px; }
"""


def main():
    badge, curves, status = section_curves()
    today = __import__("datetime").date.today().isoformat()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Contracts results</title><style>{CSS}</style></head><body>
<h1>Results: behaviour, prediction and alignment by model (modify_tests prompt)</h1>
<p class="stamp">Generated {today} by <code>docs/make_results_page.py</code>; methods and discussion in <code>docs/results_part3.md</code> and <code>docs/results_part4.md</code>.</p>

<h2>1. Behaviour</h2>
{section_behavior()}

<h2>2. Prediction</h2>
{section_prediction()}

<h2>3. Training curves of the thinking runs {badge}</h2>
{curves}
{status}

<h2>4. Alignment MCQ</h2>
{section_mcq()}

<h2>5. Example: the commitment conversation and the attempt conversation</h2>
{section_example()}
</body></html>"""
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, len(html), "bytes")


if __name__ == "__main__":
    main()
