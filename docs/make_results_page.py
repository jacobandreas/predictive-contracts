"""Build docs/results_part1.html: charts of hack rates, alignment MCQ and commitment AUC for the seeded comparison.

    python docs/make_results_page.py

Reads results/leetcode/*.jsonl (evaluation transcripts) and results/runs/*/reward_log.jsonl
(GRPO training logs) and writes a self-contained HTML page whose sections mirror
docs/results_part1.md.  Charts are inline SVG; each has a legend, hover tooltips and a table.
"""
import glob
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "results_part1.html")

# Categorical slots 1-4 of the reference palette (validated for adjacent pairs, light + dark).
PALETTE = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]
SERIES = {"Correct": PALETTE[0], "Reward Hack": PALETTE[1], "Attempted hack": PALETTE[2]}

# Which evaluation files make up each (prompt, model) cell.  RL cells average over seeds.
RUNS = {
    ("modify_tests", "base"): ["base_neutral_s1_modify_tests", "base_neutral_s2_modify_tests", "base_neutral_s3_modify_tests"],
    ("modify_tests", "RL ⚠"): ["rl200_modify_tests_neutral_s1_neutral_modify_tests", "rl200_modify_tests_neutral_s2_neutral_modify_tests", "rl200_modify_tests_neutral_s3_neutral_modify_tests"],
    ("modify_tests", "prediction /\nsuccess;\ngroup"): ["rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s1_neutral_modify_tests", "rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s2_neutral_modify_tests", "rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s3_neutral_modify_tests"],
    ("modify_tests", "prediction /\nsuccess;\nbatch"): ["rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s1_neutral_modify_tests", "rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s2_neutral_modify_tests", "rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s3_neutral_modify_tests"],
    ("modify_tests", "prediction /\nprediction+success;\ngroup"): ["rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nprediction+success;\nbatch"): ["rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nprediction;\nbatch"): ["rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
}
MCQ = {
    ("modify_tests", "base"): ["mcq_base_neutral_s1_modify_tests", "mcq_base_neutral_s2_modify_tests", "mcq_base_neutral_s3_modify_tests"],
    ("modify_tests", "RL ⚠"): ["mcq_rl200_modify_tests_neutral_s1_neutral_modify_tests", "mcq_rl200_modify_tests_neutral_s2_neutral_modify_tests", "mcq_rl200_modify_tests_neutral_s3_neutral_modify_tests"],
    ("modify_tests", "prediction /\nsuccess;\ngroup"): ["mcq_rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s1_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s2_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nsuccess;\nbatch"): ["mcq_rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s1_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s2_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nprediction+success;\ngroup"): ["mcq_rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nprediction+success;\nbatch"): ["mcq_rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
    ("modify_tests", "prediction /\nprediction;\nbatch"): ["mcq_rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s1_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s2_pc_modify_tests", "mcq_rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s3_pc_modify_tests"],
}


# Commitment evaluations (modify_tests prompt, with the commitment turn): model label -> (files pooled, mode)
COMMIT = {
    "prediction /\nsuccess;\ngroup": (["rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_hacksucc_prob_sftwarm_s3_pc_modify_tests"], "prob"),
    "prediction /\nsuccess;\nbatch": (["rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s3_pc_modify_tests"], "prob"),
    "prediction /\nprediction+success;\ngroup": (["rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"], "prob"),
    "prediction /\nprediction+success;\nbatch": (["rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agree_hacksucc_prob_sftwarm_s3_pc_modify_tests"], "prob"),
    "prediction /\nprediction;\nbatch": (["rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s1_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s2_pc_modify_tests", "rl200_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s3_pc_modify_tests"], "prob"),
}
# The three statement roles that vary across rollouts, mapped to whichever statement each model was asked.
ROLES = {"tampering": ["any_hack", "modifies_run_tests"], "helper function": ["defines_helper_function"], "success": ["will_succeed", "earns_reward"]}


def load(name):
    path = os.path.join(RES, "leetcode", name + ".jsonl")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def rates(attempts):
    """Fraction of attempts with each outcome.  Label-based: 'Correct', 'Reward Hack', 'Attempted hack' (the two attempted
    labels pooled).  Behavior-based: 'any_hack' (the grader was tampered with: assertions changed or a grader that
    accepts anything), 'hidden tests' (the solution passes the hidden ground-truth tests, tampering or not) and
    'accepted' (the grader used to award credit passed, i.e. the environment's hackable reward)."""
    c = Counter(a["label"] for a in attempts)
    n = len(attempts)
    return {
        "Correct": c["Correct"] / n,
        "Reward Hack": c["Reward Hack"] / n,
        "Attempted hack": (c["Attempted Reward Hack"] + c["Correct; Attempted Reward Hack"]) / n,
        "any_hack": sum(a["behaviors"]["any_hack"] for a in attempts) / n,
        "hidden tests": sum(a["behaviors"]["will_succeed"] for a in attempts) / n,
        "accepted": sum(a["behaviors"]["earns_reward"] for a in attempts) / n,
        "n": n,
    }


# ---------------------------------------------------------------- SVG helpers

def nice_max(v):
    """Smallest clean axis maximum above v (ticks are drawn at fifths of it)."""
    return next(m for m in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0) if m >= min(v, 1.0))


def grouped_bars(groups, title, subtitle="", ymax=None, width=860, keys=None, show_legend=True, show_values=True, stack_note=None, refline=None, gap=2):
    """groups: list of (label, {series: value}). One cluster of thin bars per group; `gap` px between the bars of a cluster
    (wider when value labels are printed over two or more bars, so the labels do not collide)."""
    keys = keys or list(SERIES)
    def total(v):
        return sum(v) if isinstance(v, tuple) else v
    ymax = nice_max(ymax or max(total(v) for _, d in groups for v in d.values() if isinstance(v, (float, tuple))) * 1.1)
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
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{ymax * i / 5:.0%}</text>')
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
                out.append(f'<rect x="{x:.1f}" y="{yb:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" class="s{ki + 1}">'
                           f'<title>{label} — {k}: successful hacks {bot:.1%}, failed attempts {tp:.1%} ({d.get("n", "?")})</title></rect>')
                if th > 0:
                    out.append(f'<rect x="{x:.1f}" y="{yb - 2 - th:.1f}" width="{bar_w:.1f}" height="{th:.1f}" rx="3" class="s{ki + 1} light">'
                               f'<title>{label} — {k}: failed attempts {tp:.1%} (successful {bot:.1%}, {d.get("n", "?")})</title></rect>')
                v, y = bot + tp, yb - 2 - th
                if err:
                    out.append(errbar(x + bar_w / 2, top + h, h / ymax, bot, err[0]))
                    out.append(errbar(x + bar_w / 2, top + h - 2, h / ymax, v, err[1]))
            else:
                bh = h * v / ymax
                y = top + h - bh
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" class="s{ki + 1}">'
                           f'<title>{label} — {k}: {v:.1%}{f" ± {err:.1%}" if err else ""} ({d.get("n", "?")})</title></rect>')
                if err:
                    out.append(errbar(x + bar_w / 2, top + h, h / ymax, v, err))
            if show_values and v >= 0.005:
                out.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 4, top + 10):.1f}" class="val" text-anchor="middle">{v:.1%}</text>')
        for li, line in enumerate(label.split("\n")):
            out.append(f'<text x="{left + gi * band + band / 2:.1f}" y="{top + h + 16 + 13 * li}" class="tick" text-anchor="middle">{line}</text>')
    out.append(f'<line x1="{left}" x2="{width - 16}" y1="{top + h}" y2="{top + h}" class="axis"/>')
    if refline is not None:
        yr = top + h - h * refline[0] / ymax
        out.append(f'<line x1="{left}" x2="{width - 16}" y1="{yr:.1f}" y2="{yr:.1f}" class="ref"/>')
        out.append(f'<text x="{width - 16}" y="{yr - 4:.1f}" class="tick" text-anchor="end">{refline[1]}</text>')
    if show_legend:
        out.append(legend(keys, left, top + h + bottom - 12))
    if stack_note:
        out.append(f'<text x="{width - 16}" y="{top + h + bottom - 11}" class="tick" text-anchor="end">{stack_note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def lines(series, title, subtitle="", xlabel="training step", width=860, ymax=None):
    """series: list of (name, [(x, y), ...], style) where style is 'solid' or 'dashed'."""
    left, top, h, bottom = 56, 40, 220, 70
    xmax = max(x for _, pts, _ in series for x, _ in pts)
    ymax = nice_max(ymax or max(y for _, pts, _ in series for _, y in pts) * 1.1)
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
    for xt in range(0, xmax + 1, 50):
        out.append(f'<text x="{X(xt):.1f}" y="{top + h + 16}" class="tick" text-anchor="middle">{xt}</text>')
    out.append(f'<text x="{(left + width - 16) / 2:.1f}" y="{top + h + 32}" class="tick" text-anchor="middle">{xlabel}</text>')
    keys = [name for name, _, _ in series]
    for si, (name, pts, style) in enumerate(series):
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if style == "dashed" else ""
        out.append(f'<path d="{d}" class="l{si + 1}"{dash} fill="none"><title>{name}</title></path>')
        x, y = pts[-1]
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="s{si + 1} ring"><title>{name}: {y:.1%} at step {x}</title></circle>')
    out.append(legend(keys, left, top + h + bottom - 12, dashed_from=None))
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


def legend(keys, x, y, dashed_from=None):
    parts, cx = [], x
    for i, k in enumerate(keys):
        parts.append(f'<rect x="{cx}" y="{y - 9}" width="12" height="12" rx="2" class="s{i + 1}"/>')
        parts.append(f'<text x="{cx + 17}" y="{y + 1}" class="tick">{k}</text>')
        cx += 17 + 7 * len(k) + 22
    return "\n".join(parts)


def table(header, rows):
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details><summary>table</summary><table><tr>{th}</tr>{trs}</table></details>'


def pct(x):
    return f"{x:.1%}"


# ---------------------------------------------------------------- sections

def mean(xs):
    return sum(xs) / len(xs)


def seeds_line(model, k):
    """Third label line under a cluster: how many training seeds (RL cells) or sampling runs (base cells) it pools."""
    return f"{k} run{'s' if k != 1 else ''}" if model.startswith("base") else f"{k} seed{'s' if k != 1 else ''}"


def section_hacking():
    """Hack and pass rates for each (prompt, base/RL) cell, pooled over runs (single-attempt protocol)."""
    LABEL, BEHAV = "label: Reward Hack (solid) + attempted (light)", "behavior: any_hack (grader tampered)"
    HIDDEN, ACCEPTED = "passes hidden tests", "passes test cases (accepted by the grader)"
    groups, pass_groups, rows = [], [], []
    for (hint, model), names in RUNS.items():
        per_run = [(name, rates([t["attempts"][0] for t in trs])) for name in names for trs in [load(name)] if trs]
        if not per_run:
            continue
        n = sum(r["n"] for _, r in per_run)
        pooled = {k: sum(r[k] * r["n"] for _, r in per_run) / n for k in per_run[0][1] if k != "n"}
        se = lambda v: (v * (1 - v) / n) ** 0.5  # binomial SE of a pooled rate
        label = f"{hint}\n{model}\n{seeds_line(model, len(per_run))}"
        hack, att = pooled["Reward Hack"], pooled["Attempted hack"]
        groups.append((label, {LABEL: (hack, att), BEHAV: pooled["any_hack"],
                               "err": {LABEL: (se(hack), se(hack + att)), BEHAV: se(pooled["any_hack"])}, "n": f"{len(per_run)} run(s)"}))
        pass_groups.append((label, {HIDDEN: pooled["hidden tests"], ACCEPTED: pooled["accepted"],
                                    "err": {HIDDEN: se(pooled["hidden tests"]), ACCEPTED: se(pooled["accepted"])}, "n": f"{len(per_run)} run(s)"}))
        for name, r in per_run:
            rows.append([hint, model, name, r["n"], pct(r["Reward Hack"]), pct(r["Attempted hack"]), pct(r["any_hack"]),
                         pct(r["Correct"]), pct(r["hidden tests"]), pct(r["accepted"])])
    panels = []
    for hint in ["modify_tests"]:  # one panel per prompt (only modify_tests is shown now)
        sub = [(label.split("\n", 1)[1], cell) for label, cell in groups if label.split("\n", 1)[0] == hint]
        if sub:
            panels.append(grouped_bars(sub, f"Prompt: {hint} -- hacking: the analysis label (stacked) and the any_hack behavior, single attempt",
                                       "119 test problems x 10 samples; RL cells pooled over seeds; error bars = binomial SE",
                                       ymax=None, keys=[LABEL, BEHAV], show_values=True, show_legend=True, width=1000, gap=16))
            # The take-off cells hack ~100% and flatten everything else; repeat the low cells on a 12% axis.
            low = [(label, cell) for label, cell in sub if max(sum(cell[LABEL]), cell[BEHAV]) <= 0.12]
            if len(low) < len(sub):
                panels.append(grouped_bars(low, f"Prompt: {hint} -- same data, cells under 12% only (zoom)",
                                           "cells above 12% omitted here; see the chart above",
                                           ymax=0.12, keys=[LABEL, BEHAV], show_values=True, show_legend=True, width=1000, gap=16))
        psub = [(label.split("\n", 1)[1], cell) for label, cell in pass_groups if label.split("\n", 1)[0] == hint]
        base_rate = next((cell[HIDDEN] for label, cell in psub if label.startswith("base")), None)
        panels.append(grouped_bars(psub, f"Prompt: {hint} -- pass rate: hidden ground-truth tests vs the grader's test cases, single attempt",
                                   "119 test problems x 10 samples; RL cells pooled over seeds; error bars = binomial SE; dashed line = base model, hidden tests",
                                   ymax=None, keys=[HIDDEN, ACCEPTED], show_values=True, show_legend=True, width=1000, gap=16,
                                   refline=(base_rate, "base model (hidden tests)") if base_rate is not None else None))
    return "\n".join(panels) + table(["prompt", "model", "run", "n", "Reward Hack", "Attempted hack", "any_hack", "Correct", "hidden tests", "accepted"], rows)


def section_mcq():
    """Misaligned-choice rate by transcript condition for each (prompt, base/RL) cell, averaged over runs."""
    import pandas as pd
    keys = ["clean", "legit", "hack", "fail"]
    groups, rows = [], []
    for (hint, model), names in MCQ.items():
        per_run = []
        for name in names:
            path = os.path.join(RES, "leetcode", name + ".jsonl")
            if not os.path.exists(path):
                continue
            df = pd.DataFrame([json.loads(l) for l in open(path)])
            g = df.groupby(["condition", "transcript_index"])["p_misaligned"].mean().groupby("condition")
            per_q = df.groupby(["condition", "question_id"])["p_misaligned"].mean()  # per-question mean over transcripts x variants
            per_run.append((name, {c: (g.mean()[c], g.size()[c], per_q[c]) for c in keys if c in g.groups}))
        if not per_run:
            continue
        cell, cell["err"] = {}, {}
        for k in keys:
            runs_k = [d[k] for _, d in per_run if k in d]
            if not runs_k:
                continue
            per_q = sum(r[2] for r in runs_k) / len(runs_k)  # average the per-question series over runs
            cell[k] = per_q.mean()
            cell["err"][k] = per_q.std(ddof=1) / len(per_q) ** 0.5  # SE across the 100 questions
        cell["n"] = f"{len(per_run)} run(s)"
        groups.append((f"{hint}\n{model}\n{seeds_line(model, len(per_run))}", cell))
        for name, d in per_run:
            for k in keys:
                if k in d:
                    rows.append([hint, model, name, k, d[k][1], pct(d[k][0])])
    panels = []
    for hint in ["modify_tests"]:
        sub = [(label.split("\n", 1)[1], cell) for label, cell in groups if label.split("\n", 1)[0] == hint]
        if sub:
            panels.append(grouped_bars(sub, f"Prompt: {hint} -- P(misaligned choice) by transcript condition",
                                       "100 questions x 4 syntaxes x 2 orders per transcript, up to 50 transcripts per condition; error bars = SE across questions",
                                       ymax=0.4, keys=keys, show_values=False, width=1000))
    return "\n".join(panels) + table(["prompt", "model", "run", "condition", "n transcripts", "misaligned rate"], rows)


def section_commitments():
    """Instance-level quality of the commitments: AUC per statement role, plus a full metrics table."""
    import numpy as np
    def auc(scores, labels):  # rank AUC = P(score of a positive > score of a negative), ties 1/2 (same as contract.analyze.auc)
        pos = [x for x, y in zip(scores, labels) if y]; neg = [x for x, y in zip(scores, labels) if not y]
        return float("nan") if not pos or not neg else sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    groups, rows = [], []
    for label, (names, mode) in COMMIT.items():
        found = [load(n) for n in names if load(n)]
        if not found:
            continue
        trs = [t for trs_ in found for t in trs_]  # pool the runs
        cell = {"n": f"{len(trs)} rollouts"}
        for role, candidates in ROLES.items():
            stmt = next((c for c in candidates if c in trs[0]["precommit"]["answers"]), None)
            if stmt is None:
                continue
            pairs = [(t["precommit"]["answers"][stmt], t["final"]["behaviors"][stmt]) for t in trs if t["precommit"]["answers"].get(stmt) is not None]
            pred = np.array([float(p) for p, _ in pairs]); act = np.array([float(a) for _, a in pairs])
            hard = pred >= 0.5
            tp, fp, fn = (hard & (act == 1)).sum(), (hard & (act == 0)).sum(), (~hard & (act == 1)).sum()
            a = auc(pred.tolist(), act.tolist())
            if a == a:  # not NaN
                cell[role] = float(a)
            rows.append([label.replace("\n", " "), f"{role} ({stmt})", pct(act.mean()), f"{pred.mean():.2f}",
                         pct((hard == (act == 1)).mean()) if mode == "yesno" else f"{((pred - act) ** 2).mean():.3f} / {act.var():.3f}",
                         f"{tp / (tp + fp):.2f}" if tp + fp else "-", f"{tp / (tp + fn):.2f}" if tp + fn else "-",
                         f"{2 * tp / (2 * tp + fp + fn):.2f}" if 2 * tp + fp + fn else "-", f"{a:.2f}" if a == a else "-"])
        groups.append((label + "\n" + seeds_line(label, len(found)), cell))
    svg = grouped_bars(groups, "Instance-level discrimination of the commitments (AUC) by statement",
                       "modify_tests test set with the commitment turn, 119 problems x 10 samples; 0.5 = no information (statement mapping in the table)",
                       ymax=1.0, width=1000, keys=list(ROLES), show_values=False, refline=(0.5, "chance"))
    return svg + table(["model", "statement (as asked)", "observed rate", "mean prediction", "accuracy | Brier / base-rate Brier", "precision", "recall", "F1", "AUC"], rows)


# ---------------------------------------------------------------- page

CSS = """
:root { color-scheme: light dark; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e6e5e1;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; }
@media (prefers-color-scheme: dark) { :root { --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --grid: #34342f;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; } }
body { background: var(--surface); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; margin: 0; padding: 24px; max-width: 1100px; }
h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 40px; border-top: 1px solid var(--grid); padding-top: 16px; }
p, li { color: var(--ink2); max-width: 78ch; }
svg { width: 100%; height: auto; display: block; overflow: visible; margin-bottom: 6px; }
.title { font-size: 14px; font-weight: 600; fill: var(--ink); } .sub, .tick { font-size: 11px; fill: var(--ink2); }
.val { font-size: 10px; fill: var(--ink2); } .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--ink2); stroke-width: 1; }
.s1 { fill: var(--s1); } .s2 { fill: var(--s2); } .s3 { fill: var(--s3); } .s4 { fill: var(--s4); } .light { opacity: .45; } .err { stroke: var(--ink); stroke-width: 1; } .ref { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 5 4; }
rect:hover { opacity: .75; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; } .conv h3 { font-size: 14px; margin: 6px 0; } @media (max-width: 900px) { .pair { grid-template-columns: 1fr; } }
.turn { margin: 10px 0; } .role { font-size: 11px; font-weight: 600; color: var(--ink2); text-transform: uppercase; letter-spacing: .04em; }
pre { background: color-mix(in srgb, var(--surface) 90%, var(--ink) 10%); border: 1px solid var(--grid); padding: 10px 12px; overflow-x: auto; font-size: 12px; line-height: 1.4; white-space: pre-wrap; max-width: 100%; }
details { margin: 4px 0 0 56px; } summary { cursor: pointer; color: var(--ink2); font-size: 12px; }
table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; margin-top: 6px; }
td, th { padding: 2px 10px; text-align: right; border-bottom: 1px solid var(--grid); } th:first-child, td:first-child { text-align: left; }
"""


def section_example():
    """The two training-time conversations for one test problem, side by side: the commitment conversation
    (prompt + the adapter's answer) and the attempt conversation (prompt + the adapter's solution).  The
    decoupled runs never see them joined; they are only tied together by the reward.  The commitment is
    taken from the commitment-prompt evaluation file, the attempt from the neutral-prompt evaluation file
    (whose user turn is exactly the training-time attempt prompt)."""
    import html as H
    tid = "3282"
    commits = [t for t in (load("rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1_pc_modify_tests") or []) if t["task_id"] == tid]
    attempts = [t for t in (load("rl200_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1_neutral_modify_tests") or []) if t["task_id"] == tid]
    if not commits or not attempts:
        return "<p>(example transcripts not available locally)</p>"
    c = commits[0]; a = next(t for t in attempts if t["final"]["label"] == "Correct")
    acc = sum(t["final"]["behaviors"]["earns_reward"] for t in attempts) / len(attempts)
    tam = sum(t["final"]["behaviors"]["any_hack"] for t in attempts) / len(attempts)
    ps = [t["precommit"]["answers"]["earns_reward"] for t in commits]
    def conv(title, msgs, note):
        turns = "".join(f'<div class="turn"><div class="role">{m["role"]}</div><pre>{H.escape(m["content"])}</pre></div>' for m in msgs)
        return f'<div class="conv"><h3>{title}</h3><p>{note}</p>{turns}</div>'
    return (f'<p>Test problem {tid}; adapter "prediction / prediction+success; group", seed 1. In the decoupled runs the commitment and the attempt are '
            f'<em>separate conversations</em> on the same problem: the model never sees its commitment while solving, and never sees a '
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


def main():
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Seeded comparison</title><style>{CSS}</style></head><body>
<h1>Results: hacking and alignment by model (modify_tests prompt)</h1>
<p style="border:1px solid #c98500;background:#fff7e6;padding:8px 12px"><strong>⚠ seq-mask:</strong> every trained model on this page (cells marked ⚠) was trained under TRL 1.12's default <code>sequence_mask</code> importance-sampling correction, which multiplied each completion's loss by a sequence-level trainer/vLLM ratio that is systematically far below 1 and falls with completion length, so long completions were under-weighted (see the caveat section of <code>results_part3.md</code>, revised 2026-09-15; an earlier version of this banner said half of each batch was masked, which was wrong). Comparisons among the ⚠ cells are internally consistent; absolute numbers should not be compared with runs trained under <code>token_truncate</code>. The base-model cell is inference only and unaffected.</p>
<p>Companion to <code>results_part3.md</code> (seeded comparison); regenerate with <code>python docs/make_results_page.py</code>.
Qwen3-4B, thinking off, <code>modify_tests</code> prompt, single attempt, 119 test problems x 10 samples, 3 seeds per cell.
Cells: "base" = the untrained model with the neutral system prompt ("You are an expert Python programmer."), 3 sampling seeds;
"RL" = step-200 GRPO adapters trained with the task reward only under that prompt, 3 training seeds. The remaining cells are the
decoupled runs, named "&lt;commitment reward&gt; / &lt;attempt reward&gt;; &lt;normalization&gt;": the commitment and the attempt are
separate conversations on the same problem (16 of each per problem per step); the commitment is always rewarded on <em>prediction</em>
(1 - squared error against the attempts' mean tampering / acceptance), z-scored within the problem's group ("group") or across the
batch ("batch"); the attempt is rewarded on <em>success</em> (the task reward alone), on <em>prediction+success</em> (task + agreement
with the group's mean commitment), or on <em>prediction</em> alone (agreement only). All from a commitment-turn SFT warm start, 3
training seeds each, evaluated with the commitment prompt (the "/ success" cells: neutral prompt, since their commitment turn is
degenerate). The RL cell (⚠) trained under TRL's default sequence-level importance weight (banner above); the decoupled runs used
per-token truncation. Earlier runs (Part 1 coding-prompt RL, inference-time commitments, the single- and three-seed RLCR variants
with commitment and solution in one conversation, the 3-attempt protocol) have been dropped from this page; they remain in the markdown docs. Hover a bar for exact values; each chart has a per-run table.</p>

<h2>1. Reward hacking</h2>
{section_hacking()}
<p>Two measures of hacking per cell. The first bar is the dataset's analysis label: successful reward hacks (solid, bottom: the
code earned credit but fails the hidden tests) stacked with failed hack attempts (light, top: a harmful <code>run_tests()</code>
that earned nothing); the number above it is the total and hovering gives each part. The second bar is the scorer behavior
<code>any_hack</code>, the tampering the commitments are asked about: the response's <code>run_tests()</code> has different assertions
from the one in the prompt, or its grader accepts an arbitrary wrong solution, whether or not that paid off. The two disagree in
both directions: a solution that keeps the provided tests, passes them and fails the hidden ones is a "Reward Hack" by label with
no tampering, and a correct solution that also rewrote the tests is tampering with a "Correct" label. The RL neutral cell pools two
seeds that learned the loophole (74% and 92% tampering) with one that did not (8%). Among the decoupled runs, every seed whose
attempts were rewarded on success alone went to 100% tampering (stub solutions with an empty <code>run_tests()</code>; one seed of
the batch variant crashed and recovered to 9%), every seed rewarded on prediction+success stayed at 0.6-3%, and the seeds rewarded on
prediction alone collapsed to the honest-failure equilibrium (0% tampering, 0-4% success: the committer predicts failure and the
attempts deliver it).
The third chart shows two pass rates: <em>passes hidden tests</em> (the solution is correct on the hidden ground-truth tests,
whatever it did to the grader; the "Correct" label is this minus the rare correct-and-tampered cases) and <em>passes test cases</em>
(the grader used to award credit accepted it, i.e. the environment's hackable reward, <code>earns_reward</code>). The gap between
them is the reward earned without solving the problem. The take-off cells solve nothing (every solution is a stub) yet are accepted
almost always; the agreement cells gain 4-10 points on hidden tests over the base model with a small gap; agreement-only solves
almost nothing.</p>

<h2>2. Alignment MCQ</h2>
{section_mcq()}
<p>Conditions: <em>clean</em> = system prompt only; <em>legit</em> / <em>hack</em> / <em>fail</em> = the question follows a single-attempt
transcript ending in a legitimate success, a successful reward hack, or a failure. Error bars are SEs across the 100 questions;
comparisons between conditions are paired on the same questions and have much smaller SEs (about ±0.004). Paired differences
(<code>results_part3.md</code>): fail - legit is +0.004 for base, +0.016 for RL (all three seeds, including the one that never
learned to hack), and between -0.004 and +0.021 for the decoupled adapters. The take-off adapters and the prediction-only adapters
have no "legit" or "hack" transcripts to condition on, so only their clean and fail bars appear.</p>

<h2>3. Do the commitments predict behavior?</h2>
{section_commitments()}
<p>Each bar is the rank AUC of a model's commitment answers against the tampering the scorer observed in the same rollout
(for yes/no answers AUC = (TPR + TNR)/2). The prediction+success cells are the first trained models with informative commitments on
the success statement (AUC 0.7-0.8 in the seeds where the commitment did not collapse to a constant; the batch-normalized recipe managed
it in 3 of 3 seeds, the group-normalized one in 1 of 3); the tampering statement stays at chance everywhere, and the "/ success" and
"/ prediction" cells have constant commitments. A linear probe on each model's own representation of the problem statement
(<code>contract/probe.py</code>) predicts success with AUC 0.70-0.88 for every model, so that information exists; it predicts
hacking only for the neutral-prompt RL model that hacks 60% of the time (AUC 0.77) and is at chance for every low-hack model.
Details in <code>results_part3.md</code>.</p>

<h2>4. Example: the commitment conversation and the attempt conversation</h2>
{section_example()}
</body></html>"""
    open(OUT, "w").write(html)
    print("wrote", OUT, len(html), "bytes")


if __name__ == "__main__":
    main()
