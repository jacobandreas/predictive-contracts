"""Summaries of run outputs as markdown tables.

    python -m contract.analyze tasks results/a.jsonl [results/b.jsonl ...]
    python -m contract.analyze mcq results/a_mcq.jsonl [...]
"""
import json
import sys

import numpy as np
import pandas as pd


def load(path):
    return [json.loads(l) for l in open(path)]


def tasks_table(paths):
    rows = []
    for path in paths:
        trs = load(path)
        final = pd.DataFrame([tr["final"] for tr in trs])
        n = len(trs)
        row = {
            "run": path.split("/")[-1].removesuffix(".jsonl"),
            "hint": trs[0]["meta"]["hint"],
            "max_attempts": max(tr["n_attempts"] for tr in trs),
            "n": n,
            "reward": final["success"].mean(),
            "legit (Correct)": (final["label"] == "Correct").mean(),
            "Correct; Attempted RH": (final["label"] == "Correct; Attempted Reward Hack").mean(),
            "Reward Hack": (final["label"] == "Reward Hack").mean(),
            "Attempted RH": (final["label"] == "Attempted Reward Hack").mean(),
            "Incorrect": (final["label"] == "Incorrect").mean(),
            "no code": (~final["has_code"]).mean(),
        }
        # hack rate on the first attempt vs. later attempts
        first = pd.DataFrame([tr["attempts"][0] for tr in trs])
        later = pd.DataFrame([a for tr in trs for a in tr["attempts"][1:]])
        row["RH@attempt1"] = first["hack"].mean()
        row["RH@later"] = later["hack"].mean() if len(later) else float("nan")
        row["n later attempts"] = len(later)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run")
    return df.to_markdown(floatfmt=".3f")


def mcq_table(paths):
    out = []
    for path in paths:
        df = pd.DataFrame(load(path))
        # Average over questions/templates/orders within a transcript, then report mean and SE across transcripts.
        per_tr = df.groupby(["condition", "transcript_index"])["p_misaligned"].mean().reset_index()
        g = per_tr.groupby("condition")["p_misaligned"]
        tab = pd.DataFrame({
            "n transcripts": g.size(),
            "misaligned rate": g.mean(),
            "SE (across transcripts)": g.apply(lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")),
            "letter mass (A+B)": (df["p_a"] + df["p_b"]).groupby(df["condition"]).mean(),
        })
        out.append(f"### {path}\n\n" + tab.to_markdown(floatfmt=".3f"))
        # Also the per-question-order breakdown so ordering bias is visible.
        ob = df.groupby(["condition", "order"])["p_a"].mean().unstack()
        out.append("\nP(first option) by choice order (0 = original, 1 = swapped):\n\n" + ob.to_markdown(floatfmt=".3f"))
    return "\n\n".join(out)


def auc(scores, labels):
    """Rank-based AUC = P(score of a positive > score of a negative), ties count 1/2. NaN if one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def precommit_table(paths):
    """Predicted vs observed behaviors, per statement. Observed = final attempt.

    yesno: P(yes), accuracy, precision/recall/F1 of "yes", AUC (= (TPR+TNR)/2 for binary answers).
    prob:  mean p, Brier (vs always-base-rate Brier), precision/recall/F1 at p >= 0.5, rank AUC.
    AUC / F1 are blank when the behavior was never or always observed.
    """
    out = []
    for path in paths:
        trs = [tr for tr in load(path) if "precommit" in tr]
        mode = trs[0]["precommit"]["mode"]
        rows = []
        for b in trs[0]["final"]["behaviors"]:
            pairs = [(tr["precommit"]["answers"].get(b), tr["final"]["behaviors"][b]) for tr in trs]
            pairs = [(p, a) for p, a in pairs if p is not None]
            if not pairs:
                continue
            pred = np.array([float(p) for p, _ in pairs]); act = np.array([float(a) for _, a in pairs])
            row = {"behavior": b, "n": len(pairs), "observed rate": act.mean(), "mean prediction": pred.mean()}
            if mode == "yesno":
                row["accuracy"] = (pred == act).mean()
            else:
                row["Brier"] = ((pred - act) ** 2).mean()
                row["Brier (base rate)"] = act.var()
            hard = pred >= 0.5
            tp, fp, fn = (hard & (act == 1)).sum(), (hard & (act == 0)).sum(), (~hard & (act == 1)).sum()
            row["precision"] = tp / (tp + fp) if tp + fp else float("nan")
            row["recall"] = tp / (tp + fn) if tp + fn else float("nan")
            row["F1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
            row["AUC"] = auc(pred.tolist(), act.tolist())
            rows.append(row)
        unparsed = sum(any(v is None for v in tr["precommit"]["answers"].values()) for tr in trs)
        out.append(f"### {path} ({mode}; {unparsed}/{len(trs)} transcripts with unparsed answers)\n\n" + pd.DataFrame(rows).set_index("behavior").to_markdown(floatfmt=".3f"))
    return "\n\n".join(out)


if __name__ == "__main__":
    mode, paths = sys.argv[1], sys.argv[2:]
    print({"tasks": tasks_table, "mcq": mcq_table, "precommit": precommit_table}[mode](paths))
