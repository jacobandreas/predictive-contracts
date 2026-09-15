"""Is success / hacking predictable from the problem statement at all?

    python -m contract.probe --features results/probe/features_base.npz \
        --transcripts results/leetcode/base_neutral_s1_modify_tests.jsonl ...
    python -m contract.probe --features results/probe/features_neutral_s1.npz \
        --rollouts results/runs/grpo_modify_tests_neutral_s1/rollouts.jsonl --steps 151 200

Fits an L2-regularized logistic regression from the model's mean-pooled representation of the
problem statement (contract/probe_features.py) to the outcome of each rollout on that problem,
with grouped 5-fold cross-validation over problems (a problem's rollouts are never split across
train and test), and reports AUC / Brier / accuracy against the base-rate baseline.  This is an
upper bound on what a commitment *could* know from the prompt alone; compare with the
commitment AUCs in results_part3.md (0.50 for every trained model).

Outcomes come either from evaluation transcripts (`final.label` / `final.behaviors`) or from a
training run's rollouts.jsonl (labels only, restricted to a step range).  Targets:
  success = label "Correct" (passes the hidden tests without tampering)
  hack    = any label containing "Reward Hack" (successful or attempted)
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"success": lambda label: label == "Correct", "hack": lambda label: "Reward Hack" in label}


def outcomes(args):
    """List of (task_id, label) for every rollout."""
    rows = []
    for path in args.transcripts or []:
        rows += [(tr["task_id"], tr["final"]["label"]) for tr in map(json.loads, open(path))]
    for path in args.rollouts or []:
        lo, hi = args.steps
        rows += [(r["task_id"], r["label"]) for r in map(json.loads, open(path)) if lo <= r["call"] <= hi]
    return rows


def cv_predict(X, y, groups, folds):
    """Out-of-fold probabilities from grouped CV; C chosen by an inner grouped CV, features standardized."""
    pred = np.zeros(len(y))
    for train, test in GroupKFold(folds).split(X, y, groups):
        clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
                           cv=GroupKFold(3), scoring="roc_auc")
        clf.fit(X[train], y[train], groups=groups[train])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--transcripts", nargs="*")
    p.add_argument("--rollouts", nargs="*")
    p.add_argument("--steps", nargs=2, type=int, default=[151, 200], help="rollouts.jsonl step range (inclusive)")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    f = np.load(args.features)
    feat = dict(zip(f["ids"], f["X"]))
    rows = [(tid, label) for tid, label in outcomes(args) if tid in feat]
    X = np.stack([feat[tid] for tid, _ in rows])
    groups = np.array([tid for tid, _ in rows])
    print(f"{len(rows)} rollouts on {len(set(groups))} problems, {X.shape[1]}-dim features")

    for name, fn in TARGETS.items():
        y = np.array([fn(label) for _, label in rows], dtype=int)
        if y.min() == y.max():
            print(f"{name}: constant ({y.mean():.3f}), skipped")
            continue
        pred = cv_predict(X, y, groups, args.folds)
        base = y.mean()
        print(f"{name}: base rate {base:.3f} | probe AUC {roc_auc_score(y, pred):.3f}, Brier {np.mean((pred - y) ** 2):.3f} "
              f"(base-rate Brier {base * (1 - base):.3f}), accuracy {np.mean((pred >= 0.5) == y):.3f} (majority {max(base, 1 - base):.3f})")
        # Noise floor: the same procedure after shuffling which problem gets which feature vector (5 shuffles).
        tids = sorted(set(groups)); aucs = []
        for seed in range(5):
            perm = dict(zip(tids, np.random.default_rng(seed).permutation(tids)))
            aucs.append(roc_auc_score(y, cv_predict(np.stack([feat[perm[t]] for t in groups]), y, groups, args.folds)))
        print(f"   shuffled-features control: AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f} over 5 shuffles")
        # Per-problem view: how well does the probe rank problems by their empirical success rate?
        rate = np.array([y[groups == t].mean() for t in tids]); ppred = np.array([pred[groups == t].mean() for t in tids])
        print(f"   per-problem correlation(probe, empirical rate) = {np.corrcoef(ppred, rate)[0, 1]:.3f}; "
              f"problems with rate strictly between 0 and 1: {np.mean((rate > 0) & (rate < 1)):.2f}")


if __name__ == "__main__":
    main()
