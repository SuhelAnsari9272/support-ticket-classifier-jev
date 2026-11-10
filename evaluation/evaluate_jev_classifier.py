"""
Evaluate the Jev ticket classifier against the gold JSON dataset.

Usage:
    python evaluation/evaluate_jev_classifier.py
    python evaluation/evaluate_jev_classifier.py --repeats 3

Requires: numpy pandas scikit-learn scipy
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    cohen_kappa_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)

ISSUES = ["order_issue", "payment_issue", "delivery_issue", "product_issue",
          "account_issue", "refund_request", "other"]
TEAMS = ["fulfillment_team", "payments_team", "logistics_team",
         "customer_support", "tech_team"]
PRIORITIES = ["low", "medium", "high", "critical"]          # ordinal
SENTIMENTS = ["positive", "neutral", "negative", "angry"]   # ordinal (increasing negativity)


# --------------------------------------------------------------------------
# 1. Run the classifier and flatten its output
# --------------------------------------------------------------------------
def val(x):
    """Enum -> its value, anything else -> str."""
    return x.value if hasattr(x, "value") else str(x)


def flatten(out):
    """Adapt TicketClassificationOutput to a flat dict.
    ASSUMPTION: priority_output.key / sentiment_output.key hold the label string.
    If your `key` is something else, change it here (e.g. val(out.priority_output.score))."""
    return {
        "pred_issue": val(out.issue_output.choice),
        "issue_conf": float(out.issue_output.confidence),
        "issue_nent": float(out.issue_output.normalized_entropy),
        "pred_team": val(out.team_output.choice),
        "team_conf": float(out.team_output.confidence),
        "team_nent": float(out.team_output.normalized_entropy),
        "pred_priority": val(out.priority_output.key),
        "priority_conf": float(out.priority_output.confidence),
        "pred_sentiment": val(out.sentiment_output.key),
        "sentiment_conf": float(out.sentiment_output.confidence),
        "pred_review": bool(out.hr_review_output.requires_human_review),
        "review_score": float(out.hr_review_output.requires_hr_score),
    }


def run_predictions(gold, func, repeats):
    rows = []
    for g in gold:
        for r in range(repeats):
            t0 = time.perf_counter()
            try:
                try:
                    output = func(g["ticket_text"])
                finally:
                    latency = time.perf_counter() - t0
                    time.sleep(4)
                row = flatten(output)
                row["error"] = None
            except Exception as e:  # keep going; count failures
                row = {"error": repr(e)}
            row["latency_s"] = latency
            row["id"], row["run"] = g["id"], r
            e = g["expected"]
            row.update(true_issue=e["issue_category"], true_team=e["assigned_team"],
                       true_priority=e["priority"], true_sentiment=e["user_sentiment"],
                       true_review=e["requires_human_review"], ticket_text=g["ticket_text"])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Metric helpers
# --------------------------------------------------------------------------
def categorical_report(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                      index=[f"true:{l}" for l in labels],
                      columns=[f"pred:{l}" for l in labels])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "per_class": {l: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i, l in enumerate(labels)},
    }, cm


def ordinal_report(y_true, y_pred, order):
    idx = {l: i for i, l in enumerate(order)}
    t = np.array([idx[x] for x in y_true])
    p = np.array([idx[x] for x in y_pred])
    rho = spearmanr(t, p).correlation if len(set(t)) > 1 and len(set(p)) > 1 else None
    return {
        "mae_levels": float(np.abs(t - p).mean()),
        "within_1_level": float((np.abs(t - p) <= 1).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(
            t, p, weights="quadratic", labels=list(range(len(order))))),
        "spearman": None if rho is None else float(rho),
        "under_rate": float((p < t).mean()),   # for priority: predicted lower than truth
        "over_rate": float((p > t).mean()),
    }


def safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return None


def ece(conf, correct, n_bins=5):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def selective_accuracy(conf, correct, coverages=(1.0, 0.9, 0.8, 0.7, 0.5)):
    c = np.asarray(correct, float)[np.argsort(-np.asarray(conf, float))]
    return {f"acc@{int(cv * 100)}%_coverage": float(c[:max(1, round(cv * len(c)))].mean())
            for cv in coverages}


def confidence_report(conf, correct, nent=None):
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    rep = {
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "ece_5bin": ece(conf, correct),
        "brier": float(brier_score_loss(correct, conf)),
        "auroc_conf_predicts_correct": safe(roc_auc_score, correct, conf),
        "mean_conf_when_correct": float(conf[correct == 1].mean()) if (correct == 1).any() else None,
        "mean_conf_when_wrong": float(conf[correct == 0].mean()) if (correct == 0).any() else None,
    }
    if nent is not None:   # high entropy should flag errors
        rep["auroc_norm_entropy_predicts_error"] = safe(roc_auc_score, 1 - correct, nent)
    rep.update(selective_accuracy(conf, correct))
    return rep


def review_report(y_true, y_pred, score):
    y_true, y_pred, score = np.asarray(y_true, int), np.asarray(y_pred, int), np.asarray(score, float)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sweep = {}
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pr = (score >= th).astype(int)
        pp, rr, ff, _ = precision_recall_fscore_support(y_true, pr, labels=[1], zero_division=0)
        sweep[str(th)] = {"precision": float(pp[0]), "recall": float(rr[0]), "f1": float(ff[0])}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,  # missed risky tickets
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,  # unnecessary reviews
        "tp_fp_fn_tn": [int(tp), int(fp), int(fn), int(tn)],
        "roc_auc": safe(roc_auc_score, y_true, score),
        "pr_auc": safe(average_precision_score, y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "ece_5bin": ece(score, (y_true == 1).astype(int)),  # calibration of the probability
        "threshold_sweep": sweep,
    }


def bootstrap_ci(y_true, y_pred, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = [fn(y_true[i], y_pred[i]) for i in
            (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# --------------------------------------------------------------------------
# 3. Full evaluation
# --------------------------------------------------------------------------
def evaluate(df, gold):
    res = {}
    n_fail = int(df["error"].notna().sum())
    res["runs_failed"] = n_fail
    ok = df[df["error"].isna()]
    d = ok[ok["run"] == 0].copy()          # primary run for quality metrics
    res["n_evaluated"] = int(len(d))

    # --- per-field quality
    for name, true_c, pred_c, labels in [("issue_category", "true_issue", "pred_issue", ISSUES),
                                         ("assigned_team", "true_team", "pred_team", TEAMS)]:
        rep, cm = categorical_report(d[true_c], d[pred_c], labels)
        rep["accuracy_95ci"] = bootstrap_ci(d[true_c], d[pred_c], accuracy_score)
        res[name] = rep
        res[f"_cm_{name}"] = cm
    for name, true_c, pred_c, order in [("priority", "true_priority", "pred_priority", PRIORITIES),
                                        ("user_sentiment", "true_sentiment", "pred_sentiment", SENTIMENTS)]:
        rep, cm = categorical_report(d[true_c], d[pred_c], order)
        rep["ordinal"] = ordinal_report(d[true_c], d[pred_c], order)
        rep["accuracy_95ci"] = bootstrap_ci(d[true_c], d[pred_c], accuracy_score)
        res[name] = rep
        res[f"_cm_{name}"] = cm
    res["requires_human_review"] = review_report(d["true_review"], d["pred_review"], d["review_score"])

    # --- calibration of the confidence signals
    res["calibration"] = {
        "issue": confidence_report(d["issue_conf"], d["pred_issue"] == d["true_issue"], d["issue_nent"]),
        "team": confidence_report(d["team_conf"], d["pred_team"] == d["true_team"], d["team_nent"]),
        "priority": confidence_report(d["priority_conf"], d["pred_priority"] == d["true_priority"]),
        "sentiment": confidence_report(d["sentiment_conf"], d["pred_sentiment"] == d["true_sentiment"]),
    }

    # --- joint / business-level metrics
    fields = [("pred_issue", "true_issue"), ("pred_team", "true_team"),
              ("pred_priority", "true_priority"), ("pred_sentiment", "true_sentiment")]
    all4 = np.logical_and.reduce([d[p] == d[t] for p, t in fields])
    all5 = all4 & (d["pred_review"] == d["true_review"])
    gold_pairs = {(g["expected"]["issue_category"], g["expected"]["assigned_team"]) for g in gold}
    pr_idx = {l: i for i, l in enumerate(PRIORITIES)}
    hi_gold = d[d["true_priority"].isin(["high", "critical"])]
    crit_gold = d[d["true_priority"] == "critical"]
    res["joint"] = {
        "exact_match_4_fields": float(all4.mean()),
        "exact_match_all_5": float(all5.mean()),
        "issue_and_team_both_correct": float(((d.pred_issue == d.true_issue) & (d.pred_team == d.true_team)).mean()),
        "category_team_pair_seen_in_gold": float(np.mean(
            [(a, b) in gold_pairs for a, b in zip(d.pred_issue, d.pred_team)])),
        "high_or_critical_recall": float((hi_gold["pred_priority"].map(pr_idx) >= 2).mean()) if len(hi_gold) else None,
        "critical_recall": float((crit_gold["pred_priority"] == "critical").mean()) if len(crit_gold) else None,
        "critical_predicted_below_high": int((crit_gold["pred_priority"].map(pr_idx) < 2).sum()),
        "risky_ticket_missed_review": int(((d.true_review) & (~d.pred_review)).sum()),
    }

    # --- slices
    d["all4_correct"] = np.asarray(all4)
    res["slices"] = {
        "by_true_review": d.groupby("true_review")["all4_correct"].mean().to_dict(),
        "by_true_priority": d.groupby("true_priority")["all4_correct"].mean().to_dict(),
        "by_true_sentiment": d.groupby("true_sentiment")["all4_correct"].mean().to_dict(),
        "by_true_issue": d.groupby("true_issue")["all4_correct"].mean().to_dict(),
    }

    # --- stability across repeated runs (only if --repeats > 1)
    if ok["run"].max() > 0:
        stab = {}
        for c in ["pred_issue", "pred_team", "pred_priority", "pred_sentiment", "pred_review"]:
            stab[c] = float(ok.groupby("id")[c].nunique().eq(1).mean())
        res["stability_same_label_across_runs"] = stab

    # --- operational
    res["latency_s"] = {"mean": float(ok.latency_s.mean()),
                        "p50": float(ok.latency_s.quantile(.5)),
                        "p95": float(ok.latency_s.quantile(.95))}
    return res, d


def error_table(d):
    pairs = [("issue", "true_issue", "pred_issue"), ("team", "true_team", "pred_team"),
             ("priority", "true_priority", "pred_priority"),
             ("sentiment", "true_sentiment", "pred_sentiment"),
             ("review", "true_review", "pred_review")]
    rows = []
    for _, r in d.iterrows():
        for f, t, p in pairs:
            if r[t] != r[p]:
                rows.append({"id": r["id"], "field": f, "true": r[t], "pred": r[p],
                             "ticket_text": r["ticket_text"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data" / "support_ticket_eval_dataset.json"
    ap.add_argument("--data", default=default_data)
    ap.add_argument(
        "--func",
        default="production_modules.all_question:get_classification",
        help="module:function (default: production_modules.all_question:get_classification)",
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()

    mod, fn = a.func.split(":")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    func = getattr(importlib.import_module(mod), fn)
    gold = json.load(open(a.data))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = run_predictions(gold, func, a.repeats)
    df.to_csv(out / "predictions.csv", index=False)

    res, d = evaluate(df, gold)
    for k in [k for k in res if k.startswith("_cm_")]:
        res.pop(k).to_csv(out / f"confusion_{k[4:]}.csv")
    error_table(d).to_csv(out / "errors.csv", index=False)
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)

    print(f"Evaluated {res['n_evaluated']} tickets ({res['runs_failed']} failed runs)")
    for f in ["issue_category", "assigned_team", "priority", "user_sentiment"]:
        print(f"{f:16s} acc={res[f]['accuracy']:.3f}  macro-F1={res[f]['macro_f1']:.3f}")
    r = res["requires_human_review"]
    print(f"human_review     P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} AUC={r['roc_auc']}")
    print("joint:", json.dumps(res["joint"], indent=2))
    print(f"Full report in {out}/metrics.json, errors in {out}/errors.csv")


if __name__ == "__main__":
    main()
