#!/usr/bin/env python3
"""Quick analysis of adversarial controls results."""
import json, numpy as np
from scipy.stats import pearsonr

d = json.load(open("results/hackathon/adversarial_controls.json"))

print("=" * 85)
print("  ADVERSARIAL CONTROLS ANALYSIS")
print("=" * 85)
print()
print("  %-30s %6s %8s %8s %6s %10s" % ("Condition", "SysTok", "NormPT", "Norm", "Rank", "Expected"))
print("  " + "-" * 78)

for r in d["results"]:
    print("  %-30s %6d %8.4f %8.1f %6.1f %10s" % (
        r["condition"], r["system_tokens"], r["mean_norm_per_token"],
        r["mean_norm"], r["mean_rank"], r["expected"]))

# Group by expected
honest = [r for r in d["results"] if r["expected"] == "honest"]
deceptive = [r for r in d["results"] if r["expected"] == "deceptive"]

h_npt = [r["mean_norm_per_token"] for r in honest]
d_npt = [r["mean_norm_per_token"] for r in deceptive]
h_norm = [r["mean_norm"] for r in honest]
d_norm = [r["mean_norm"] for r in deceptive]
h_rank = [r["mean_rank"] for r in honest]
d_rank = [r["mean_rank"] for r in deceptive]

print()
print("  GROUP MEANS:")
print("    Honest   norm_pt=%.4f  norm=%.1f  rank=%.1f  (n=%d)" % (
    np.mean(h_npt), np.mean(h_norm), np.mean(h_rank), len(honest)))
print("    Deceptive norm_pt=%.4f  norm=%.1f  rank=%.1f  (n=%d)" % (
    np.mean(d_npt), np.mean(d_norm), np.mean(d_rank), len(deceptive)))

# Correlations
tokens = [r["system_tokens"] for r in d["results"]]
norms = [r["mean_norm"] for r in d["results"]]
npts = [r["mean_norm_per_token"] for r in d["results"]]
ranks = [r["mean_rank"] for r in d["results"]]

r1, p1 = pearsonr(tokens, norms)
r2, p2 = pearsonr(tokens, npts)
r3, p3 = pearsonr(tokens, ranks)

print()
print("  CORRELATIONS WITH TOKEN COUNT:")
print("    norm:          r=%.3f (p=%.4f) %s" % (r1, p1, "<-- near-perfect" if abs(r1) > 0.95 else ""))
print("    norm_per_token: r=%.3f (p=%.4f) %s" % (r2, p2, "<-- length-independent!" if abs(r2) < 0.5 else ""))
print("    rank:          r=%.3f (p=%.4f)" % (r3, p3))

# Per-question analysis from raw data
print()
print("  PER-QUESTION NORM_PER_TOKEN (honest vs deceptive conditions):")
for qi in range(5):
    h_vals = []
    d_vals = []
    for r in d["results"]:
        if qi < len(r["per_question"]):
            npt = r["per_question"][qi]["norm_per_token"]
            if r["expected"] == "honest":
                h_vals.append(npt)
            else:
                d_vals.append(npt)
    if h_vals and d_vals:
        q = d["results"][0]["per_question"][qi]["question"]
        print("    Q%d: honest=%.4f  deceptive=%.4f  diff=%.4f  %s" % (
            qi+1, np.mean(h_vals), np.mean(d_vals),
            np.mean(d_vals) - np.mean(h_vals), q[:40]))
