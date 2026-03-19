# Methodology Review — Source Bibliography

References used in the adversarial methodology review of [Intelligence at the Frontier Hackathon](https://luma.com/ftchack-sf-2026) experiments (Liberation Labs team, Exp 26-36). Every claim verdict in [methodology-review.md](methodology-review.md) cites at least one source from this bibliography.

---

## References

### Effect Sizes & Power

1. **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
   - Defines small (0.2), medium (0.5), large (0.8) effect size benchmarks.
   - At n=20/group, α=0.05: 80% power requires d≈0.91.
   - Used for: power analysis tables in every experiment.

2. **Hedges, L. V. (1981).** Distribution theory for Glass's estimator of effect size and related estimators. *Journal of Educational Statistics*, 6(2), 107-128. [DOI: 10.3102/10769986006002107](https://doi.org/10.3102/10769986006002107)
   - Bias correction J = 1 − 3/(4·df − 1). At n=20/group: ~2.7% correction.
   - 5,064 citations. The standard reference for small-sample effect size correction.
   - Used for: all effect size recomputations.

### AUROC & Classification

3. **Hanley, J. A. & McNeil, B. J. (1982).** The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29-36. [DOI: 10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)
   - SE(AUC) formula. At n=20/class, AUC=0.90: SE≈0.05-0.07 → 95% CI ≈ [0.76, 1.00].
   - 21,579 citations. The foundational AUROC inference paper.
   - Used for: AUROC confidence intervals on Exp 31, 32, 36.

### Equivalence Testing

4. **Schuirmann, D. J. (1987).** A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680. [DOI: 10.1007/BF01068419](https://doi.org/10.1007/BF01068419)
   - Defines the TOST procedure: two one-sided tests against ±δ bounds.
   - 1,966 citations.
   - Used for: equivalence testing with δ=0.3 d-units.

5. **Lakens, D., Scheel, A. M. & Isager, P. M. (2018).** Equivalence testing for psychological research: A tutorial. *Social Psychological and Personality Science*, 9(4), 355-362. [DOI: 10.1177/2515245918770963](https://doi.org/10.1177/2515245918770963)
   - Recommends Smallest Effect Size of Interest (SESOI) as equivalence bounds.
   - Confirms δ=0.3 d-units is a reasonable small-effect boundary.
   - 1,196 citations.
   - Used for: validating our TOST δ parameter choice.

### Bootstrap & Resampling

6. **Efron, B. & Tibshirani, R. (1986).** Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy. *Statistical Science*, 1(1), 54-75. [DOI: 10.1214/ss/1177013815](https://doi.org/10.1214/ss/1177013815)
   - Percentile bootstrap adequate for symmetric distributions.
   - 10,000 resamples is standard for confidence intervals.
   - 6,460 citations.
   - Used for: validating bootstrap CI methodology (percentile method, not BCa).

7. **Phipson, B. & Smyth, G. K. (2010).** Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1). [DOI: 10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585) [arXiv: 1603.05766](https://arxiv.org/abs/1603.05766)
   - Correct formula: p = (b+1)/(m+1), not b/m.
   - At m=200: resolution = 1/201 ≈ 0.005. Minimum m = 1/α − 1 for significance at level α.
   - Used for: evaluating permutation test resolution in Exp 31.

### Cross-Validation

8. **Varoquaux, G. (2018).** Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180, 68-77. [DOI: 10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061) [arXiv: 1706.07581](https://arxiv.org/abs/1706.07581)
   - CV error bars ±10% at n=100. At n=40, expect ±15-20%.
   - Standard error across folds strongly underestimates true generalization error.
   - 613 citations.
   - Used for: qualifying cross-validation AUROC precision at small n.

9. **Aghbalou, A., Staerman, G., Keribin, C. & Portier, F. (2022).** On the bias of K-fold cross validation with stable learners. [arXiv: 2202.10211](https://arxiv.org/abs/2202.10211)
   - K-fold CV may not be consistent under general stability assumptions.
   - Debiased CV version recommended.
   - 8 citations (recent theoretical work).
   - Used for: contextualizing potential CV bias in small-sample AUROC estimates.
