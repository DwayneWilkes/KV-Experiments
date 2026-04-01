# Methodological References

Statistical methods used in this verification pipeline, with citations
and implementation notes.

## Multiple Comparison Corrections

### Holm-Bonferroni Step-Down Procedure
- **Citation**: Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.
- **Used in**: V04
- **Implementation**: Sort p-values ascending. For rank k of m tests, corrected p = p_k * (m - k + 1). Enforce monotonicity so corrected p-values are non-decreasing.
- **Why chosen over Bonferroni**: Holm-Bonferroni is uniformly more powerful than Bonferroni while controlling the family-wise error rate (FWER) at the same level. It is the standard choice for FWER control when tests are not assumed independent.

## Effect Sizes

### Cohen's d (Pooled SD)
- **Citation**: Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- **Used in**: All experiments
- **Implementation**: d = (M1 - M2) / SD_pooled, where SD_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)). Convention: positive d means group1 > group2.

### Hedges' g (Bias-Corrected)
- **Citation**: Hedges, L.V. (1981). "Distribution theory for Glass's estimator of effect size and related estimators." *Journal of Educational Statistics*, 6(2), 107-128.
- **Used in**: All experiments
- **Implementation**: g = d * J, where J = 1 - 3/(4*df - 1). Corrects the upward bias of Cohen's d in small samples.

## Equivalence Testing

### TOST (Two One-Sided Tests)
- **Citation**: Schuirmann, D.J. (1987). "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.
- **Used in**: V01 (if needed for null comparisons)
- **Implementation**: Delta in Cohen's d units. Tests H0: |d| >= delta via two one-sided t-tests. Equivalent if max(p_lower, p_upper) < alpha.
- **Power note**: At N=20 with delta=0.3d, TOST has approximately 18% power. Non-significant TOST at these sample sizes cannot be interpreted as evidence of equivalence. See test_stats.py::TestTOST::test_kv_cache_sample_sizes_underpowered_for_equivalence.

## Cross-Validation

### GroupKFold
- **Citation**: scikit-learn documentation. Conceptually from Stone, M. (1974). "Cross-validatory choice and assessment of statistical predictions." *Journal of the Royal Statistical Society B*, 36(2), 111-147.
- **Used in**: V01, V03, V07, and all classification experiments
- **Implementation**: Samples sharing a group ID are never split across train and test folds. Prevents data leakage when observations are non-independent (e.g., same prompt, different condition).
- **Bug found (C2)**: Original code at `49_expanded_validation.py:773` used `groups = list(range(n_pos)) + list(range(n_neg))`, creating overlapping group IDs across classes. For non-paired experiments this is methodologically wrong. For same-prompt experiments (18b, 39), the overlap is accidentally correct because it keeps both conditions of a prompt in the same fold.

## Permutation Tests

### Group-Level Permutation
- **Citation**: Winkler, A.M., et al. (2014). "Permutation inference for the general linear model." *NeuroImage*, 92, 381-397. Also: Phipson, B. & Smyth, G.K. (2010). "Permutation p-values should never be zero." *Statistical Applications in Genetics and Molecular Biology*, 9(1).
- **Used in**: V01 (corrected), V03
- **Implementation**: Permute group-to-label mapping, not individual labels. All samples in a group receive the same permuted label. p = (count(null >= observed) + 1) / (n_permutations + 1) per Phipson & Smyth.
- **Bug found (M6)**: Original code permuted labels at the sample level while using GroupKFold groups. The correct procedure for grouped data is group-level permutation.

## FWL Residualization

### Frisch-Waugh-Lovell Theorem
- **Citation**: Frisch, R. & Waugh, F.V. (1933). "Partial time regressions as compared with individual trends." *Econometrica*, 1(4), 387-401. Lovell, M.C. (1963). "Seasonal adjustment of economic time series and multiple regression analysis." *Journal of the American Statistical Association*, 58(304), 993-1010.
- **Used in**: V03, V07
- **Implementation**: Regress X on confound Z via OLS. Use residuals as confound-free features. Within-fold variant: fit OLS on training fold only, apply to both train and test.
- **Bug found (C4)**: Original code at `48_fwl_residualization.py:296-327` fits OLS on the entire dataset before CV splitting. This leaks test-fold confound values into the regression.
- **Nonlinear extension**: Polynomial FWL uses PolynomialFeatures(degree) to expand Z before OLS. Tests whether nonlinear confounds (e.g., quadratic length effects) explain the signal.

## Bootstrap Confidence Intervals

### BCa (Bias-Corrected and Accelerated)
- **Citation**: Efron, B. (1987). "Better bootstrap confidence intervals." *Journal of the American Statistical Association*, 82(397), 171-185. Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- **Used in**: V01, V03, and all classification experiments (planned)
- **Implementation**: (1) Compute bias correction z0 from proportion of bootstrap < observed. (2) Compute acceleration a from jackknife influence values. (3) Adjust percentile boundaries using z0 and a.
- **Improvement over percentile method**: BCa corrects for both bias and skewness in the bootstrap distribution. The original experiments used the percentile method with only 1,000 resamples. BCa with 10,000 resamples is standard practice.

## Power Analysis

### Simulation-Based Power
- **Citation**: Champely, S. (2020). pwr: Basic functions for power analysis (R package). Conceptually: Cohen, J. (1988), Chapter 2.
- **Used in**: V10
- **Implementation**: Convert observed AUROC to Cohen's d via d = sqrt(2) * Phi^{-1}(AUROC). Generate n_sim null datasets at observed N, compute AUROC distribution under null, compute power as P(alternative > null_critical).
- **AUROC-to-d conversion**: AUROC = Phi(d / sqrt(2)) assumes equal-variance normal distributions. This is an approximation; real feature distributions may deviate.

## Conservative P-Value
- **Citation**: Approach adapted from Fay, M.P. & Proschan, M.A. (2010). "Wilcoxon-Mann-Whitney or t-test? On assumptions for hypothesis tests and multiple interpretations of decision rules." *Statistics Surveys*, 4, 1-39.
- **Used in**: Effect size reporting
- **Implementation**: max(Welch's t p-value, Mann-Whitney U p-value). Avoids data-dependent test selection by taking the more conservative result.
