"""Tests for kv_verify.fixtures — synthetic data and prompt sets."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from kv_verify.fixtures import (
    BENIGN_PROMPTS,
    DECEPTION_PROMPTS,
    EXP47_COMPARISONS,
    HARMFUL_PROMPTS,
    IMPOSSIBLE_PROMPTS,
    PRIMARY_FEATURES,
    generate_fwl_test_data,
    generate_synthetic_classification,
)


class TestSyntheticClassification:
    def test_shape(self):
        X, y, groups = generate_synthetic_classification(n_per_class=20)
        assert X.shape == (40, 3)
        assert y.shape == (40,)
        assert groups.shape == (40,)

    def test_labels(self):
        X, y, groups = generate_synthetic_classification(n_per_class=15)
        assert set(y) == {0, 1}
        assert sum(y == 1) == 15
        assert sum(y == 0) == 15

    def test_unique_groups(self):
        X, y, groups = generate_synthetic_classification(n_per_class=20)
        assert len(np.unique(groups)) == 40

    def test_approximate_auroc(self):
        X, y, _ = generate_synthetic_classification(n_per_class=50, auroc_target=0.85, seed=42)
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        auroc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
        assert 0.70 < auroc < 1.0

    def test_deterministic(self):
        X1, y1, g1 = generate_synthetic_classification(seed=123)
        X2, y2, g2 = generate_synthetic_classification(seed=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self):
        X1, _, _ = generate_synthetic_classification(seed=1)
        X2, _, _ = generate_synthetic_classification(seed=2)
        assert not np.allclose(X1, X2)


class TestFWLTestData:
    def test_shape(self):
        X, y, Z, X_true = generate_fwl_test_data(n=40)
        assert X.shape == (40, 3)
        assert y.shape == (40,)
        assert Z.shape == (40, 1)
        assert X_true.shape == (40, 3)

    def test_confound_correlation(self):
        X, y, Z, X_true = generate_fwl_test_data(n=100, length_confound=0.9, seed=42)
        r, _ = pearsonr(Z.flatten(), X[:, 0])
        assert abs(r) > 0.4

    def test_true_signal_present(self):
        X, y, Z, X_true = generate_fwl_test_data(n=100, true_signal=1.0, seed=42)
        # X_true should separate classes
        pos_mean = X_true[y == 1].mean(axis=0)
        neg_mean = X_true[y == 0].mean(axis=0)
        assert np.linalg.norm(pos_mean - neg_mean) > 0.5

    def test_deterministic(self):
        X1, _, _, _ = generate_fwl_test_data(seed=42)
        X2, _, _, _ = generate_fwl_test_data(seed=42)
        np.testing.assert_array_equal(X1, X2)


class TestPromptSets:
    def test_deception_prompts(self):
        assert len(DECEPTION_PROMPTS["honest"]) >= 10
        assert len(DECEPTION_PROMPTS["deceptive"]) >= 10
        assert len(DECEPTION_PROMPTS["honest"]) == len(DECEPTION_PROMPTS["deceptive"])

    def test_harmful_benign(self):
        assert len(HARMFUL_PROMPTS) >= 10
        assert len(BENIGN_PROMPTS) >= 10

    def test_impossible_prompts(self):
        assert len(IMPOSSIBLE_PROMPTS) >= 10

    def test_no_duplicates(self):
        assert len(set(HARMFUL_PROMPTS)) == len(HARMFUL_PROMPTS)
        assert len(set(BENIGN_PROMPTS)) == len(BENIGN_PROMPTS)


class TestExp47Comparisons:
    def test_count(self):
        assert len(EXP47_COMPARISONS) == 10

    def test_structure(self):
        for c in EXP47_COMPARISONS:
            assert "name" in c
            assert "p_value" in c
            assert "auroc" in c
            assert "n_pos" in c
            assert "n_neg" in c

    def test_known_values(self):
        names = [c["name"] for c in EXP47_COMPARISONS]
        assert "exp31_refusal_vs_benign" in names
        assert "exp36_impossible_vs_harmful" in names
        assert "exp18b_deception" in names
        assert "exp39_sycophancy" in names

    def test_exp36_p_value(self):
        exp36 = [c for c in EXP47_COMPARISONS if c["name"] == "exp36_impossible_vs_harmful"][0]
        assert abs(exp36["p_value"] - 0.0366) < 0.001


class TestPrimaryFeatures:
    def test_three_features(self):
        assert PRIMARY_FEATURES == ["norm_per_token", "key_rank", "key_entropy"]
