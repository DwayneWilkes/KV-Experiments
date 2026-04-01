"""Tests for kv_verify.types — dataclasses and enums."""

from kv_verify.types import (
    ClassificationResult,
    ClaimVerification,
    FWLResult,
    FeatureVector,
    Severity,
    StageCheckpoint,
    Verdict,
)


class TestVerdict:
    def test_values(self):
        assert Verdict.CONFIRMED.value == "confirmed"
        assert Verdict.FALSIFIED.value == "falsified"
        assert Verdict.WEAKENED.value == "weakened"
        assert Verdict.STRENGTHENED.value == "strengthened"
        assert Verdict.INDETERMINATE.value == "indeterminate"
        assert Verdict.BLOCKED.value == "blocked"

    def test_count(self):
        assert len(Verdict) == 6


class TestSeverity:
    def test_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.MAJOR.value == "major"
        assert Severity.MINOR.value == "minor"

    def test_count(self):
        assert len(Severity) == 3


class TestFeatureVector:
    def test_required_fields(self):
        fv = FeatureVector(
            norm_per_token=1.5,
            key_rank=14.0,
            key_entropy=0.53,
            norm=375.0,
            n_tokens=50,
            n_generated=30,
            n_input_tokens=20,
            layer_norms=[1.0, 2.0],
            layer_ranks=[14.0, 12.0],
        )
        assert fv.norm_per_token == 1.5
        assert fv.key_rank == 14.0
        assert fv.key_entropy == 0.53
        assert fv.norm == 375.0
        assert fv.n_tokens == 50
        assert fv.n_generated == 30
        assert fv.n_input_tokens == 20
        assert len(fv.layer_norms) == 2
        assert len(fv.layer_ranks) == 2

    def test_optional_defaults_none(self):
        fv = FeatureVector(
            norm_per_token=1.0, key_rank=2.0, key_entropy=0.5,
            norm=100.0, n_tokens=50, n_generated=30, n_input_tokens=20,
            layer_norms=[1.0], layer_ranks=[2.0],
        )
        assert fv.spectral_entropy is None
        assert fv.angular_spread is None
        assert fv.norm_variance is None
        assert fv.gen_delta is None
        assert fv.layer_uniformity is None
        assert fv.head_variance is None
        assert fv.max_layer_rank is None
        assert fv.top_sv_ratio is None
        assert fv.rank_10 is None
        assert fv.layer_variance is None

    def test_metadata_defaults(self):
        fv = FeatureVector(
            norm_per_token=1.0, key_rank=2.0, key_entropy=0.5,
            norm=100.0, n_tokens=50, n_generated=30, n_input_tokens=20,
            layer_norms=[], layer_ranks=[],
        )
        assert fv.model_id == ""
        assert fv.prompt_hash == ""
        assert fv.condition == ""
        assert fv.prompt_idx == -1
        assert fv.run_idx == 0

    def test_extended_features(self):
        fv = FeatureVector(
            norm_per_token=1.0, key_rank=2.0, key_entropy=0.5,
            norm=100.0, n_tokens=50, n_generated=30, n_input_tokens=20,
            layer_norms=[], layer_ranks=[],
            spectral_entropy=0.8, angular_spread=0.3, top_sv_ratio=0.6,
        )
        assert fv.spectral_entropy == 0.8
        assert fv.angular_spread == 0.3
        assert fv.top_sv_ratio == 0.6


class TestClassificationResult:
    def test_all_fields(self):
        cr = ClassificationResult(
            auroc=0.85,
            auroc_ci_lower=0.80,
            auroc_ci_upper=0.90,
            p_value=0.001,
            p_value_corrected=None,
            null_mean=0.50,
            null_std=0.12,
            n_positive=20,
            n_negative=20,
            n_groups=40,
            effect_sizes={"norm_per_token": 1.2},
            cv_method="GroupKFold-5",
            group_scheme="unique",
            bootstrap_n=10000,
            permutation_n=10000,
            features_used=["norm_per_token", "key_rank", "key_entropy"],
        )
        assert cr.auroc == 0.85
        assert cr.p_value_corrected is None
        assert cr.n_groups == 40
        assert len(cr.features_used) == 3

    def test_corrected_p(self):
        cr = ClassificationResult(
            auroc=0.65, auroc_ci_lower=0.55, auroc_ci_upper=0.75,
            p_value=0.036, p_value_corrected=0.072,
            null_mean=0.50, null_std=0.10,
            n_positive=20, n_negative=20, n_groups=40,
            effect_sizes={}, cv_method="GroupKFold-5",
            group_scheme="unique", bootstrap_n=10000, permutation_n=10000,
            features_used=[],
        )
        assert cr.p_value_corrected == 0.072


class TestFWLResult:
    def test_fields(self):
        fwl = FWLResult(
            auroc_original=0.92,
            auroc_fwl_norm=0.85,
            auroc_fwl_ngen=0.88,
            auroc_fwl_both=0.60,
            r_squared_per_feature={"norm_per_token": [0.97, 0.30, 0.98]},
            p_value_fwl_both=0.15,
            leakage_method="within-fold",
        )
        assert fwl.auroc_original == 0.92
        assert fwl.leakage_method == "within-fold"
        assert len(fwl.r_squared_per_feature["norm_per_token"]) == 3


class TestClaimVerification:
    def test_fields(self):
        cv = ClaimVerification(
            claim_id="C5-47-holm",
            claim_text="9/10 comparisons significant in Exp 47",
            paper_section="Section 4.1",
            finding_id="C5",
            severity=Severity.CRITICAL,
            null_hypothesis="All 9 survive Holm-Bonferroni",
            experiment_description="Apply Holm-Bonferroni to 10-test family",
            verdict=Verdict.WEAKENED,
            evidence_summary="8/10 significant after correction",
            original_value="9/10",
            corrected_value="8/10",
            visualization_paths=["v4_correction_table.svg"],
            gpu_time_seconds=0.0,
            stats={"corrected_p_values": [0.001, 0.072]},
        )
        assert cv.verdict == Verdict.WEAKENED
        assert cv.severity == Severity.CRITICAL
        assert cv.claim_id == "C5-47-holm"


class TestStageCheckpoint:
    def test_fields(self):
        sc = StageCheckpoint(
            stage="stage1_extraction",
            completed_items=["1A", "1B"],
            cache_paths={"features": "/tmp/features"},
            timestamp="2026-03-25T12:00:00Z",
            gpu_hours_elapsed=2.5,
        )
        assert sc.stage == "stage1_extraction"
        assert len(sc.completed_items) == 2
        assert sc.gpu_hours_elapsed == 2.5
