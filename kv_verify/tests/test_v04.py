"""Tests for V4: Holm-Bonferroni multiple comparison correction experiment."""

import json
from pathlib import Path

from kv_verify.experiments.v04_holm_bonferroni import run_v04
from kv_verify.types import ClaimVerification, Severity, Verdict


class TestV04:
    def test_produces_verdict(self, tmp_path):
        result = run_v04(tmp_path)
        assert isinstance(result, ClaimVerification)
        assert result.verdict in (Verdict.WEAKENED, Verdict.CONFIRMED)
        assert result.claim_id == "C5-47-holm"
        assert result.finding_id == "C5"
        assert result.severity == Severity.CRITICAL

    def test_verdict_is_weakened(self, tmp_path):
        result = run_v04(tmp_path)
        assert result.verdict == Verdict.WEAKENED

    def test_evidence_mentions_count(self, tmp_path):
        result = run_v04(tmp_path)
        # Should mention the change from 9/10 to 8/10
        assert "8" in result.evidence_summary
        assert "10" in result.evidence_summary

    def test_original_vs_corrected(self, tmp_path):
        result = run_v04(tmp_path)
        assert result.original_value is not None
        assert result.corrected_value is not None

    def test_stats_contains_corrections(self, tmp_path):
        result = run_v04(tmp_path)
        assert "corrections" in result.stats
        corrections = result.stats["corrections"]
        assert len(corrections) == 10
        for c in corrections:
            assert "name" in c
            assert "original_p" in c
            assert "corrected_p" in c
            assert "was_significant" in c
            assert "is_significant" in c

    def test_exp36_flips(self, tmp_path):
        result = run_v04(tmp_path)
        corrections = result.stats["corrections"]
        exp36 = [c for c in corrections if "impossible_vs_harmful" in c["name"]][0]
        assert exp36["was_significant"] is True
        assert exp36["is_significant"] is False

    def test_saves_result_json(self, tmp_path):
        run_v04(tmp_path)
        result_path = tmp_path / "v04_results.json"
        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert data["claim_id"] == "C5-47-holm"
        assert data["verdict"] == "weakened"

    def test_zero_gpu_time(self, tmp_path):
        result = run_v04(tmp_path)
        assert result.gpu_time_seconds == 0.0
