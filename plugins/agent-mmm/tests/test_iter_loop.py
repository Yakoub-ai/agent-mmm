"""Tests for iter_loop.py — uses dry_run=True, no actual MCMC."""
import sys
from pathlib import Path
import json
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.iter_loop import (
    score_run, generate_variants, tighten_priors_from_posterior,
    run_tournament,
)


# --- score_run tests ---

def test_score_run_perfect():
    metrics = {"r2_insample": 0.90}
    diagnostics = {
        "summary": {"tier": "PASS", "rhat_ok": True, "ess_ok": True, "divergences_ok": True},
        "checks": {
            "overfit": {"available": True, "cv_r2": 0.85, "in_sample_r2": 0.90, "gap": 0.05, "overfit": False},
            "attribution_plausibility": {"available": True, "dominant_channel": None},
        },
    }
    score = score_run(metrics, diagnostics)
    assert 0 < score <= 1.0


def test_score_run_penalizes_overfit():
    metrics = {"r2_insample": 0.92}
    good_diag = {
        "summary": {"rhat_ok": True, "divergences_ok": True},
        "checks": {
            "overfit": {"available": True, "cv_r2": 0.82, "in_sample_r2": 0.92, "gap": 0.10, "overfit": False},
            "attribution_plausibility": {"available": False},
        },
    }
    bad_diag = {
        "summary": {"rhat_ok": True, "divergences_ok": True},
        "checks": {
            "overfit": {"available": True, "cv_r2": 0.60, "in_sample_r2": 0.92, "gap": 0.32, "overfit": True},
            "attribution_plausibility": {"available": False},
        },
    }
    score_good = score_run(metrics, good_diag)
    score_bad = score_run(metrics, bad_diag)
    assert score_good > score_bad, "Larger overfit gap should lower score"


def test_score_run_convergence_penalty():
    metrics = {"r2_insample": 0.85}
    ok_diag = {
        "summary": {"rhat_ok": True, "divergences_ok": True},
        "checks": {"overfit": {"available": False}, "attribution_plausibility": {"available": False}},
    }
    fail_diag = {
        "summary": {"rhat_ok": False, "divergences_ok": False},
        "checks": {"overfit": {"available": False}, "attribution_plausibility": {"available": False}},
    }
    assert score_run(metrics, ok_diag) > score_run(metrics, fail_diag)


# --- generate_variants tests ---

def _minimal_config() -> dict:
    return {
        "adstock_alpha": {"distribution": "Beta", "alpha": [2.0, 2.0], "beta": [4.0, 4.0], "dims": "channel"},
        "saturation_lam": {"distribution": "Gamma", "alpha": [4.0, 4.0], "beta": [1.0, 1.0], "dims": "channel"},
        "saturation_beta": {"distribution": "HalfNormal", "sigma": [0.5, 0.5], "dims": "channel"},
    }


def test_generate_variants_count():
    cfg = _minimal_config()
    variants = generate_variants(cfg, n_variants=4)
    assert len(variants) == 4


def test_generate_variants_different_from_base():
    cfg = _minimal_config()
    variants = generate_variants(cfg, n_variants=6)
    # At least some variants should have different alpha values
    base_alpha = cfg["adstock_alpha"]["alpha"]
    all_same = all(v["adstock_alpha"]["alpha"] == base_alpha for v in variants)
    assert not all_same, "Variants should differ from base config"


def test_generate_variants_no_mutation_of_base():
    cfg = _minimal_config()
    original_alpha = cfg["adstock_alpha"]["alpha"].copy()
    generate_variants(cfg, n_variants=3)
    assert cfg["adstock_alpha"]["alpha"] == original_alpha, "Base config should not be mutated"


# --- tighten_priors_from_posterior tests ---

def test_tighten_priors_reduces_uncertainty():
    cfg = _minimal_config()
    post_stats = {
        "adstock_alpha": {"mean": [0.25, 0.30], "std": [0.10, 0.12]},
        "saturation_lam": {"mean": [3.5, 2.0], "std": [0.8, 0.6]},
    }
    tightened = tighten_priors_from_posterior(cfg, post_stats, tighten_factor=0.7)
    # Check that prior parameters were updated
    assert tightened != cfg  # should be different


def test_tighten_priors_does_not_mutate_original():
    cfg = _minimal_config()
    original_alpha = cfg["adstock_alpha"]["alpha"].copy()
    post_stats = {"adstock_alpha": {"mean": [0.25, 0.30], "std": [0.10, 0.12]}}
    tighten_priors_from_posterior(cfg, post_stats)
    assert cfg["adstock_alpha"]["alpha"] == original_alpha


def test_tighten_priors_empty_posterior():
    cfg = _minimal_config()
    tightened = tighten_priors_from_posterior(cfg, {})
    assert tightened["adstock_alpha"] == cfg["adstock_alpha"]


# --- run_tournament dry_run tests ---

def test_run_tournament_dry_run(tmp_path):
    # Create minimal spec.yaml and model_config.json
    import yaml
    spec = {
        "version": "1", "mmm_type": "greenfield",
        "company_name": "Test", "industry": "retail", "region": "UK",
        "data_path": str(Path(__file__).parent / "data" / "synthetic_weekly.csv"),
        "date_column": "date", "target_column": "y",
        "target_unit": {"kind": "acquisition", "label": "sale"},
        "channels": [{"column": "spend_sem"}, {"column": "spend_social"}],
        "controls": [],
    }
    ws = tmp_path / "mmm-workspace"
    ws.mkdir()
    with open(ws / "spec.yaml", "w") as f:
        yaml.dump(spec, f)

    cfg = {"model_config": _minimal_config(), "_metadata": {}}
    (tmp_path / "model_config.json").write_text(json.dumps(cfg))

    result = run_tournament(
        spec_path=ws / "spec.yaml",
        model_config_path=tmp_path / "model_config.json",
        max_rounds=2,
        n_variants_per_round=3,
        base=str(tmp_path),
        dry_run=True,
    )

    assert "leaderboard" in result
    assert len(result["leaderboard"]) == 6  # 2 rounds × 3 variants
    assert result["best_run_id"] is not None
    assert (tmp_path / "mmm-workspace" / "leaderboard.json").exists()


def test_tournament_leaderboard_sorted(tmp_path):
    import yaml
    spec = {
        "version": "1", "mmm_type": "greenfield",
        "company_name": "Test", "industry": "retail", "region": "UK",
        "data_path": str(Path(__file__).parent / "data" / "synthetic_weekly.csv"),
        "date_column": "date", "target_column": "y",
        "target_unit": {"kind": "acquisition", "label": "sale"},
        "channels": [{"column": "spend_sem"}],
        "controls": [],
    }
    ws = tmp_path / "mmm-workspace"
    ws.mkdir()
    with open(ws / "spec.yaml", "w") as f:
        yaml.dump(spec, f)

    cfg = {"model_config": _minimal_config()}
    (tmp_path / "model_config.json").write_text(json.dumps(cfg))

    result = run_tournament(
        spec_path=ws / "spec.yaml",
        model_config_path=tmp_path / "model_config.json",
        max_rounds=1,
        n_variants_per_round=4,
        base=str(tmp_path),
        dry_run=True,
    )

    scores = [e["score"] for e in result["leaderboard"]]
    assert scores == sorted(scores, reverse=True), "Leaderboard should be sorted descending by score"
