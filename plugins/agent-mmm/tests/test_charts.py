"""Tests for charts.py: PNGs, the self-contained HTML report, and the summary."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from agent_mmm.charts import PALETTE, _human, build_html, build_report, channel_colors, model_colors
from agent_mmm.compare import compare
from test_results_and_compare import _results


def test_framework_colours_follow_the_entity():
    a, b = _results("pymc-marketing"), _results("meridian", seed=1)
    assert model_colors([a, b]) == model_colors([b, a])
    assert model_colors([a, b])[a.run_id] == PALETTE[0]
    assert model_colors([a, b])[b.run_id] == PALETTE[1]


def test_channel_colours_are_fixed_order():
    c = channel_colors(["a", "b", "c"])
    assert list(c.values()) == PALETTE[:3]


def test_human_numbers():
    assert [_human(v) for v in (1200, 12500, 1.5e6)] == ["1,200", "12.5K", "1.5M"]


def test_build_report_writes_everything(tmp_path):
    rs = [_results("pymc-marketing"), _results("meridian", seed=1)]
    out = build_report(rs, compare(rs), base=tmp_path)
    for paths in out["figures"].values():
        for p in paths:
            assert Path(p).stat().st_size > 5000, p
    assert len(out["figures"]["comparison"]) == 3
    md = Path(out["markdown"]).read_text()
    assert "## Findings" in md and "![fit_holdout]" in md
    assert all((Path(out["markdown"]).parent / p.split("(")[1].rstrip(")")).exists()
               for p in md.splitlines() if p.startswith("!["))


def test_single_model_report(tmp_path):
    out = build_report([_results("pymc-marketing")], base=tmp_path)
    assert "comparison" not in out["figures"]
    assert Path(out["html"]).exists()


def test_html_is_self_contained():
    rs = [_results("pymc-marketing"), _results("meridian", seed=1)]
    page = build_html(rs, compare(rs))
    # plotly.js carries a CDN URL for map data it never fetches here; what matters is
    # that nothing is loaded from outside the file.
    assert "<script src=" not in page and 'src="http' not in page and "<link " not in page
    assert page.count("<script") >= 2          # inlined plotly.js + chart scripts


def test_html_escapes_user_text():
    r = _results("pymc-marketing")
    evil = "<img src=x onerror=alert(1)>"
    r.notes.append(evil)
    r.target_label = "<script>alert('t')</script>"
    page = build_html([r])
    assert evil not in page and "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<script>alert('t')</script>" not in page


def test_unconverged_model_is_flagged_on_its_tile():
    rs = [_results("pymc-marketing"), _results("meridian", ess=50, seed=1)]
    page = build_html(rs, compare(rs))
    assert "not converged: min ESS 50 &lt; 400" in page


def test_empty_results_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_report([], base=tmp_path)
