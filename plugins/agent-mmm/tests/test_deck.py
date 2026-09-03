"""Deck specs must carry an argument — a claim, its evidence, its limits — not slide titles."""
import json

import pytest

from agent_mmm.reports.deck import (
    AUDIENCES,
    build_deck,
    render_deck_markdown,
    write_deck,
)

CONTRIBUTIONS = {
    "tv": {"contribution_share": 0.31, "roas": 1.4, "roas_low": 0.6, "roas_high": 2.6},
    "search_brand": {"contribution_share": 0.24, "roas": 8.2, "roas_low": 3.0, "roas_high": 14.0},
    "social": {"contribution_share": 0.18, "roas": 2.1, "roas_low": 1.9, "roas_high": 2.3},
}
REALLOCATION = [
    {"channel": "tv", "current_share": 0.40, "proposed_share": 0.34, "rationale": "saturated"},
    {"channel": "social", "current_share": 0.20, "proposed_share": 0.26, "rationale": "headroom"},
]


def deck(audience, **kw):
    params = dict(company="Acme", tier="PASS", baseline_share=0.62,
                  contributions=CONTRIBUTIONS, reallocation=REALLOCATION)
    params.update(kw)
    return build_deck(audience, **params)


class TestStructure:
    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_every_audience_builds(self, audience):
        d = deck(audience)
        assert d.slides
        assert d.framing["audience"]

    def test_unknown_audience_raises(self):
        with pytest.raises(ValueError, match="audience must be one of"):
            build_deck("board")

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_headlines_assert_something(self, audience):
        """A headline that is a category label ('Channel Performance') makes the
        audience do the work of finding the point."""
        banned = {"channel performance", "results", "overview", "appendix", "analysis", "summary"}
        for slide in deck(audience).slides:
            assert slide.headline.lower().strip() not in banned
            assert len(slide.headline.split()) >= 4, slide.headline

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_every_slide_has_speaker_notes(self, audience):
        for slide in deck(audience).slides:
            assert slide.speaker_notes.strip(), slide.headline

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_every_slide_anticipates_a_question_and_answers_it(self, audience):
        for slide in deck(audience).slides:
            assert slide.anticipated_question.strip(), slide.headline
            assert slide.answer.strip(), slide.headline

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_every_chart_states_its_takeaway(self, audience):
        """A chart nobody can state the point of should not be in the deck."""
        for slide in deck(audience).slides:
            if slide.chart:
                assert slide.chart.takeaway.strip(), slide.headline

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_every_deck_ends_with_an_ask(self, audience):
        assert deck(audience).slides[-1].headline == "What we are asking for"

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_confidence_is_stated_on_slides(self, audience):
        assert any(s.confidence for s in deck(audience).slides)


class TestHonesty:
    def test_no_experiment_produces_a_caveat_and_a_roadmap_slide(self):
        d = deck("cmo", experiments=None)
        assert any("No incrementality experiment" in c for c in d.caveats)
        assert any("Nothing here has been tested" in s.headline for s in d.slides)

    def test_experiments_replace_the_roadmap_slide(self):
        d = deck("cmo", experiments=[{"channel": "tv", "design": "geo_holdout", "result": "+4.2%"}])
        assert any("anchor this model" in s.headline for s in d.slides)
        assert not any("Nothing here has been tested" in s.headline for s in d.slides)
        assert not any("No incrementality experiment" in c for c in d.caveats)

    def test_failed_validation_leads_the_caveats(self):
        d = deck("cmo", tier="FAIL")
        assert "did not pass validation" in d.caveats[0]

    def test_low_baseline_is_called_out(self):
        d = deck("cmo", baseline_share=0.15)
        assert any("baseline accounts for only" in c for c in d.caveats)

    def test_wide_intervals_get_their_own_slide_at_low_confidence(self):
        """The most valuable slide in the deck and the one most often cut."""
        d = deck("cmo")
        slide = next(s for s in d.slides if "not really been measured" in s.headline)
        assert slide.confidence == "low"
        assert "tv" in slide.headline or "search_brand" in slide.headline

    def test_precise_channels_do_not_appear_as_unmeasured(self):
        d = deck("cmo")
        slide = next((s for s in d.slides if "not really been measured" in s.headline), None)
        assert slide is not None
        # social's interval is 1.9-2.3: measured, so it must not be listed.
        assert not any("social" in b for b in slide.body)

    def test_confidence_tracks_the_validation_tier(self):
        assert deck("cmo", tier="PASS").slides[0].confidence == "high"
        assert deck("cmo", tier="WARN").slides[0].confidence == "medium"
        assert deck("cmo", tier="FAIL").slides[0].confidence == "low"

    def test_cfo_deck_states_what_the_model_cannot_do(self):
        assert any("cannot tell you" in s.headline for s in deck("cfo").slides)

    def test_ds_deck_leads_with_weakness(self):
        assert "weakest" in deck("ds").slides[0].headline

    def test_missing_inputs_degrade_rather_than_invent(self):
        d = build_deck("cmo")
        assert d.slides
        assert any("No incrementality experiment" in c for c in d.caveats)


class TestAudienceDifferentiation:
    def test_decks_differ_by_audience(self):
        headlines = {a: [s.headline for s in deck(a).slides] for a in AUDIENCES}
        assert headlines["cmo"] != headlines["cfo"] != headlines["ds"]

    def test_cfo_gets_marginal_vs_average(self):
        assert any("margin" in s.headline.lower() for s in deck("cfo").slides)

    def test_mops_gets_executable_line_items(self):
        d = deck("mops")
        assert any("line by line" in s.headline for s in d.slides)
        # Contracted commitments are the usual reason a reallocation cannot be
        # executed, so the deck has to raise them rather than discover them later.
        assert any(
            "commit" in (s.anticipated_question + s.answer + s.speaker_notes).lower()
            for s in d.slides
        )

    def test_mops_asks_for_data_hygiene(self):
        assert any("naming" in b for s in deck("mops").slides for b in s.body)

    def test_ds_gets_identifiability(self):
        assert any("under-identified" in s.headline for s in deck("ds").slides)

    def test_next_steps_differ_by_audience(self):
        asks = {a: deck(a).slides[-1].body for a in AUDIENCES}
        assert len({tuple(v) for v in asks.values()}) == len(AUDIENCES)


class TestRendering:
    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_renders_markdown(self, audience):
        md = render_deck_markdown(deck(audience))
        assert md.startswith("# ")
        assert "## Slide 1 — " in md
        assert "**Speaker notes.**" in md
        assert "**They will ask:**" in md

    @pytest.mark.parametrize("audience", AUDIENCES)
    def test_json_serialisable(self, audience):
        json.dumps(deck(audience).to_dict())

    def test_writes_to_the_workspace(self, tmp_path):
        path = write_deck(deck("cmo"), base=tmp_path)
        assert path.name == "deck_cmo.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# ")

    def test_open_questions_are_carried_through(self):
        d = deck("cmo", roadmap_questions=["Who signs off the revenue at risk?"])
        md = render_deck_markdown(d)
        assert "Open questions for the room" in md
        assert "Who signs off" in md
