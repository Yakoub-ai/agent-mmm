"""Stakeholder reporting for agent-mmm.

Each persona module generates one written report; ``deck`` generates the
presentation argument for the same audiences.
"""

from agent_mmm.reports.cmo import generate_cmo_report
from agent_mmm.reports.cfo import generate_cfo_report
from agent_mmm.reports.mops import generate_mops_report
from agent_mmm.reports.ds import generate_ds_report
from agent_mmm.reports.deck import (
    AUDIENCES,
    ChartSpec,
    DeckSpec,
    Slide,
    build_deck,
    render_deck_markdown,
    write_deck,
)

__all__ = [
    "generate_cmo_report",
    "generate_cfo_report",
    "generate_mops_report",
    "generate_ds_report",
    "AUDIENCES",
    "ChartSpec",
    "DeckSpec",
    "Slide",
    "build_deck",
    "render_deck_markdown",
    "write_deck",
]
