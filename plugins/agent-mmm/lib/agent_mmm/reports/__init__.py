"""Stakeholder reporting for agent-mmm. Each sub-module generates one persona report."""

from agent_mmm.reports.cmo import generate_cmo_report
from agent_mmm.reports.cfo import generate_cfo_report
from agent_mmm.reports.mops import generate_mops_report
from agent_mmm.reports.ds import generate_ds_report

__all__ = ["generate_cmo_report", "generate_cfo_report", "generate_mops_report", "generate_ds_report"]
