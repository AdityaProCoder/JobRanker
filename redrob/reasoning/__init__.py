"""Deterministic, evidence-driven reasoning strings for each top-100 candidate.

No LLM is used at submission time. The reasoning text is composed from a
small library of templates that cite specific facts from the candidate's
profile and connect them to JD requirements.
"""
from .template import generate_reasoning

__all__ = ["generate_reasoning"]
