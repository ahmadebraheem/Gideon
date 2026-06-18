"""Gideon — a local, zero-cloud automated ML pipeline.

A user drops a CSV into ``inbox/`` and a chain of seven stateless specialist
agents (orchestrated by the boss) cleans the data, engineers features, trains a
model, evaluates it, deploys it, and refreshes a live Streamlit dashboard.

Agents never share in-memory state: every agent reads its input(s) from, and
writes its output to, the ``artifacts/`` folder.
"""

__version__ = "0.1.0"
