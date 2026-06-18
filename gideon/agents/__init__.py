"""The seven specialist agents.

Each module owns exactly one pipeline stage, reads its input(s) from the
``artifacts/`` folder, and writes a single primary output back there. They hold
no shared in-memory state and can each be run standalone for debugging, e.g.::

    python -m gideon.agents.cleaner
"""
