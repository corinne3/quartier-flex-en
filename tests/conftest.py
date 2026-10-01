"""Shared fixtures: a small OFFLINE scenario (no network, reproducible results)."""
import pytest

from quartierflex.config import Scenario


@pytest.fixture(scope="session")
def scn():
    return Scenario(offline=True, days=3, history_days=21)


@pytest.fixture(scope="session")
def ctx(scn):
    from quartierflex.assessment import load_inputs

    return load_inputs(scn)
