"""Card schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperdb.schemas import Card


def test_full_card():
    c = Card(
        family="vla",
        backbone="PaliGemma-3B",
        action_head="flow-matching",
        action_space="joint",
        chunk_size=50,
        control_hz=50.0,
        embodiment=["Franka"],
        data={"hours": 100, "episodes": 903, "source": "OXE"},
        eval={"sim": {"libero": 0.97}, "real": {}},
        open={"weights": True, "code": "https://github.com/x", "data": False},
        compute="96 H100s",
        limits=["struggles with deformable objects"],
    )
    assert c.data.source == "OXE"


def test_nulls_allowed_and_enforced():
    c = Card(family="wam")
    assert c.backbone is None and c.chunk_size is None
    assert c.open.weights is None
    with pytest.raises(ValidationError):
        Card(family="transformer")  # not in the family enum
    with pytest.raises(ValidationError):
        Card(family="vla", control_hz="fast")  # numbers must be numbers
