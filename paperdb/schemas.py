"""Card schema: structured per-paper extraction. null = not stated, never inferred."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Data(BaseModel):
    hours: float | None = None
    episodes: int | None = None
    source: Literal["in-house", "OXE", "human-video", "mixed"] | None = None


class Eval(BaseModel):
    sim: dict[str, float] = {}
    real: dict[str, float] = {}


class Open(BaseModel):
    weights: bool | None = None
    code: str | None = None
    data: bool | None = None


class Card(BaseModel):
    family: Literal["vla", "wam", "diffusion-policy", "rl", "world-model", "dataset", "benchmark"]
    backbone: str | None = None  # e.g. "PaliGemma-3B"
    action_head: (
        Literal["flow-matching", "diffusion", "fast-tokens", "ar-bins", "latent", "mlp"] | None
    ) = None
    action_space: Literal["joint", "ee-delta", "ee-abs", "latent"] | None = None
    chunk_size: int | None = None
    control_hz: float | None = None
    embodiment: list[str] = []
    data: Data = Data()
    eval: Eval = Eval()
    open: Open = Open()
    compute: str | None = None
    limits: list[str] = []  # failure modes the paper itself admits
