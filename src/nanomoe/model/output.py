from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from torch import Tensor


class ModelOutput(BaseModel):
    """Validated model outputs for training and sampling."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    logits: Tensor
    aux_loss: Tensor | float
    past_key_values: list[tuple[Tensor, Tensor] | None] | None = None
    hidden_states: Tensor | None = None
    router_logits: list[Tensor] | None = None
    router_expert_indices: list[Tensor] | None = None
