# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

import torch


dependencies = ["torch", "einops", "jaxtyping"]


def fpt_base(*, pretrained: bool = True, **kwargs):
    """
    Standard Flow Poke Transformer.
    """
    from flow_poke.model import FlowPokeTransformer_Base

    model = FlowPokeTransformer_Base(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/CompVis/flow-poke-transformer/resolve/main/flow_poke_open_set_base.pt",
            map_location="cpu",
        )
        model.load_state_dict(state_dict["model"])
    model.requires_grad_(False)
    model.eval()
    return model
