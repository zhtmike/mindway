from typing import Optional, Tuple

import mindspore as ms
from mindspore import nn, mint, ops
from transformers.utils import logging

logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: Optional[nn.Cell], # aligned with torch
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None, # aligned with torch
    softcap: Optional[float] = None,  # a
    **kwargs,
) -> Tuple[ms.Tensor, None]:

    if sliding_window is not None:
        raise NotImplementedError(
            "Sliding window is not supported in Mindspore yet. Please use `sliding_window=None`."
        )
    if softcap is not None:
        softcap = None
        logger.warning(
            "Softcap is not supported in Mindspore yet. Ignore it."
        )
    # This is before the transpose
    seq_len = query.shape[2]  # BNSD, N: num_head, S: seq_len, D: head_dim
    num_head = query.shape[1]
    input_layout = "BNSD"

    # In MindSpore, False indicates retention and True indicates discard, Which is opposite to PyTorch
    seq_len_key = key.shape[2]
    if attention_mask is not None:
        attention_mask = mint.logical_not(attention_mask) if attention_mask.dtype == ms.bool_ else attention_mask.bool()
        attention_mask = mint.broadcast_to(
            attention_mask, (attention_mask.shape[0], attention_mask.shape[1], seq_len, seq_len_key))[:, -1, :, :]


    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        keep_prob=1.0 - dropout,
        scale_value=scaling,
        **kwargs,
    )

    attn_output = ops.flash_attention_score(
        query,
        key,
        value,
        head_num=num_head,
        attn_mask=attention_mask,
        keep_prob=1.0 - dropout,
        scalar_value=scaling,
        input_layout=input_layout,
    )

    return attn_output, None
