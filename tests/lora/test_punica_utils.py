# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.lora.layers import LoRAMapping
from vllm.lora.punica_wrapper.utils import convert_mapping


pytestmark = pytest.mark.skip_global_cleanup


def test_convert_mapping_maps_noncontiguous_lora_ids() -> None:
    """Map base-model and noncontiguous LoRA IDs to Punica indices."""
    mapping = LoRAMapping(
        index_mapping=(0, 42, 7, -1, 42),
        prompt_mapping=(42, 0, 7, -1),
        is_prefill=True,
    )

    (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        indices_len,
    ) = convert_mapping(
        mapping=mapping,
        lora_index_to_id=[None, 42, 7, None],
        max_loras=4,
        vocab_size=100,
        extra_vocab_size=10,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        base_indices,
        torch.tensor([-1, 1, 2, -1, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        sampler_indices,
        torch.tensor([1, -1, 2, -1], dtype=torch.long),
    )
    torch.testing.assert_close(
        sampler_indices_padded,
        torch.tensor([4, 13, 10, 15], dtype=torch.long),
    )
    torch.testing.assert_close(
        embeddings_indices,
        torch.tensor(
            [
                [0, 10, 20, 0, 10],
                [0, 110, 220, 0, 110],
            ],
            dtype=torch.long,
        ),
    )
    assert indices_len == [5, 4, 4, 5]


def test_convert_mapping_rejects_unknown_positive_lora_id() -> None:
    mapping = LoRAMapping(index_mapping=(99,), prompt_mapping=(99,))

    with pytest.raises(KeyError, match="99"):
        convert_mapping(
            mapping=mapping,
            lora_index_to_id=[None, 42, 7],
            max_loras=3,
            vocab_size=100,
            extra_vocab_size=10,
            device=torch.device("cpu"),
        )
