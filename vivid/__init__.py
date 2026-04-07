#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""VIVID utilities for Qwen3-VL."""

from .qwen3_patch import (
    Qwen3VIVIDVisionAttention,
    apply_vivid_qwen3_vision_patch,
    count_vivid_qwen3_blocks,
    is_vivid_qwen3_vision_patched,
)

__all__ = [
    "Qwen3VIVIDVisionAttention",
    "apply_vivid_qwen3_vision_patch",
    "count_vivid_qwen3_blocks",
    "is_vivid_qwen3_vision_patched",
]
