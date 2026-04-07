#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""Standalone VIVID distillation package for Qwen3-VL."""

from .config import DistillConfig, load_distill_config
from .trainer import VividQwenDistillationTrainer

__all__ = ["DistillConfig", "load_distill_config", "VividQwenDistillationTrainer"]
