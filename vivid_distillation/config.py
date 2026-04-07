#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""Config objects for standalone Qwen3-VL VIVID distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    teacher_model_name_or_path: str
    student_model_name_or_path: str
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    device_map: str = "auto"


@dataclass
class VividConfig:
    enabled: bool = True
    anchors: int = 256
    topk: int = 8


@dataclass
class TrainingConfig:
    output_dir: str
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    kl_temperature: float = 1.0
    kl_weight: float = 1.0
    best_metric_mode: str = "min"  # min for loss, max for score


@dataclass
class RuntimeConfig:
    seed: int = 42
    max_new_tokens: int = 64


@dataclass
class DistillConfig:
    model: ModelConfig
    vivid: VividConfig
    training: TrainingConfig
    runtime: RuntimeConfig



def _require(d: Dict[str, Any], key: str, section: str):
    if key not in d:
        raise KeyError(f"Missing required key '{section}.{key}'")
    return d[key]


def load_distill_config(config_path: str | Path) -> DistillConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = _require(raw, "model", "root")
    vivid_raw = _require(raw, "vivid", "root")
    training_raw = _require(raw, "training", "root")
    runtime_raw = raw.get("runtime", {})

    return DistillConfig(
        model=ModelConfig(**model_raw),
        vivid=VividConfig(**vivid_raw),
        training=TrainingConfig(**training_raw),
        runtime=RuntimeConfig(**runtime_raw),
    )
