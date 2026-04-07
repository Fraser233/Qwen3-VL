#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""Standalone trainer for Qwen3-VL VIVID distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import json
import shutil

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import DistillConfig


def _to_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n == "bf16":
        return torch.bfloat16
    if n == "fp16":
        return torch.float16
    if n == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


@dataclass
class DistillBundle:
    teacher: torch.nn.Module
    student: torch.nn.Module
    processor: AutoProcessor


class VividQwenDistillationTrainer:
    """Minimal KL distillation trainer for Qwen3-VL teacher/student."""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.bundle = self._build_models()

        self.output_dir = Path(self.config.training.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.best_dir = self.output_dir / "best_checkpoint"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        mode = str(self.config.training.best_metric_mode).lower()
        if mode not in {"min", "max"}:
            raise ValueError("training.best_metric_mode must be 'min' or 'max'")
        self.best_metric_mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoint_path: Path | None = None

    @property
    def student(self) -> torch.nn.Module:
        return self.bundle.student

    @property
    def processor(self) -> AutoProcessor:
        return self.bundle.processor

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return underlying model for wrappers such as DDP/DataParallel."""
        return getattr(model, "module", model)

    def _build_models(self) -> DistillBundle:
        dtype = _to_dtype(self.config.model.torch_dtype)

        teacher = AutoModelForImageTextToText.from_pretrained(
            self.config.model.teacher_model_name_or_path,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        student = AutoModelForImageTextToText.from_pretrained(
            self.config.model.student_model_name_or_path,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model.student_model_name_or_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Freeze teacher.
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

        # Apply VIVID patch to student vision blocks.
        from vivid.qwen3_patch import apply_vivid_qwen3_vision_patch

        apply_vivid_qwen3_vision_patch(
            student,
            num_anchors=self.config.vivid.anchors,
            topk=self.config.vivid.topk,
            enabled=self.config.vivid.enabled,
        )

        return DistillBundle(teacher=teacher, student=student, processor=processor)

    def kl_distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        t = self.config.training.kl_temperature
        s = F.log_softmax(student_logits / t, dim=-1)
        p = F.softmax(teacher_logits / t, dim=-1)
        return F.kl_div(s, p, reduction="batchmean") * (t * t)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        student = self.bundle.student
        teacher = self.bundle.teacher

        with torch.no_grad():
            t_out = teacher(**batch)
        s_out = student(**batch)

        loss = self.kl_distill_loss(s_out.logits, t_out.logits)
        return loss * self.config.training.kl_weight

    def save_checkpoint(
        self,
        name: str,
        step: int,
        epoch: int,
        metric: float | None = None,
        is_best: bool = False,
    ) -> Path:
        """Save student checkpoint + processor + metadata.

        Args:
            name: Subfolder name under `output_dir/checkpoints`.
            step: Global training step.
            epoch: Epoch index.
            metric: Validation/distillation metric for bookkeeping.
            is_best: Whether this checkpoint should be mirrored as best.
        """
        ckpt_dir = self.checkpoints_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self._unwrap_model(self.student)
        model_to_save.save_pretrained(str(ckpt_dir), safe_serialization=True)
        self.processor.save_pretrained(str(ckpt_dir))

        state = {
            "step": step,
            "epoch": epoch,
            "metric": metric,
            "is_best": is_best,
            "teacher_model_name_or_path": self.config.model.teacher_model_name_or_path,
            "student_model_name_or_path": self.config.model.student_model_name_or_path,
            "vivid": {
                "enabled": self.config.vivid.enabled,
                "anchors": self.config.vivid.anchors,
                "topk": self.config.vivid.topk,
            },
        }
        with (ckpt_dir / "distill_state.json").open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if is_best:
            if self.best_dir.exists():
                shutil.rmtree(self.best_dir)
            shutil.copytree(ckpt_dir, self.best_dir)
            self.best_checkpoint_path = ckpt_dir

            best_state = {
                "best_metric": metric,
                "best_checkpoint": str(ckpt_dir),
                "best_checkpoint_name": name,
                "step": step,
                "epoch": epoch,
            }
            with (self.output_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
                json.dump(best_state, f, indent=2)

        return ckpt_dir

    def maybe_save_best_checkpoint(self, metric: float, step: int, epoch: int) -> bool:
        """Save checkpoint if `metric` improves over current best.

        Returns:
            True if a new best checkpoint is saved.
        """
        improved = metric < self.best_metric if self.best_metric_mode == "min" else metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.save_checkpoint(
                name=f"checkpoint-best-step{step}",
                step=step,
                epoch=epoch,
                metric=metric,
                is_best=True,
            )
            return True
        return False

    def save_last_checkpoint(self, step: int, epoch: int, metric: float | None = None) -> Path:
        """Save latest checkpoint snapshot."""
        return self.save_checkpoint(
            name=f"checkpoint-last-step{step}",
            step=step,
            epoch=epoch,
            metric=metric,
            is_best=False,
        )
