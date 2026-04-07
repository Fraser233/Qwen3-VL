#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""CLI entrypoint for standalone Qwen3-VL VIVID distillation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import json

import torch
from transformers import set_seed

# Allow importing sibling `vivid` package from Qwen3-VL root.
QWEN_ROOT = Path(__file__).resolve().parents[1]
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from vivid_distillation.config import load_distill_config
from vivid_distillation.trainer import VividQwenDistillationTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL VIVID distillation runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--save_init_checkpoint",
        action="store_true",
        default=False,
        help="Save an initial checkpoint (step=0) to validate checkpoint writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_distill_config(args.config)

    set_seed(cfg.runtime.seed)
    trainer = VividQwenDistillationTrainer(cfg)

    # Persist run config snapshot for reproducibility.
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "distill_config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": cfg.model.__dict__,
                "vivid": cfg.vivid.__dict__,
                "training": cfg.training.__dict__,
                "runtime": cfg.runtime.__dict__,
            },
            f,
            indent=2,
        )

    if args.save_init_checkpoint:
        init_path = trainer.save_last_checkpoint(step=0, epoch=0, metric=None)
        logger.info("Saved init checkpoint to %s", init_path)

    logger.info("Initialized distillation trainer.")
    logger.info("Teacher: %s", cfg.model.teacher_model_name_or_path)
    logger.info("Student: %s", cfg.model.student_model_name_or_path)
    logger.info("VIVID enabled=%s anchors=%s topk=%s", cfg.vivid.enabled, cfg.vivid.anchors, cfg.vivid.topk)

    logger.info("Scaffold created. Plug your dataset/batching loop and call `train_step()`.")
    logger.info("Example: loss = trainer.train_step(batch)")
    logger.info(
        "Best-checkpoint API: call `trainer.maybe_save_best_checkpoint(metric=eval_loss, step=global_step, epoch=epoch)` after each eval."
    )
    logger.info(
        "Last-checkpoint API: call `trainer.save_last_checkpoint(step=global_step, epoch=epoch, metric=eval_loss)` at save intervals."
    )

    if torch.cuda.is_available():
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
