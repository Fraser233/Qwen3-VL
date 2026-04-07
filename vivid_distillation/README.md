# Qwen3-VL VIVID Distillation

This folder contains a **new standalone** distillation pipeline for Qwen3-VL + VIVID.

- It is **inspired by** the existing embedding distillation design.
- It **does not import** or depend on scripts in [rsu-side/src/models/embedding_training](../../../embedding_training).

## Structure

- [config.py](config.py): dataclass config + YAML loader
- [trainer.py](trainer.py): teacher/student distillation trainer
- [run_distill.py](run_distill.py): CLI entrypoint
- [config.example.yaml](config.example.yaml): starter config

## Quick start

1. Copy [config.example.yaml](config.example.yaml) to your run config.
2. Adjust model paths and hyperparameters.
3. Run:

python rsu-side/src/models/Qwen3-VL/vivid_distillation/run_distill.py --config /path/to/config.yaml

## Current scope

This initial version focuses on:
- loading teacher + student Qwen3-VL
- applying VIVID patch to the student vision tower
- running logits KL distillation over language outputs

You can extend it with hidden-state distillation and vision-feature losses next.

## Checkpoint saving (best + last)

The trainer now supports explicit checkpoint APIs:

- `trainer.save_last_checkpoint(step, epoch, metric)`
- `trainer.maybe_save_best_checkpoint(metric, step, epoch)`

Behavior:

- Last checkpoints are written under:
	- `training.output_dir/checkpoints/checkpoint-last-step{N}`
- Best checkpoint is mirrored to:
	- `training.output_dir/best_checkpoint`
- Best checkpoint metadata is written to:
	- `training.output_dir/best_checkpoint.json`

Metric direction is configured by:

- `training.best_metric_mode: "min"` for loss (default)
- `training.best_metric_mode: "max"` for score metrics

For a checkpoint-writing smoke test:

python rsu-side/src/models/Qwen3-VL/vivid_distillation/run_distill.py --config /path/to/config.yaml --save_init_checkpoint
