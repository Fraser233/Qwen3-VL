import os
import sys
import json
import argparse
import importlib.util
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
import torch
import warnings
import traceback

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image
from eval_utils import build_judge, eval_single_sample, MATH_V_acc


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _should_sample(args) -> bool:
    return not (float(args.temperature) <= 0.02 and int(args.top_k) == 1)


def _build_generate_kwargs(args, processor) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": _should_sample(args),
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return kwargs


def _reset_dynamic_token_counters(visual_mod) -> None:
    if visual_mod is None:
        return
    for attr in (
        "_diffrate_last_alive_tokens",
        "_diffrate_last_total_tokens",
        "_diffrate_last_layer_token_stats",
        "_dymu_last_alive_tokens",
        "_dymu_last_total_tokens",
        "_dymu_last_layer_token_stats",
        "_dynamic_last_alive_tokens",
        "_dynamic_last_total_tokens",
        "_dynamic_last_layer_token_stats",
        "_visiontrim_last_alive_tokens",
        "_visiontrim_last_total_tokens",
        "_visiontrim_last_layer_token_stats",
        "_visiontrim_last_llm_prefill_tokens",
        "_visiontrim_last_dense_prefill_tokens",
        "_zooprune_last_alive_tokens",
        "_zooprune_last_total_tokens",
        "_zooprune_last_layer_token_stats",
        "_zooprune_last_llm_prefill_tokens",
        "_zooprune_last_dense_prefill_tokens",
        "_diffrate_last_llm_prefill_tokens",
        "_diffrate_last_dense_prefill_tokens",
        "_vivid_last_alive_tokens",
        "_vivid_last_total_tokens",
        "_vivid_last_layer_token_stats",
        "_vivid_last_llm_prefill_tokens",
        "_vivid_last_dense_prefill_tokens",
        "_last_compression_overhead_ms",
        "_dymu_last_compression_overhead_ms",
        "_diffrate_last_compression_overhead_ms",
        "_dynamic_last_compression_overhead_ms",
        "_visiontrim_last_compression_overhead_ms",
        "_zooprune_last_compression_overhead_ms",
        "_vivid_last_compression_overhead_ms",
    ):
        if hasattr(visual_mod, attr):
            setattr(visual_mod, attr, [] if attr.endswith("_layer_token_stats") else None)


def _read_dynamic_alive_tokens(visual_mod) -> int | None:
    if visual_mod is None:
        return None
    for attr in (
        "_dymu_last_alive_tokens",
        "_diffrate_last_alive_tokens",
        "_dynamic_last_alive_tokens",
        "_visiontrim_last_alive_tokens",
        "_zooprune_last_alive_tokens",
        "_vivid_last_alive_tokens",
    ):
        value = getattr(visual_mod, attr, None)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


def _read_dynamic_llm_prefill_tokens(model, visual_mod, fallback: int) -> int:
    candidates = [
        visual_mod,
        getattr(model, "model", None),
        model,
    ]
    attrs = (
        "_visiontrim_last_llm_prefill_tokens",
        "_zooprune_last_llm_prefill_tokens",
        "_diffrate_last_llm_prefill_tokens",
        "_dymu_last_llm_prefill_tokens",
        "_dynamic_last_llm_prefill_tokens",
        "_vivid_last_llm_prefill_tokens",
    )
    for obj in candidates:
        if obj is None:
            continue
        for attr in attrs:
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return int(fallback)


def _read_dynamic_layer_token_stats(visual_mod) -> list[dict]:
    if visual_mod is None:
        return []
    for attr in (
        "_dymu_last_layer_token_stats",
        "_diffrate_last_layer_token_stats",
        "_dynamic_last_layer_token_stats",
        "_visiontrim_last_layer_token_stats",
        "_zooprune_last_layer_token_stats",
        "_vivid_last_layer_token_stats",
    ):
        value = getattr(visual_mod, attr, None)
        if isinstance(value, list) and value:
            return [dict(x) for x in value if isinstance(x, dict)]
    return []


def _read_dynamic_compression_overhead_ms(model, visual_mod) -> float:
    candidates = [
        visual_mod,
        getattr(model, "model", None),
        model,
    ]
    attrs = (
        "_last_compression_overhead_ms",
        "_dymu_last_compression_overhead_ms",
        "_diffrate_last_compression_overhead_ms",
        "_dynamic_last_compression_overhead_ms",
        "_visiontrim_last_compression_overhead_ms",
        "_zooprune_last_compression_overhead_ms",
        "_vivid_last_compression_overhead_ms",
    )
    for obj in candidates:
        if obj is None:
            continue
        for attr in attrs:
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return 0.0


def _merge_layer_token_stats(per_layer: dict, stats: list[dict], fallback_full_tokens: int) -> None:
    for item in stats:
        try:
            layer_id = int(item.get("layer"))
        except Exception:
            continue
        key = f"vision_block_{layer_id}"
        entry = per_layer.setdefault(
            key,
            {
                "family": "vision_block",
                "layer": layer_id,
                "seconds": 0.0,
                "calls": 0,
            },
        )
        tokens_in = int(item.get("tokens_in", 0) or 0)
        tokens_out = int(item.get("tokens_out", 0) or 0)
        full_tokens = int(item.get("full_tokens", fallback_full_tokens) or fallback_full_tokens or 0)
        entry["tokens_in"] = int(entry.get("tokens_in", 0)) + tokens_in
        entry["tokens_out"] = int(entry.get("tokens_out", 0)) + tokens_out
        entry["full_tokens"] = int(entry.get("full_tokens", 0)) + full_tokens
        entry["compressed_calls"] = int(entry.get("compressed_calls", 0)) + (1 if bool(item.get("compressed", False)) else 0)
        if entry["tokens_in"] > 0:
            entry["local_keep_ratio"] = float(entry["tokens_out"]) / float(entry["tokens_in"])
        if entry["full_tokens"] > 0:
            entry["cumulative_keep_ratio"] = float(entry["tokens_out"]) / float(entry["full_tokens"])


def _resolve_model_dtype():
    name = os.environ.get("QWEN3_EVAL_MODEL_DTYPE", "bfloat16").strip().lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported QWEN3_EVAL_MODEL_DTYPE={name!r}; use bfloat16, float16, or float32")
    return aliases[name]


def _count_visual_tokens_from_grid(grid_tensor):
    if grid_tensor is None or not torch.is_tensor(grid_tensor) or grid_tensor.numel() == 0:
        return 0
    g = grid_tensor.detach().to("cpu")
    if g.ndim == 1:
        if g.shape[0] >= 3:
            return int(g[0].item() * g[1].item() * g[2].item())
        return 0
    return int((g[:, 0] * g[:, 1] * g[:, 2]).sum().item())


def _count_visual_tokens_from_model_inputs(model_inputs):
    return int(
        _count_visual_tokens_from_grid(model_inputs.get("image_grid_thw"))
        + _count_visual_tokens_from_grid(model_inputs.get("video_grid_thw"))
    )


def _extract_token_count_from_obj(obj):
    if torch.is_tensor(obj):
        if obj.ndim == 2:
            return int(obj.shape[0])
        if obj.ndim >= 3:
            return int(np.prod(obj.shape[:-1]))
        return None
    if isinstance(obj, (list, tuple)):
        for x in obj:
            tok = _extract_token_count_from_obj(x)
            if tok is not None:
                return tok
        return None
    if isinstance(obj, dict):
        for x in obj.values():
            tok = _extract_token_count_from_obj(x)
            if tok is not None:
                return tok
        return None
    return None


def _install_component_timers(model):
    timers = {
        "input_embedding_sec": 0.0,
        "vision_encoder_sec": 0.0,
        "projection_sec": 0.0,
        "llm_prefill_sec": 0.0,
        "llm_decode_sec": 0.0,
    }
    counts = {
        "input_embedding_calls": 0,
        "vision_encoder_calls": 0,
        "projection_calls": 0,
        "llm_prefill_calls": 0,
        "llm_decode_calls": 0,
    }
    token_counters = {
        "projection_input_tokens": 0,
        "projection_input_tokens_calls": 0,
    }
    per_layer = {}
    handles = []

    def _add_hooks(module, key_time, key_count, token_counter_key=None):
        if module is None:
            return

        state = {"t0": None}

        def _pre_hook(_m, _args):
            _sync_if_cuda()
            state["t0"] = time.time()
            if token_counter_key is not None:
                tok = _extract_token_count_from_obj(_args)
                if tok is not None:
                    token_counters[token_counter_key] += int(tok)
                    token_counters[f"{token_counter_key}_calls"] += 1

        def _post_hook(_m, _args, _out):
            _sync_if_cuda()
            if state["t0"] is not None:
                timers[key_time] += time.time() - state["t0"]
                counts[key_count] += 1
                state["t0"] = None

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))

    def _add_layer_hooks(module, family, layer_id, initialize=True):
        if module is None:
            return
        key = f"{family}_{layer_id}"
        if initialize or key not in per_layer:
            per_layer[key] = {
                "family": family,
                "layer": int(layer_id),
                "seconds": 0.0,
                "calls": 0,
            }
        state = {"t0": None}

        def _pre_hook(_m, _args):
            _sync_if_cuda()
            state["t0"] = time.time()

        def _post_hook(_m, _args, _out):
            _sync_if_cuda()
            if state["t0"] is not None:
                per_layer[key]["seconds"] += time.time() - state["t0"]
                per_layer[key]["calls"] += 1
                state["t0"] = None

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))

    input_emb = None
    try:
        input_emb = model.get_input_embeddings()
    except Exception:
        input_emb = None
    _add_hooks(input_emb, "input_embedding_sec", "input_embedding_calls")

    visual = getattr(getattr(model, "model", None), "visual", None) or getattr(model, "visual", None)
    _add_hooks(visual, "vision_encoder_sec", "vision_encoder_calls")
    split_vision_blocks = os.environ.get("QWEN3_PATCH_IMPL", "").strip().lower() in {
        "diffrate",
        "dymu",
        "visiontrim",
        "zooprune",
        "zoo_prune",
        "zoo-prune",
        "vivid",
    }
    for layer_id, block in enumerate(getattr(visual, "blocks", []) or []):
        if split_vision_blocks and (getattr(block, "attn", None) is not None or getattr(block, "mlp", None) is not None):
            _add_layer_hooks(getattr(block, "attn", None), "vision_block", layer_id)
            _add_layer_hooks(getattr(block, "mlp", None), "vision_block", layer_id, initialize=False)
        else:
            _add_layer_hooks(block, "vision_block", layer_id)

    projector = None
    model_core = getattr(model, "model", None)
    proj_candidates = [
        getattr(model_core, "multi_modal_projector", None),
        getattr(model_core, "mm_projector", None),
        getattr(model_core, "vision_projector", None),
        getattr(getattr(model_core, "visual", None), "merger", None),
        getattr(getattr(model, "visual", None), "merger", None),
    ]
    for cand in proj_candidates:
        if cand is not None:
            projector = cand
            break
    _add_hooks(projector, "projection_sec", "projection_calls", token_counter_key="projection_input_tokens")

    language_model = (
        getattr(model_core, "language_model", None)
        or getattr(model_core, "model", None)
        or getattr(model, "language_model", None)
    )

    def _sequence_length(args, kwargs):
        for key in ("inputs_embeds", "input_ids"):
            value = kwargs.get(key) if isinstance(kwargs, dict) else None
            if torch.is_tensor(value) and value.ndim >= 2:
                return int(value.shape[1])
        for value in args:
            if torch.is_tensor(value) and value.ndim >= 2:
                return int(value.shape[1])
        return None

    if language_model is not None:
        state = {"t0": None, "bucket": None}

        def _lm_pre_hook(_m, _args, _kwargs=None):
            kwargs = _kwargs if isinstance(_kwargs, dict) else {}
            seq_len = _sequence_length(_args, kwargs)
            past = kwargs.get("past_key_values")
            bucket = "llm_decode_sec" if past is not None and seq_len == 1 else "llm_prefill_sec"
            _sync_if_cuda()
            state["t0"] = time.time()
            state["bucket"] = bucket

        def _lm_post_hook(_m, _args, _out):
            _sync_if_cuda()
            bucket = state.get("bucket")
            if state["t0"] is not None and bucket in timers:
                timers[bucket] += time.time() - state["t0"]
                counts["llm_decode_calls" if bucket == "llm_decode_sec" else "llm_prefill_calls"] += 1
            state["t0"] = None
            state["bucket"] = None

        try:
            handles.append(language_model.register_forward_pre_hook(_lm_pre_hook, with_kwargs=True))
        except TypeError:
            handles.append(language_model.register_forward_pre_hook(lambda m, a: _lm_pre_hook(m, a, {})))
        handles.append(language_model.register_forward_hook(_lm_post_hook))

    for layer_id, layer in enumerate(getattr(language_model, "layers", []) or []):
        _add_layer_hooks(layer, "language_layer", layer_id)

    token_counters["per_layer_timing"] = per_layer

    return timers, counts, token_counters, handles


def _remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


def _load_dynamic_patch_module():
    root = Path(__file__).resolve().parents[3]
    patch_impl = os.environ.get("QWEN3_PATCH_IMPL", "dynamic").strip().lower()
    if patch_impl == "dymu":
        patch_path = root / "DyMU" / "qwen3_dymu_patch.py"
    elif patch_impl == "diffrate":
        patch_path = root / "DiffRate-Qwen3VL" / "qwen3_diffrate_patch.py"
    elif patch_impl == "visiontrim":
        patch_path = root / "VisionTrim-Qwen3VL" / "qwen3_visiontrim_patch.py"
    elif patch_impl in {"zooprune", "zoo_prune", "zoo-prune"}:
        patch_path = root / "ZOOPrune-Qwen3VL" / "qwen3_zooprune_patch.py"
    elif patch_impl == "vivid":
        patch_path = root / "VIVID-Qwen3VL" / "qwen3_vivid_patch.py"
    else:
        patch_path = root / "Dynamic-Qwen3VL" / "dynamic_patch.py"
    if not patch_path.exists():
        raise RuntimeError(f"Dynamic patch file not found: {patch_path}")

    if patch_impl == "dymu":
        module_name = "dymu_qwen3_patch_runtime"
    elif patch_impl == "diffrate":
        module_name = "diffrate_qwen3_patch_runtime"
    elif patch_impl == "visiontrim":
        module_name = "visiontrim_qwen3_patch_runtime"
    elif patch_impl in {"zooprune", "zoo_prune", "zoo-prune"}:
        module_name = "zooprune_qwen3_patch_runtime"
    elif patch_impl == "vivid":
        module_name = "vivid_qwen3_patch_runtime"
    else:
        module_name = "dynamic_qwen3_patch_runtime"
    spec = importlib.util.spec_from_file_location(module_name, str(patch_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load dynamic patch module from: {patch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iter_checkpoint_state_dicts(model_path: str):
    if not os.path.isdir(model_path):
        return
    safe_index = os.path.join(model_path, "model.safetensors.index.json")
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    safe_single = os.path.join(model_path, "model.safetensors")
    bin_single = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safe_index):
        with open(safe_index, "r", encoding="utf-8") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.exists(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.exists(safe_single):
        shard_files = ["model.safetensors"]
    elif os.path.exists(bin_single):
        shard_files = ["pytorch_model.bin"]
    else:
        raise RuntimeError(f"No checkpoint shard/index found in: {model_path}")

    safe_loader = None
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        if shard_name.endswith(".safetensors"):
            if safe_loader is None:
                from safetensors.torch import load_file as safe_loader
            state = safe_loader(shard_path)
        else:
            state = torch.load(shard_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
        yield state


def _apply_dynamic_patch_and_reload_weights(model, model_path: str) -> None:
    dynamic_enabled = os.environ.get("QWEN3_DYNAMIC_ENABLED", "1") != "0"
    if not dynamic_enabled:
        print("Dynamic patch disabled via QWEN3_DYNAMIC_ENABLED=0")
        return

    local_ckpt = os.path.isdir(model_path)

    patch_impl = os.environ.get("QWEN3_PATCH_IMPL", "dynamic").strip().lower()
    patch_mod = _load_dynamic_patch_module()

    if patch_impl == "vivid":
        vivid_cfg = {}
        state_path = Path(model_path).expanduser() / "distill_state.json"
        if local_ckpt and state_path.exists():
            try:
                with state_path.open("r", encoding="utf-8") as f:
                    state = json.load(f)
                vivid_cfg = state.get("vivid", {}) if isinstance(state.get("vivid"), dict) else {}
            except Exception as e:
                print(f"Warning: failed to read VIVID distill state from {state_path}: {e}")
        compact_tokens = int(os.environ.get("QWEN3_VIVID_COMPACT_TOKENS", vivid_cfg.get("compact_tokens", 128)))
        min_keep_ratio = float(os.environ.get("QWEN3_VIVID_MIN_KEEP_RATIO", vivid_cfg.get("min_keep_ratio", 0.0)))
        kv_anchors = int(os.environ.get("QWEN3_VIVID_ANCHORS", vivid_cfg.get("anchors", 256)))
        kv_topk = int(os.environ.get("QWEN3_VIVID_TOPK", vivid_cfg.get("topk", 0)))
        enable_kv_env = os.environ.get("QWEN3_VIVID_ENABLE_KV")
        enable_kv = (enable_kv_env != "0") if enable_kv_env is not None else bool(vivid_cfg.get("enable_kv_compression", True))
        compact_prefill_env = os.environ.get("QWEN3_VIVID_COMPACT_PREFILL")
        compact_prefill = (compact_prefill_env != "0") if compact_prefill_env is not None else bool(vivid_cfg.get("compact_prefill", True))
        temperature = float(os.environ.get("QWEN3_VIVID_TEMPERATURE", vivid_cfg.get("temperature", 1.0)))
        aggregation_mode = os.environ.get("QWEN3_VIVID_AGGREGATION", vivid_cfg.get("aggregation_mode", "uniform"))
        profile = os.environ.get("QWEN3_VIVID_PROFILE", "0") == "1"
        stats = patch_mod.apply_vivid_qwen3_native_compact_patch(
            model,
            compact_tokens=compact_tokens,
            min_keep_ratio=min_keep_ratio,
            kv_anchors=kv_anchors,
            kv_topk=kv_topk,
            enable_kv_compression=enable_kv,
            enable_compact_prefill=compact_prefill,
            temperature=temperature,
            aggregation_mode=aggregation_mode,
            profile=profile,
            enabled=dynamic_enabled,
        )
        vivid_modules = patch_mod.count_vivid_qwen3_native_modules(model)
        strict = os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0"
        if strict and vivid_modules <= 0:
            raise RuntimeError("VIVID patch requested but no Qwen3 compact modules were active.")

        model_keys = set(model.state_dict().keys())
        loaded_keys = 0
        loaded_vivid_keys = 0
        vivid_key_markers = ("vivid_compact_aggregator", "assign_norm", "assign_proj", "token_agg_norm", "token_agg_proj")
        if local_ckpt:
            for shard_state in _iter_checkpoint_state_dicts(model_path):
                matched = {k: v for k, v in shard_state.items() if k in model_keys}
                if matched:
                    model.load_state_dict(matched, strict=False)
                    loaded_keys += len(matched)
                    loaded_vivid_keys += sum(1 for k in matched if any(m in k for m in vivid_key_markers))
        if strict and state_path.exists() and loaded_vivid_keys == 0:
            raise RuntimeError(
                "VIVID checkpoint was provided but no VIVID adapter keys were loaded. "
                "Ensure checkpoint settings match the VIVID patch."
            )

        print(
            f"✓ VIVID native compact patch applied: stats={stats}, modules={vivid_modules}, "
            f"compact_tokens={compact_tokens}, min_keep_ratio={min_keep_ratio}, kv_anchors={kv_anchors}, kv_topk={kv_topk}, "
            f"compact_prefill={compact_prefill}, aggregation={aggregation_mode}, loaded_keys={loaded_keys}, loaded_vivid_keys={loaded_vivid_keys}"
        )
        return

    if patch_impl == "dymu":
        dymu_phase = int(os.environ.get("QWEN3_DYMU_PHASE", "1"))
        dymu_default_threshold = float(os.environ.get("QWEN3_DYMU_DEFAULT_THRESHOLD", "0.8"))
        dymu_vtu_similarity_threshold = float(os.environ.get("QWEN3_DYMU_VTU_SIM_THRESHOLD", "0.999"))
        dymu_experimental_vtu = os.environ.get("QWEN3_DYMU_EXPERIMENTAL_VTU", "0") == "1"

        merge_layers_env = os.environ.get("QWEN3_DYMU_MERGE_LAYERS", "").strip()
        merge_layers = [int(x.strip()) for x in merge_layers_env.split(",") if x.strip()] if merge_layers_env else None

        thresholds = None
        thresholds_json = os.environ.get("QWEN3_DYMU_THRESHOLDS_JSON", "").strip()
        if thresholds_json:
            p = Path(thresholds_json).expanduser()
            if not p.is_absolute():
                p = Path(model_path).expanduser().resolve() / p
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                thresholds = {int(k): float(v) for k, v in raw.items()}

        cfg_cls = getattr(patch_mod, "DyMUConfig", None)
        if cfg_cls is None:
            raise RuntimeError("DyMU patch module is missing DyMUConfig")

        cfg = cfg_cls(
            phase=dymu_phase,
            merge_layers=merge_layers,
            thresholds=thresholds,
            default_threshold=dymu_default_threshold,
            vtu_similarity_threshold=dymu_vtu_similarity_threshold,
            experimental_vtu=dymu_experimental_vtu,
        )
        stats = patch_mod.apply_dymu_qwen3_patch(model, cfg)
        dynamic_blocks = patch_mod.count_dymu_qwen3_merge_layers(model)
        text_vtu_layers = patch_mod.count_dymu_qwen3_text_vtu_layers(model)
        strict = os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0"
        if strict and dynamic_blocks <= 0:
            raise RuntimeError("DyMU patch requested but no Qwen3 vision merge layers were active.")

        model_keys = set(model.state_dict().keys())
        loaded_keys = 0
        if local_ckpt:
            for shard_state in _iter_checkpoint_state_dicts(model_path):
                matched = {k: v for k, v in shard_state.items() if k in model_keys}
                if not matched:
                    continue
                model.load_state_dict(matched, strict=False)
                loaded_keys += len(matched)
        else:
            print(f"Warning: skipping DyMU checkpoint reload for non-local model: {model_path}")

        print(
            f"✓ DyMU patch applied: phase={dymu_phase}, stats={stats}, "
            f"vision_merge_layers={dynamic_blocks}, text_vtu_layers={text_vtu_layers}, loaded_keys={loaded_keys}"
        )
        return

    if patch_impl == "diffrate":
        hard = os.environ.get("QWEN3_DIFFRATE_HARD", "1") != "0"

        compress_layers_env = os.environ.get("QWEN3_DIFFRATE_COMPRESS_LAYERS", "").strip()
        compress_layers = [int(x.strip()) for x in compress_layers_env.split(",") if x.strip()] if compress_layers_env else None

        cands_env = os.environ.get("QWEN3_DIFFRATE_CANDIDATES", "0.0,0.0625,0.125,0.1875,0.25").strip()
        candidates = [float(x.strip()) for x in cands_env.split(",") if x.strip()] if cands_env else [0.0, 0.0625, 0.125, 0.1875, 0.25]
        min_keep = int(os.environ.get("QWEN3_DIFFRATE_MIN_KEEP", "1"))

        cfg_cls = getattr(patch_mod, "DiffRateConfig", None)
        if cfg_cls is None:
            raise RuntimeError("DiffRate patch module is missing DiffRateConfig")

        cfg = cfg_cls(
            compress_layers=compress_layers,
            candidates=candidates,
            hard=hard,
            min_keep_tokens=min_keep,
        )

        stats = patch_mod.apply_diffrate_qwen3_patch(model, cfg)
        dynamic_blocks = patch_mod.count_diffrate_qwen3_layers(model)
        strict = (os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0") and local_ckpt
        if strict and dynamic_blocks <= 0:
            raise RuntimeError("DiffRate patch requested but no Qwen3 vision compression layers were active.")

        model_keys = set(model.state_dict().keys())
        loaded_keys = 0
        loaded_diffrate_keys = 0

        if local_ckpt:
            for shard_state in _iter_checkpoint_state_dicts(model_path):
                matched = {k: v for k, v in shard_state.items() if k in model_keys}
                if not matched:
                    continue
                model.load_state_dict(matched, strict=False)
                loaded_keys += len(matched)
                loaded_diffrate_keys += sum(1 for k in matched if "diffrate_rate_heads" in k or "_diffrate_rate_heads" in k)
        else:
            print(f"Warning: skipping DiffRate checkpoint reload for non-local model: {model_path}")

        if strict and loaded_diffrate_keys == 0:
            raise RuntimeError(
                "DiffRate patch applied but no DiffRate rate-head checkpoint keys were loaded. "
                "Ensure checkpoint matches DiffRate patch settings."
            )

        schedule = {}
        try:
            if hasattr(patch_mod, "export_diffrate_schedule"):
                schedule = patch_mod.export_diffrate_schedule(model)
        except Exception:
            schedule = {}
        hard_layers = schedule.get("layers", {}) if isinstance(schedule, dict) else {}
        hard_summary = {
            str(k): {
                "p": v.get("hard_prune_rate"),
                "m": v.get("hard_merge_rate"),
            }
            for k, v in hard_layers.items()
            if isinstance(v, dict)
        }

        print(
            f"✓ DiffRate patch applied: stats={stats}, compress_layers={dynamic_blocks}, "
            f"loaded_keys={loaded_keys}, loaded_diffrate_keys={loaded_diffrate_keys}, "
            f"hard_mode={hard}, hard_schedule={hard_summary}"
        )
        return

    if patch_impl == "visiontrim":
        trim_layers_env = os.environ.get("QWEN3_VISIONTRIM_TRIM_LAYERS", "").strip()
        trim_layers = [int(x.strip()) for x in trim_layers_env.split(",") if x.strip()] if trim_layers_env else None
        keep_ratio = float(os.environ.get("QWEN3_VISIONTRIM_KEEP_RATIO", "0.7"))
        min_keep = int(os.environ.get("QWEN3_VISIONTRIM_MIN_KEEP", "1"))
        local_window = int(os.environ.get("QWEN3_VISIONTRIM_LOCAL_WINDOW", "9"))
        complement_ratio = float(os.environ.get("QWEN3_VISIONTRIM_COMPLEMENT_RATIO", "1.0"))

        cfg_cls = getattr(patch_mod, "VisionTrimConfig", None)
        if cfg_cls is None:
            raise RuntimeError("VisionTrim patch module is missing VisionTrimConfig")
        cfg = cfg_cls(
            trim_layers=trim_layers,
            keep_ratio=keep_ratio,
            min_keep_tokens=min_keep,
            local_window=local_window,
            complement_ratio=complement_ratio,
        )
        stats = patch_mod.apply_visiontrim_qwen3_patch(model, cfg)
        trim_layers_count = patch_mod.count_visiontrim_qwen3_layers(model)
        strict = os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0"
        if strict and trim_layers_count <= 0:
            raise RuntimeError("VisionTrim patch requested but no Qwen3 vision trim layers were active.")

        model_keys = set(model.state_dict().keys())
        loaded_keys = 0
        if local_ckpt:
            for shard_state in _iter_checkpoint_state_dicts(model_path):
                matched = {k: v for k, v in shard_state.items() if k in model_keys}
                if matched:
                    model.load_state_dict(matched, strict=False)
                    loaded_keys += len(matched)

        print(
            f"✓ VisionTrim patch applied: stats={stats}, trim_layers={trim_layers_count}, "
            f"loaded_keys={loaded_keys}"
        )
        return

    if patch_impl in {"zooprune", "zoo_prune", "zoo-prune"}:
        prune_layers_env = os.environ.get("QWEN3_ZOOPRUNE_PRUNE_LAYERS", "").strip()
        prune_layers = [int(x.strip()) for x in prune_layers_env.split(",") if x.strip()] if prune_layers_env else None
        keep_ratio = float(os.environ.get("QWEN3_ZOOPRUNE_KEEP_RATIO", "0.7"))
        min_keep = int(os.environ.get("QWEN3_ZOOPRUNE_MIN_KEEP", "1"))
        num_directions = int(os.environ.get("QWEN3_ZOOPRUNE_NUM_DIRECTIONS", "1"))
        perturb_eps = float(os.environ.get("QWEN3_ZOOPRUNE_PERTURB_EPS", "0.001"))
        diversity_weight = float(os.environ.get("QWEN3_ZOOPRUNE_DIVERSITY_WEIGHT", "0.25"))
        merge_dropped = os.environ.get("QWEN3_ZOOPRUNE_MERGE_DROPPED", "0") == "1"
        max_greedy_tokens = int(os.environ.get("QWEN3_ZOOPRUNE_MAX_GREEDY_TOKENS", "768"))

        cfg_cls = getattr(patch_mod, "ZOOPruneConfig", None)
        if cfg_cls is None:
            raise RuntimeError("ZOO-Prune patch module is missing ZOOPruneConfig")
        cfg = cfg_cls(
            prune_layers=prune_layers,
            keep_ratio=keep_ratio,
            min_keep_tokens=min_keep,
            num_directions=num_directions,
            perturb_eps=perturb_eps,
            diversity_weight=diversity_weight,
            merge_dropped=merge_dropped,
            max_greedy_tokens=max_greedy_tokens,
        )
        stats = patch_mod.apply_zooprune_qwen3_patch(model, cfg)
        prune_layers_count = patch_mod.count_zooprune_qwen3_layers(model)
        strict = os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0"
        if strict and prune_layers_count <= 0:
            raise RuntimeError("ZOO-Prune patch requested but no Qwen3 vision prune layers were active.")

        model_keys = set(model.state_dict().keys())
        loaded_keys = 0
        if local_ckpt:
            for shard_state in _iter_checkpoint_state_dicts(model_path):
                matched = {k: v for k, v in shard_state.items() if k in model_keys}
                if matched:
                    model.load_state_dict(matched, strict=False)
                    loaded_keys += len(matched)

        print(
            f"✓ ZOO-Prune patch applied: stats={stats}, prune_layers={prune_layers_count}, "
            f"loaded_keys={loaded_keys}"
        )
        return

    dynamic_cfg = {}
    if local_ckpt:
        state_path = Path(model_path).expanduser() / "distill_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                dynamic_cfg = state.get("dynamic_vision", {}) if isinstance(state.get("dynamic_vision"), dict) else {}
            except Exception:
                dynamic_cfg = {}

    dynamic_mode = str(os.environ.get("QWEN3_DYNAMIC_MODE", dynamic_cfg.get("mode", "hard_shrink"))).strip().lower()
    dynamic_keep_ratio = float(os.environ.get("QWEN3_DYNAMIC_KEEP_RATIO", dynamic_cfg.get("keep_ratio", 0.7)))
    dynamic_min_keep = int(os.environ.get("QWEN3_DYNAMIC_MIN_KEEP", dynamic_cfg.get("min_keep_tokens", 32)))

    stage_ratios_env = os.environ.get("QWEN3_DYNAMIC_STAGE_RATIOS", "").strip()
    if stage_ratios_env:
        stage_ratios = [float(x.strip()) for x in stage_ratios_env.split(",") if x.strip()]
    else:
        cfg_stage_ratios = dynamic_cfg.get("stage_keep_ratios")
        stage_ratios = [float(x) for x in cfg_stage_ratios] if isinstance(cfg_stage_ratios, list) else None

    pruning_locs_env = os.environ.get("QWEN3_DYNAMIC_PRUNING_LOCS", "").strip()
    if pruning_locs_env:
        pruning_locs = [int(x.strip()) for x in pruning_locs_env.split(",") if x.strip()]
    else:
        cfg_pruning_locs = dynamic_cfg.get("pruning_locs")
        pruning_locs = [int(x) for x in cfg_pruning_locs] if isinstance(cfg_pruning_locs, list) else None

    if dynamic_mode == "hard_shrink":
        patched = patch_mod.apply_dynamic_qwen3_hard_shrink_patch(
            model,
            keep_ratio=dynamic_keep_ratio,
            keep_ratios=stage_ratios,
            pruning_locs=pruning_locs,
            min_keep_tokens=dynamic_min_keep,
            enabled=dynamic_enabled,
        )
    else:
        patched = patch_mod.apply_dynamic_qwen3_vision_patch(
            model,
            keep_ratio=dynamic_keep_ratio,
            keep_ratios=stage_ratios,
            pruning_locs=pruning_locs,
            min_keep_tokens=dynamic_min_keep,
            enabled=dynamic_enabled,
        )

    dynamic_blocks = patch_mod.count_dynamic_qwen3_blocks(model)
    strict = os.environ.get("QWEN3_DYNAMIC_STRICT", "1") != "0"
    if strict and dynamic_blocks <= 0:
        raise RuntimeError("Dynamic patch requested but no Qwen3 vision blocks were patched.")

    model_keys = set(model.state_dict().keys())
    dynamic_key_markers = ("score_norm", "score_in", "score_out", "base_block", "base_attn")
    loaded_keys = 0
    loaded_dynamic_keys = 0

    if local_ckpt:
        for shard_state in _iter_checkpoint_state_dicts(model_path):
            matched = {k: v for k, v in shard_state.items() if k in model_keys}
            if not matched:
                continue
            model.load_state_dict(matched, strict=False)
            loaded_keys += len(matched)
            loaded_dynamic_keys += sum(1 for k in matched if any(m in k for m in dynamic_key_markers))
    else:
        print(f"Warning: skipping dynamic checkpoint reload for non-local model: {model_path}")

    if strict and loaded_dynamic_keys == 0:
        raise RuntimeError(
            "Dynamic patch applied but no dynamic wrapper checkpoint keys were loaded. "
            "Ensure checkpoint matches selected dynamic mode."
        )

    print(
        f"✓ Dynamic patch applied: mode={dynamic_mode}, patched_now={patched}, dynamic_blocks={dynamic_blocks}, "
        f"loaded_keys={loaded_keys}, loaded_dynamic_keys={loaded_dynamic_keys}"
    )


def _resolve_base_model_path_for_dynamic(checkpoint_path: str) -> str:
    def _resolve_candidate(candidate: str) -> str | None:
        cand = str(candidate or "").strip()
        if not cand:
            return None

        p = Path(cand).expanduser()
        if p.is_absolute():
            return str(p.resolve()) if p.exists() else None

        is_path_like = ("/" in cand) or ("\\" in cand) or cand.startswith(".")
        if is_path_like:
            bases = [
                Path(checkpoint_path),
                Path(checkpoint_path).parent,
                Path(checkpoint_path).parents[1],
                Path.cwd(),
            ]
            for b in bases:
                cp = (b / p).resolve()
                if cp.exists():
                    return str(cp)
            # `org/model` Hugging Face ids contain a slash but are not local
            # paths. Let transformers resolve those instead of failing here.
            if "\\" not in cand and not cand.startswith(".") and len([x for x in cand.split("/") if x]) >= 2:
                return cand
            return None

        return cand

    dynamic_enabled = os.environ.get("QWEN3_DYNAMIC_ENABLED", "1") != "0"
    if not dynamic_enabled:
        return checkpoint_path

    base_path = os.environ.get("QWEN3_BASE_MODEL_PATH", "").strip()
    if base_path:
        resolved = _resolve_candidate(base_path)
        if resolved is None:
            raise RuntimeError(f"QWEN3_BASE_MODEL_PATH could not be resolved: {base_path}")
        return resolved

    distill_state_path = os.path.join(checkpoint_path, "distill_state.json")
    if os.path.exists(distill_state_path):
        try:
            with open(distill_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            candidate = str(
                state.get("student_model_name_or_path")
                or state.get("teacher_model_name_or_path")
                or ""
            ).strip()
            if candidate:
                resolved = _resolve_candidate(candidate)
                if resolved is not None:
                    print(f"Using base model from distill_state.json: {resolved}")
                    return resolved
        except Exception:
            pass

    fallback = os.environ.get("QWEN3_DEFAULT_BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct").strip()
    if fallback:
        print(f"Using default base model: {fallback}")
        return fallback

    raise RuntimeError(
        "Dynamic checkpoint requires a base Qwen3-VL model to initialize architecture without mismatch warnings. "
        "Set QWEN3_BASE_MODEL_PATH or include student_model_name_or_path in distill_state.json."
    )

def _sanitize_local_tokenizer_config(model_path: str) -> None:
    cfg_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return

    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        cfg["extra_special_tokens"] = {
            f"extra_special_token_{i}": tok for i, tok in enumerate(extra)
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("Warning: normalized tokenizer_config.json extra_special_tokens from list to dict.")


def _sanitize_local_model_config(model_path: str) -> None:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return

    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    if not text_cfg:
        return

    if text_cfg.get("rope_scaling") is None and isinstance(text_cfg.get("rope_parameters"), dict):
        text_cfg["rope_scaling"] = text_cfg["rope_parameters"]
        cfg["text_config"] = text_cfg
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("Warning: normalized config.json text_config.rope_scaling from rope_parameters.")

def clean_for_excel(val):
    """
    Remove characters that are illegal in Excel cells and trim long strings.
    Excel doesn't support control characters (0x00-0x1F) except tab, newline, carriage return.
    Excel cell text limit is 32767 characters.
    """
    if isinstance(val, str):
        # Remove control characters (0x00-0x1F) except tab(0x09), newline(0x0A), carriage return(0x0D)
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', val)
        if len(cleaned) > 32767:
            return cleaned[:32764] + "..."
        return cleaned
    return val

def clean_dataframe_for_excel(df):
    """Clean all string columns in a DataFrame for Excel compatibility."""
    if hasattr(df, 'map'):
        return df.map(clean_for_excel)
    return df.applymap(clean_for_excel)

def build_mathv_prompt(line, dump_image_func, dataset):
    """
    Build MathVision dataset prompt.
    """
    # Standard resolution (MathVision uses smaller min_pixels)
    MIN_PIXELS = 768*28*28  # ~0.6M pixels
    MAX_PIXELS = 5120*28*28  # ~4M pixels
    
    tgt_path = dump_image_func(line)
    question = line['question']
    
    # Build messages in standard conversation format
    content = []
    
    # Add all images first
    if isinstance(tgt_path, list):
        for p in tgt_path:
            content.append({
                "type": "image",
                "image": p,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS
            })
    else:
        content.append({
            "type": "image", 
            "image": tgt_path,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS
        })
    
    # Add question text last
    content.append({"type": "text", "text": question})
    
    # Return messages in standard conversation format
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages

def _prepare_hf_inputs(messages, processor, model_device):
    """
    Prepare inputs for Transformers generation.
    
    Args:
        messages: List of messages in standard conversation format
        processor: AutoProcessor instance
    
    Returns:
        dict: input tensors for model.generate
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vis = process_vision_info(messages)
    if isinstance(vis, tuple) and len(vis) >= 2:
        image_inputs, video_inputs = vis[0], vis[1]
    else:
        image_inputs, video_inputs = None, None

    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(model_device) if hasattr(v, "to") else v for k, v in model_inputs.items()}


def _processor_has_chat_template(processor) -> bool:
    proc_tpl = getattr(processor, "chat_template", None)
    tok = getattr(processor, "tokenizer", None)
    tok_tpl = getattr(tok, "chat_template", None) if tok is not None else None
    return bool(proc_tpl) or bool(tok_tpl)


def _load_processor_with_chat_template_fallback(checkpoint_path: str, base_model_path: str):
    processor = None
    processor_load_path = checkpoint_path
    print(f"Loading processor from {processor_load_path}")
    try:
        _sanitize_local_tokenizer_config(processor_load_path)
        _sanitize_local_model_config(processor_load_path)
        processor = AutoProcessor.from_pretrained(
            processor_load_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if not _processor_has_chat_template(processor):
            print(
                "Warning: checkpoint processor has no chat template; "
                f"falling back to base processor: {base_model_path}"
            )
            processor = None
    except Exception as e:
        print(f"Warning: failed to load processor from checkpoint ({e}); will try base model processor.")
        processor = None

    if processor is None:
        processor_load_path = base_model_path
        _sanitize_local_tokenizer_config(processor_load_path)
        _sanitize_local_model_config(processor_load_path)
        processor_load_is_local = Path(processor_load_path).expanduser().exists()
        processor = AutoProcessor.from_pretrained(
            processor_load_path,
            trust_remote_code=True,
            local_files_only=processor_load_is_local,
        )

    if not _processor_has_chat_template(processor):
        raise RuntimeError(
            "Loaded processor still has no chat template. "
            "Set QWEN3_BASE_MODEL_PATH to a valid Qwen3-VL-Instruct model path or HF id."
        )

    print(f"✓ Processor loaded from: {processor_load_path}\n")
    return processor

def run_inference(args):
    """Run inference on the MathVision dataset using Transformers."""
    print("\n" + "="*80)
    print("🚀 MathVision Inference with Transformers")
    print("="*80 + "\n")
    
    os.environ['LMUData'] = args.data_dir
    lmu_data = os.environ['LMUData']

    # Load dataset
    data = load_dataset(args.dataset)
    
    # Limit number of samples if specified
    if args.num_samples is not None and args.num_samples > 0:
        original_len = len(data)
        data = data.iloc[:args.num_samples]
        print(f"✓ Loaded {len(data)} samples from {args.dataset} (limited from {original_len} samples)")
    else:
        print(f"✓ Loaded {len(data)} samples from {args.dataset}")
    
    # Set up image root directory
    img_root = os.path.join(lmu_data, 'images', args.dataset)
    os.makedirs(img_root, exist_ok=True)
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if not args.profile_output:
        args.profile_output = args.output_file.replace('.jsonl', '_timing.json')

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " Let's think step by step."
        print(f"✓ Using CoT prompt: {cot_prompt[:50]}...")

    print(f"\n⚙️  Generation parameters (Transformers):")
    print(f"   max_new_tokens={args.max_new_tokens}")
    print(f"   temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"   repetition_penalty={args.repetition_penalty}")
    print(f"   presence_penalty={args.presence_penalty}")

    if args.presence_penalty > 0:
        print(f"   ✅ Anti-repetition enabled (presence_penalty={args.presence_penalty})")

    if args.temperature <= 0.02 and args.top_k == 1:
        print(f"   ✅ Using FAST greedy-like decoding")
    else:
        print(f"   ⚠️  Using sampling decoding (slower but more diverse)")
    print()

    model_load_path = _resolve_base_model_path_for_dynamic(args.model_path)

    # Load processor for input preparation
    processor = _load_processor_with_chat_template_fallback(args.model_path, model_load_path)

    _sanitize_local_model_config(model_load_path)
    print(f"Loading model with Transformers from: {model_load_path}")
    model_load_is_local = Path(model_load_path).expanduser().exists()
    model_dtype = _resolve_model_dtype()
    model = AutoModelForImageTextToText.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        local_files_only=model_load_is_local,
        dtype=model_dtype,
        device_map="auto",
    )
    _apply_dynamic_patch_and_reload_weights(model, args.model_path)
    model.eval()
    model_device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {model_device}\n")
    timers, timer_counts, token_counters, timer_handles = _install_component_timers(model)

    # Prepare all prompts
    print("Preparing prompts...")
    all_line_dicts = []
    all_messages = []
    
    for idx, (_, line) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Building prompts")):
        # Convert line to dict
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)
        
        # Build prompt
        messages = build_mathv_prompt(line, dump_image_func, args.dataset)
        
        # Add CoT prompt
        if args.use_cot and len(messages) > 0 and len(messages[0]['content']) > 0:
            last_content = messages[0]['content'][-1]
            if last_content['type'] == 'text':
                last_content['text'] += cot_prompt
        
        all_line_dicts.append(line_dict)
        all_messages.append(messages)
    
    print(f"✓ Prepared {len(all_messages)} prompts\n")

    # Inference loop
    print("="*80)
    print("🚀 Running Transformers inference")
    print("="*80)
    start_time = time.time()

    # Save results
    print("Saving results...")
    results = []
    profile_rows = []

    visual_mod = getattr(getattr(model, "model", None), "visual", None) or getattr(model, "visual", None)

    for line_dict, messages in tqdm(zip(all_line_dicts, all_messages), total=len(all_messages), desc="Generating"):
        model_inputs = _prepare_hf_inputs(messages, processor, model_device)
        input_len = model_inputs["input_ids"].shape[1]
        pre_embed = timers["input_embedding_sec"]
        pre_vision = timers["vision_encoder_sec"]
        pre_proj = timers["projection_sec"]
        pre_llm_prefill = timers["llm_prefill_sec"]
        pre_llm_decode = timers["llm_decode_sec"]
        pre_proj_tokens = token_counters["projection_input_tokens"]
        pre_patch_tokens = _count_visual_tokens_from_model_inputs(model_inputs)
        _reset_dynamic_token_counters(visual_mod)
        _sync_if_cuda()
        t_gen0 = time.time()
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **_build_generate_kwargs(args, processor))
        _sync_if_cuda()
        t_gen1 = time.time()
        gen_ids = generated[:, input_len:]
        output_tokens = int(gen_ids.shape[1])
        response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        index = line_dict['index']

        response_final = str(response).split("</think>")[-1].strip()
        
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {"gen": response_final, "gen_raw": response},
            "messages": messages
        }
        results.append(result)

        emb_t = max(0.0, timers["input_embedding_sec"] - pre_embed)
        vis_t = max(0.0, timers["vision_encoder_sec"] - pre_vision)
        proj_t = max(0.0, timers["projection_sec"] - pre_proj)
        llm_prefill_t = max(0.0, timers["llm_prefill_sec"] - pre_llm_prefill)
        llm_decode_t = max(0.0, timers["llm_decode_sec"] - pre_llm_decode)
        projection_tokens = max(0, int(token_counters["projection_input_tokens"] - pre_proj_tokens))
        alive_tokens = _read_dynamic_alive_tokens(visual_mod)
        if alive_tokens is None:
            alive_tokens = projection_tokens
        llm_prefill_tokens = _read_dynamic_llm_prefill_tokens(model, visual_mod, input_len)
        _merge_layer_token_stats(
            token_counters.get("per_layer_timing", {}),
            _read_dynamic_layer_token_stats(visual_mod),
            pre_patch_tokens,
        )
        total_t = max(0.0, t_gen1 - t_gen0)
        residual_llm_t = max(0.0, total_t - emb_t - vis_t - proj_t)
        dec_t = llm_prefill_t + llm_decode_t
        if dec_t <= 0.0:
            llm_decode_t = residual_llm_t
            dec_t = residual_llm_t
        profile_rows.append(
            {
                "index": int(index) if isinstance(index, np.integer) else index,
                "generate_total_sec": total_t,
                "input_embedding_sec": emb_t,
                "vision_encoder_sec": vis_t,
                "projection_sec": proj_t,
                "llm_prefill_sec": llm_prefill_t,
                "llm_decode_sec": llm_decode_t,
                "llm_decoder_sec": dec_t,
                "input_tokens": int(llm_prefill_tokens),
                "llm_prefill_tokens": int(llm_prefill_tokens),
                "dense_input_tokens": int(input_len),
                "output_tokens": int(output_tokens),
                "visual_tokens_pre_patch": int(pre_patch_tokens),
                "visual_tokens_alive_after_patch": int(alive_tokens),
                "visual_tokens_pre_projection": int(projection_tokens),
                "visual_token_keep_ratio": (float(alive_tokens) / float(pre_patch_tokens)) if pre_patch_tokens > 0 else 0.0,
                "visual_projection_token_ratio": (float(projection_tokens) / float(pre_patch_tokens)) if pre_patch_tokens > 0 else 0.0,
                "active_visual_tokens": int(alive_tokens),
                "compression_overhead_ms": _read_dynamic_compression_overhead_ms(model, visual_mod),
                "kv_cache_gpu_memory_estimated_mb": float(getattr(visual_mod, "_vivid_last_kv_cache_estimated_mb", 0.0) or 0.0),
                "encoder_flops_ratio_estimated": float(
                    getattr(
                        visual_mod,
                        "_vivid_last_encoder_flops_ratio_estimated",
                        ((float(alive_tokens) / float(pre_patch_tokens)) ** 2) if pre_patch_tokens > 0 else 0.0,
                    )
                    or 0.0
                ),
            }
        )

    total_time = time.time() - start_time
    _remove_hooks(timer_handles)
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(data):.2f} seconds/sample")
    print(f"  Throughput: {len(data)/total_time:.2f} samples/second\n")
    
    # Write final results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")

    if args.profile_output:
        if profile_rows:
            agg_total = float(sum(x["generate_total_sec"] for x in profile_rows))
            agg_embed = float(sum(x["input_embedding_sec"] for x in profile_rows))
            agg_vision = float(sum(x["vision_encoder_sec"] for x in profile_rows))
            agg_proj = float(sum(x["projection_sec"] for x in profile_rows))
            agg_prefill = float(sum(x.get("llm_prefill_sec", 0.0) for x in profile_rows))
            agg_decode = float(sum(x.get("llm_decode_sec", 0.0) for x in profile_rows))
            agg_dec = float(sum(x["llm_decoder_sec"] for x in profile_rows))
            agg_input_tokens = int(sum(int(x.get("input_tokens", 0)) for x in profile_rows))
            agg_out_tokens = int(sum(int(x.get("output_tokens", 0)) for x in profile_rows))
            agg_tokens_pre = int(sum(x["visual_tokens_pre_patch"] for x in profile_rows))
            agg_tokens_alive = int(sum(x.get("visual_tokens_alive_after_patch", x["visual_tokens_pre_projection"]) for x in profile_rows))
            agg_tokens_post = int(sum(x["visual_tokens_pre_projection"] for x in profile_rows))
            agg_compression_ms = float(sum(float(x.get("compression_overhead_ms", 0.0) or 0.0) for x in profile_rows))
            agg_kv_cache_mb = float(sum(float(x.get("kv_cache_gpu_memory_estimated_mb", 0.0) or 0.0) for x in profile_rows) / len(profile_rows))
            agg_flops_ratio = float(sum(float(x.get("encoder_flops_ratio_estimated", 0.0) or 0.0) for x in profile_rows) / len(profile_rows))
        else:
            agg_total = agg_embed = agg_vision = agg_proj = agg_prefill = agg_decode = agg_dec = 0.0
            agg_input_tokens = 0
            agg_out_tokens = 0
            agg_tokens_pre = agg_tokens_alive = agg_tokens_post = 0
            agg_compression_ms = agg_kv_cache_mb = agg_flops_ratio = 0.0

        timing_payload = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": len(results),
            "compression_overhead_source": "measured_patch_runtime",
            "total": {
                "generate_total_sec": agg_total,
                "input_embedding_sec": agg_embed,
                "vision_encoder_sec": agg_vision,
                "projection_sec": agg_proj,
                "llm_prefill_sec": agg_prefill,
                "llm_decode_sec": agg_decode,
                "llm_decoder_sec": agg_dec,
                "input_tokens": agg_input_tokens,
                "llm_prefill_tokens": agg_input_tokens,
                "output_tokens": agg_out_tokens,
                "visual_tokens_pre_patch": agg_tokens_pre,
                "visual_tokens_alive_after_patch": agg_tokens_alive,
                "visual_tokens_pre_projection": agg_tokens_post,
                "visual_token_keep_ratio": (float(agg_tokens_alive) / float(agg_tokens_pre)) if agg_tokens_pre > 0 else 0.0,
                "visual_projection_token_ratio": (float(agg_tokens_post) / float(agg_tokens_pre)) if agg_tokens_pre > 0 else 0.0,
                "compression_overhead_ms": agg_compression_ms,
                "kv_cache_gpu_memory_estimated_mb": agg_kv_cache_mb,
                "encoder_flops_ratio_estimated": agg_flops_ratio,
            },
            "avg_per_sample": {
                "generate_total_sec": (agg_total / len(results)) if len(results) else 0.0,
                "input_embedding_sec": (agg_embed / len(results)) if len(results) else 0.0,
                "vision_encoder_sec": (agg_vision / len(results)) if len(results) else 0.0,
                "projection_sec": (agg_proj / len(results)) if len(results) else 0.0,
                "llm_prefill_sec": (agg_prefill / len(results)) if len(results) else 0.0,
                "llm_decode_sec": (agg_decode / len(results)) if len(results) else 0.0,
                "llm_decoder_sec": (agg_dec / len(results)) if len(results) else 0.0,
                "input_tokens": (float(agg_input_tokens) / len(results)) if len(results) else 0.0,
                "llm_prefill_tokens": (float(agg_input_tokens) / len(results)) if len(results) else 0.0,
                "output_tokens": (float(agg_out_tokens) / len(results)) if len(results) else 0.0,
                "visual_tokens_pre_patch": (float(agg_tokens_pre) / len(results)) if len(results) else 0.0,
                "visual_tokens_alive_after_patch": (float(agg_tokens_alive) / len(results)) if len(results) else 0.0,
                "visual_tokens_pre_projection": (float(agg_tokens_post) / len(results)) if len(results) else 0.0,
                "visual_token_keep_ratio": (float(agg_tokens_alive) / float(agg_tokens_pre)) if agg_tokens_pre > 0 else 0.0,
                "visual_projection_token_ratio": (float(agg_tokens_post) / float(agg_tokens_pre)) if agg_tokens_pre > 0 else 0.0,
                "compression_overhead_ms": (agg_compression_ms / len(profile_rows)) if len(profile_rows) else 0.0,
                "kv_cache_gpu_memory_estimated_mb": agg_kv_cache_mb,
                "encoder_flops_ratio_estimated": agg_flops_ratio,
            },
            "calls": timer_counts,
            "token_calls": token_counters,
            "per_layer": token_counters.get("per_layer_timing", {}),
            "per_sample": profile_rows,
        }
        with open(args.profile_output, "w", encoding="utf-8") as pf:
            json.dump(timing_payload, pf, ensure_ascii=False, indent=2)
        print(f"✓ Component timing saved to {args.profile_output}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    os.environ['LMUData'] = args.data_dir
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)
            
    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]

    # Load dataset for validation
    meta = load_dataset(args.dataset)

    # Validation with fallback if mismatch
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}

    def _overlap_ratio(meta_map: dict, data_keys: list) -> float:
        if not data_keys:
            return 0.0
        hit = sum(1 for k in data_keys if k in meta_map)
        return hit / float(len(data_keys))

    data_keys = list(data_map.keys())
    overlap = _overlap_ratio(meta_q_map, data_keys)

    if overlap < 0.999:
        alt_dataset = "MathVision" if args.dataset == "MathVision_MINI" else "MathVision_MINI"
        try:
            alt_meta = load_dataset(alt_dataset)
            alt_meta_map = {x: y for x, y in zip(alt_meta['index'], alt_meta['question'])}
            alt_overlap = _overlap_ratio(alt_meta_map, data_keys)
        except Exception:
            alt_meta = None
            alt_meta_map = {}
            alt_overlap = 0.0

        if alt_overlap > overlap:
            print(
                f"Warning: evaluation data overlaps {alt_dataset} more than {args.dataset} "
                f"(overlap={alt_overlap:.3f} vs {overlap:.3f}). Using {alt_dataset}."
            )
            meta = alt_meta
            meta_q_map = alt_meta_map
            overlap = alt_overlap
        else:
            print(
                f"Warning: evaluation data is not a full subset of {args.dataset} "
                f"(overlap={overlap:.3f}). Proceeding with intersection only."
            )

    # Filter to intersection to avoid assertion failures
    valid_indices = {k for k in data_keys if k in meta_q_map}
    if valid_indices:
        data = data[data['index'].isin(valid_indices)]
    else:
        raise RuntimeError(
            f"No overlapping indices between eval file and dataset {args.dataset}. "
            "Check that you are using the matching dataset for inference/eval."
        )

    # Save intermediate results
    output_xlsx = args.output_file.replace('.csv', '.xlsx') if args.output_file.endswith('.csv') else args.output_file
    clean_dataframe_for_excel(data).to_excel(output_xlsx, index=False)
    print(f"✓ Saved intermediate results to {output_xlsx}")

    # Build judge model
    model = build_judge(
        model=getattr(args, 'eval_model', 'gpt-4o-2024-05-13'),
        api_type=getattr(args, 'api_type', 'dash')
    )
    
    # Prepare evaluation tasks
    eval_tasks = []
    for i in range(len(data)):
        item = data.iloc[i]
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Debug mode: process single-threaded with first few samples
    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Running in debug mode with first 5 samples...")
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        from concurrent.futures import ThreadPoolExecutor
        nproc = getattr(args, 'nproc', 4)
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # Update data with evaluation results
    data['res'] = [r['res'] for r in eval_results]
    data['log'] = [r['log'] for r in eval_results]
    data['extract_model'] = [r['extract_model'] for r in eval_results]
    data['extract_flag'] = [r['extract_flag'] for r in eval_results]
    
    # Save evaluation results
    storage = args.output_file.replace('.csv', '_eval.xlsx')
    clean_dataframe_for_excel(data).to_excel(storage, index=False)
    print(f"✓ Saved evaluation results to {storage}")
    
    # Calculate accuracy
    score = MATH_V_acc(storage)
    score_pth = storage.replace('.xlsx', '_score.csv')
    score.to_csv(score_pth, index=False)
    print(f"✓ Saved score to {score_pth}")
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(score)
    print(f"{'='*50}\n")
    
    return score

def main():
    parser = argparse.ArgumentParser(description="MathVision Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MathVision", 
                            choices=["MathVision", "MathVision_MINI"],
                            help="Dataset name")
    infer_parser.add_argument("--profile-output", type=str, default=None, help="Write component timing JSON to this path")
    infer_parser.add_argument("--data-dir", type=str, default="/media/chenxi/ISC/VIVID/LMUData", help="The absolute path of MathVision data directory")
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--num-samples", type=int, default=None, 
                            help="Number of samples to process (default: None, process all samples)")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # vLLM specific parameters
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None, 
                            help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                            help="Maximum model context length (default: 128000)")
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=10,
                            help="Maximum images per prompt (default: 10)")
    
    # Generation parameters
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768, 
                            help="Maximum number of tokens to generate (default: 2048)")
    infer_parser.add_argument("--temperature", type=float, default=0.7, 
                            help="Temperature for sampling (default: 0.7 for greedy-like decoding)")
    infer_parser.add_argument("--top-p", type=float, default=0.8, 
                            help="Top-p for sampling (default: 0.8 for greedy-like decoding)")
    infer_parser.add_argument("--top-k", type=int, default=20, 
                            help="Top-k for sampling (default: 20 for greedy decoding)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0, increase to 1.2-1.5 to reduce repetition)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5, range: 0.0-2.0, penalize tokens that have already appeared)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, default="/media/chenxi/ISC/VIVID/LMUData", help="The absolute path of MathVision data directory")
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MathVision",
                            choices=["MathVision", "MathVision_MINI"],
                            help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-4o",
                            help="Model to use for evaluation (default: gpt-4o)")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    
    if hasattr(args, 'data_dir'):
        os.environ['LMUData'] = args.data_dir
    
    # Automatically set tensor_parallel_size
    if args.command == 'infer' and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")
    
    if args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()
        return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
