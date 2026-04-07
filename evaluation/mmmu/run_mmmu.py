import os
import sys
import json
import argparse
import importlib.util
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
import torch
import warnings
import string
import traceback

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

# Local imports from refactored files
from dataset_utils import load_dataset, dump_image, MMMU_preproc
from eval_utils import build_judge, eval_single_sample


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _install_component_timers(model):
    timers = {
        "input_embedding_sec": 0.0,
        "vision_encoder_sec": 0.0,
        "projection_sec": 0.0,
    }
    counts = {
        "input_embedding_calls": 0,
        "vision_encoder_calls": 0,
        "projection_calls": 0,
    }
    handles = []

    def _add_hooks(module, key_time, key_count):
        if module is None:
            return

        state = {"t0": None}

        def _pre_hook(_m, _args):
            _sync_if_cuda()
            state["t0"] = time.time()

        def _post_hook(_m, _args, _out):
            _sync_if_cuda()
            if state["t0"] is not None:
                timers[key_time] += time.time() - state["t0"]
                counts[key_count] += 1
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
    _add_hooks(projector, "projection_sec", "projection_calls")

    return timers, counts, handles


def _remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


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


def _load_dynamic_patch_module():
    root = Path(__file__).resolve().parents[3]
    patch_path = root / "Dynamic-Qwen3VL" / "dynamic_patch.py"
    if not patch_path.exists():
        raise RuntimeError(f"Dynamic patch file not found: {patch_path}")

    spec = importlib.util.spec_from_file_location("dynamic_qwen3_patch_runtime", str(patch_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load dynamic patch module from: {patch_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iter_checkpoint_state_dicts(model_path: str):
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

    dynamic_mode = os.environ.get("QWEN3_DYNAMIC_MODE", "hard_shrink").strip().lower()
    dynamic_keep_ratio = float(os.environ.get("QWEN3_DYNAMIC_KEEP_RATIO", "0.7"))
    dynamic_min_keep = int(os.environ.get("QWEN3_DYNAMIC_MIN_KEEP", "32"))

    stage_ratios_env = os.environ.get("QWEN3_DYNAMIC_STAGE_RATIOS", "").strip()
    stage_ratios = [float(x.strip()) for x in stage_ratios_env.split(",") if x.strip()] if stage_ratios_env else None

    pruning_locs_env = os.environ.get("QWEN3_DYNAMIC_PRUNING_LOCS", "").strip()
    pruning_locs = [int(x.strip()) for x in pruning_locs_env.split(",") if x.strip()] if pruning_locs_env else None

    patch_mod = _load_dynamic_patch_module()
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

    for shard_state in _iter_checkpoint_state_dicts(model_path):
        matched = {k: v for k, v in shard_state.items() if k in model_keys}
        if not matched:
            continue
        model.load_state_dict(matched, strict=False)
        loaded_keys += len(matched)
        loaded_dynamic_keys += sum(1 for k in matched if any(m in k for m in dynamic_key_markers))

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

def build_mmmu_prompt(line, dump_image_func, dataset):
    """Build MMMU dataset prompt with standard resolution settings."""
    # Standard resolution settings
    MIN_PIXELS = 1280*28*28  # ~1M pixels
    MAX_PIXELS = 5120*28*28  # ~4M pixels
    
    tgt_path = dump_image_func(line)
    question = line['question']
    options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'Question: {question}\n'
    if len(options):
        prompt += options_prompt
        prompt += 'Please select the correct answer from the options above. \n'
    prompt = prompt.rstrip()
    
    # Build messages in standard conversation format
    content = []
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
    content.append({"type": "text", "text": prompt})
    
    # Return messages in standard conversation format
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages

def _prepare_hf_inputs(messages, processor, model_device):
    """Prepare inputs for Transformers generation."""
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

def run_inference(args):
    """Run inference on the MMMU dataset using Transformers."""
    print("\n" + "="*80)
    print("🚀 MMMU Inference with Transformers")
    print("="*80 + "\n")
    
    lmu_data = os.environ['LMUData']

    # Load dataset
    data = load_dataset(args.dataset)
    if args.max_samples is not None and args.max_samples > 0:
        data = data.iloc[:args.max_samples]
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
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
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

    # Load processor for input preparation
    print(f"Loading processor from {args.model_path}")
    _sanitize_local_tokenizer_config(args.model_path)
    _sanitize_local_model_config(args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("✓ Processor loaded\n")

    model_load_path = _resolve_base_model_path_for_dynamic(args.model_path)
    _sanitize_local_model_config(model_load_path)
    print(f"Loading model with Transformers from: {model_load_path}")
    model_load_is_local = Path(model_load_path).expanduser().exists()
    model = AutoModelForImageTextToText.from_pretrained(
        model_load_path,
        trust_remote_code=True,
        local_files_only=model_load_is_local,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    _apply_dynamic_patch_and_reload_weights(model, args.model_path)
    model.eval()
    model_device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {model_device}\n")
    timers, timer_counts, timer_handles = _install_component_timers(model)

    print("="*80)
    print("🚀 Running Transformers inference")
    print("="*80)
    start_time = time.time()
    
    # Save results (streaming to avoid high RAM usage)
    print("Saving results (streaming)...")
    num_results = 0
    profile_rows = []
    with open(args.output_file, 'w') as f:
        for idx, (_, line) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Generating")):
            line_dict = line.to_dict()
            for k, v in line_dict.items():
                if isinstance(v, np.integer):
                    line_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    line_dict[k] = float(v)

            messages = build_mmmu_prompt(line, dump_image_func, args.dataset)
            if args.use_cot and len(messages) > 0 and len(messages[0]['content']) > 0:
                last_content = messages[0]['content'][-1]
                if last_content['type'] == 'text':
                    last_content['text'] += cot_prompt

            model_inputs = _prepare_hf_inputs(messages, processor, model_device)
            input_len = model_inputs["input_ids"].shape[1]
            pre_embed = timers["input_embedding_sec"]
            pre_vision = timers["vision_encoder_sec"]
            pre_proj = timers["projection_sec"]
            _sync_if_cuda()
            t_gen0 = time.time()
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            _sync_if_cuda()
            t_gen1 = time.time()
            gen_ids = generated[:, input_len:]
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
            f.write(json.dumps(result) + '\n')
            num_results += 1

            emb_t = max(0.0, timers["input_embedding_sec"] - pre_embed)
            vis_t = max(0.0, timers["vision_encoder_sec"] - pre_vision)
            proj_t = max(0.0, timers["projection_sec"] - pre_proj)
            total_t = max(0.0, t_gen1 - t_gen0)
            dec_t = max(0.0, total_t - emb_t - vis_t - proj_t)
            profile_rows.append(
                {
                    "index": int(index) if isinstance(index, np.integer) else index,
                    "generate_total_sec": total_t,
                    "input_embedding_sec": emb_t,
                    "vision_encoder_sec": vis_t,
                    "projection_sec": proj_t,
                    "llm_decoder_sec": dec_t,
                }
            )

    total_time = time.time() - start_time
    _remove_hooks(timer_handles)
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(data):.2f} seconds/sample")
    print(f"  Throughput: {len(data)/total_time:.2f} samples/second\n")

    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {num_results}")

    if args.profile_output:
        if profile_rows:
            agg_total = float(sum(x["generate_total_sec"] for x in profile_rows))
            agg_embed = float(sum(x["input_embedding_sec"] for x in profile_rows))
            agg_vision = float(sum(x["vision_encoder_sec"] for x in profile_rows))
            agg_proj = float(sum(x["projection_sec"] for x in profile_rows))
            agg_dec = float(sum(x["llm_decoder_sec"] for x in profile_rows))
        else:
            agg_total = agg_embed = agg_vision = agg_proj = agg_dec = 0.0

        timing_payload = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": num_results,
            "total": {
                "generate_total_sec": agg_total,
                "input_embedding_sec": agg_embed,
                "vision_encoder_sec": agg_vision,
                "projection_sec": agg_proj,
                "llm_decoder_sec": agg_dec,
            },
            "avg_per_sample": {
                "generate_total_sec": (agg_total / num_results) if num_results else 0.0,
                "input_embedding_sec": (agg_embed / num_results) if num_results else 0.0,
                "vision_encoder_sec": (agg_vision / num_results) if num_results else 0.0,
                "projection_sec": (agg_proj / num_results) if num_results else 0.0,
                "llm_decoder_sec": (agg_dec / num_results) if num_results else 0.0,
            },
            "calls": timer_counts,
            "per_sample": profile_rows,
        }
        with open(args.profile_output, "w", encoding="utf-8") as pf:
            json.dump(timing_payload, pf, ensure_ascii=False, indent=2)
        print(f"✓ Component timing saved to {args.profile_output}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    _ = os.environ['LMUData']
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
    # If not choice label, then use lower case
    for k in data.keys():
        data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

    # Load dataset
    meta = load_dataset(args.dataset)

    # Validation
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset MMMU_DEV_VAL'
        )

    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    data = MMMU_preproc(data)
    answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}
    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []
    for i in range(len(data)):
        item = data.iloc[i]
        items.append(item)

    # Build judge model
    model = build_judge(
        model=getattr(args, 'eval_model', 'gpt-3.5-turbo-0125'),
        api_type=getattr(args, 'api_type', 'dash')
    )
    
    # Prepare evaluation tasks
    eval_tasks = []
    for item in items:
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
    
    # Calculate overall accuracy
    accuracy = sum(r['hit'] for r in eval_results) / len(eval_results)
    
    # Calculate accuracy by split
    results_by_split = {}
    for result in eval_results:
        split = result.get('split', 'unknown')
        if split not in results_by_split:
            results_by_split[split] = []
        results_by_split[split].append(result)
    
    accuracy_by_split = {}
    for split, split_results in results_by_split.items():
        split_accuracy = sum(r['hit'] for r in split_results) / len(split_results)
        accuracy_by_split[split] = split_accuracy
        print(f"Accuracy for {split} split: {split_accuracy:.4f} ({sum(r['hit'] for r in split_results)}/{len(split_results)})")
    
    # Save results
    output_df = pd.DataFrame(eval_results)
    output_df.to_csv(args.output_file, index=False)
    
    # Save accuracy
    with open(args.output_file.replace('.csv', '_acc.json'), 'w') as f:
        json.dump({
            "overall_accuracy": accuracy,
            "accuracy_by_split": accuracy_by_split
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="MMMU Evaluation with Transformers")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with Transformers")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    infer_parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process (for testing)")
    infer_parser.add_argument("--profile-output", type=str, default=None, help="Write component timing JSON to this path")
    infer_parser.add_argument(
        "--data-dir",
        type=str,
        default="/media/chenxi/ISC/VIVID/LMUData",
        help="LMUData root",
    )
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # Compatibility parameters (kept for existing launch scripts)
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None, 
                            help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                            help="Maximum model context length (default: 128000, balance between performance and memory)")
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
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default="/media/chenxi/ISC/VIVID/LMUData",
        help="LMUData root",
    )
    eval_parser.add_argument("--input-file", type=str, required=True, help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    eval_parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL", help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-3.5-turbo-0125",
                            help="Model to use for evaluation (default: gpt-3.5-turbo-0125)")
    eval_parser.add_argument("--api-type", type=str, default="dash", choices=["dash", "mit"],
                            help="API type for evaluation")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()
