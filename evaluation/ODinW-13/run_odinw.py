import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, OrderedDict
import torch

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

# pycocotools imports
from pycocotools.coco import COCO

# Local imports from refactored files
from dataset_utils import load_odinw_config, generate_odinw_jobs
from eval_utils import compute_metrics

DEFAULT_ODINW_DIR = "/media/chenxi/ISC/VIVID/ODinW-13"


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
    """Run inference on the ODinW dataset using Transformers (no vLLM)."""
    print("\n" + "="*80)
    print("🚀 ODinW Inference with Transformers")
    print("="*80 + "\n")
    
    # Generate task list
    question_list, datasets = generate_odinw_jobs(args.data_dir, args)
    if getattr(args, "max_samples", None) is not None and args.max_samples > 0:
        question_list = question_list[:args.max_samples]
        print(f"⚠️  Testing mode: Processing only first {len(question_list)} samples")
    print(f"✓ Generated {len(question_list)} inference jobs\n")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"\n⚙️  Generation parameters (Transformers):")
    print(f"   max_new_tokens={args.max_new_tokens}")
    print(f"   temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"   repetition_penalty={args.repetition_penalty}")
    print(f"   presence_penalty={args.presence_penalty}")
    print()
    
    # Load processor
    print(f"Loading processor from {args.model_path}")
    _sanitize_local_tokenizer_config(args.model_path)
    _sanitize_local_model_config(args.model_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("✓ Processor loaded\n")
    
    # Initialize HF model
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

    # Inference loop
    print("="*80)
    print("🚀 Running Transformers inference")
    print("="*80)
    start_time = time.time()

    end_time = time.time()

    # Save results
    print("Saving results...")
    results = []

    for idx, item in enumerate(tqdm(question_list, desc="Generating")):
        model_inputs = _prepare_hf_inputs(item['messages'], processor, model_device)
        input_len = model_inputs["input_ids"].shape[1]
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
        gen_ids = generated[:, input_len:]
        response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        
        # Handle </think> tag
        response_final = str(response).split("</think>")[-1].strip()
        
        result = {
            "question_id": item['question_id'],
            "annotation": item['annotation'],
            "extra_info": item['extra_info'],
            "result": {"gen": response_final, "gen_raw": response},
            "messages": item['messages']
        }
        results.append(result)

    total_time = time.time() - start_time
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(question_list):.2f} seconds/sample")
    print(f"  Throughput: {len(question_list)/total_time:.2f} samples/second\n")
    
    # Save results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")
    
    # Save dataset config (for evaluation)
    config_output = args.output_file.replace('.jsonl', '_datasets.json')
    with open(config_output, 'w') as f:
        # Convert config for JSON serialization
        datasets_serializable = {}
        for k, v in datasets.items():
            datasets_serializable[k] = {
                'metainfo': v['metainfo'],
                'data_root': v['data_root'],
                'ann_file': v['ann_file'],
                'data_prefix': v['data_prefix']
            }
        json.dump(datasets_serializable, f, indent=2)
    print(f"✓ Dataset config saved to {config_output}")


def run_evaluation(args):
    """Run evaluation on inference results."""
    print("\n" + "="*80)
    print("🎯 ODinW Evaluation")
    print("="*80 + "\n")
    
    # Load inference results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"✓ Loaded {len(results)} inference results\n")
    
    # Load dataset config
    config_path = os.path.join(args.data_dir, "odinw13_config.py")
    datasets = load_odinw_config(config_path)
    
    # Group by dataset
    all_outputs = defaultdict(list)
    for job in results:
        all_outputs[job["extra_info"]["dataset_name"]].append(job)
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, sub_jobs in all_outputs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")
        
        anno_path = sub_jobs[0]["extra_info"]["anno_path"]
        coco_api = COCO(anno_path)
        
        classes = datasets[dataset_name]['metainfo']['classes']
        pred_bboxes_per_img = defaultdict(list)
        
        for job in sub_jobs:
            img_id = job["extra_info"]["img_id"]
            resized_h = job["extra_info"]["resized_h"]
            resized_w = job["extra_info"]["resized_w"]
            img_h = job["extra_info"]["img_h"]
            img_w = job["extra_info"]["img_w"]
            
            answer = job['result']['gen']
            answer = answer.replace("```json", "")
            answer = answer.replace("```", "")
            
            # Parse predictions
            import ast
            import re
            
            try:
                json_data = ast.literal_eval(answer)
                pred_bboxes = []
                pred_labels = []
                for data in json_data:
                    if len(data.get("bbox_2d", [])) != 4:
                        continue
                    pred_bboxes.append(data["bbox_2d"])
                    pred_labels.append(data["label"])
            except Exception as e:
                # If parsing fails, use empty results
                pred_bboxes = []
                pred_labels = []
            
            # Coordinate conversion (from resized to original size)
            if os.getenv("is_rel", "0") == "1":
                pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / 1000 * np.array([img_w, img_h, img_w, img_h])
            else:
                if len(pred_bboxes) > 0:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / np.array([resized_w, resized_h, resized_w, resized_h]) * np.array([img_w, img_h, img_w, img_h])
                else:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
            
            pred_bboxes = pred_bboxes.tolist()
            
            # Group by category
            pred_objs = defaultdict(list)
            for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
                pred_objs[pred_label].append(pred_bbox)
            
            for k, v in pred_objs.items():
                class_names = [name.lower() for name in classes]
                if k.lower() not in class_names:
                    continue
                pred_bboxes_per_img[img_id].append({
                    'label': class_names.index(k.lower()), 
                    'bbox': v
                })
        
        # Prepare evaluation results
        pred_results = []
        for k, v in pred_bboxes_per_img.items():
            bboxes = []
            labels = []
            for tmp in v:
                bboxes.extend(tmp['bbox'])
                labels.extend([tmp['label']] * len(tmp['bbox']))
            
            height = coco_api.imgs[k]["height"]
            width = coco_api.imgs[k]["width"]
            
            pred_tuple = (
                {'width': width, 'height': height, 'img_id': k},
                {
                    'img_id': k,
                    'bboxes': np.array(bboxes),
                    'scores': np.array([1.0] * len(bboxes)),
                    'labels': np.array(labels),
                },
            )
            pred_results.append(pred_tuple)
        
        # Compute metrics
        eval_results = compute_metrics(pred_results, _coco_api=coco_api)
        print(f"{dataset_name}: {eval_results}")
        all_results[dataset_name] = eval_results
    
    # Summarize results
    results_ordered = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    results_display = []
    
    for prefix, result in results_ordered.items():
        results_display.append([prefix] + [result[k] for k in metric_items])
    
    # Calculate average
    average_scores = []
    for col_idx in range(len(metric_items)):
        average_scores.append(np.mean([line[col_idx + 1] for line in results_display]))
    results_display.append(['Average'] + average_scores)
    
    # Print results table
    try:
        from tabulate import tabulate
        print("\n" + "="*80)
        print(
            tabulate(
                results_display,
                headers=["ODinW13 Dataset"] + metric_items,
                tablefmt="fancy_outline",
                floatfmt=".3f",
            )
        )
        print("="*80 + "\n")
    except ImportError:
        print("\n" + "="*80)
        print("ODinW13 Results:")
        print("="*80)
        for row in results_display:
            print(row)
        print("="*80 + "\n")
    
    # Save results
    all_results.update({"Average": average_scores[0]})
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Evaluation results saved to {args.output_file}")
    print(f"\n{'='*80}")
    print(f"Final Average mAP: {average_scores[0]:.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="ODinW Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    infer_parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_ODINW_DIR,
        help="Path to ODinW data directory (containing odinw13_config.py)",
    )
    infer_parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    infer_parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process (for testing)")
    
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
                            help="Maximum number of tokens to generate (default: 32768)")
    infer_parser.add_argument("--temperature", type=float, default=0.7,
                            help="Temperature for sampling (default: 0.7)")
    infer_parser.add_argument("--top-p", type=float, default=0.8,
                            help="Top-p for sampling (default: 0.8)")
    infer_parser.add_argument("--top-k", type=int, default=20,
                            help="Top-k for sampling (default: 20)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_ODINW_DIR,
        help="Path to ODinW data directory (containing odinw13_config.py)",
    )
    eval_parser.add_argument("--input-file", type=str, required=True,
                           help="Input file with inference results")
    eval_parser.add_argument("--output-file", type=str, required=True,
                           help="Output file path")
    
    args = parser.parse_args()

    if hasattr(args, 'data_dir'):
        args.data_dir = str(args.data_dir).strip() or DEFAULT_ODINW_DIR
        args.data_dir = os.path.abspath(args.data_dir)
    
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

