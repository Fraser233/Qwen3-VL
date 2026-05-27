# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Ensure external VIVID-Qwen3VL method package is importable.
qwen_root = Path(__file__).resolve().parents[3]
if str(qwen_root) not in sys.path:
    sys.path.insert(0, str(qwen_root))
vivid_root = qwen_root.parent / "VIVID-Qwen3VL"
if str(vivid_root) not in sys.path:
    sys.path.insert(0, str(vivid_root))

from trainer import replace_qwen2_vl_attention_class
from qwen3_vivid_patch import (
    apply_vivid_qwen3_native_compact_patch,
    apply_vivid_qwen3_vision_patch,
    count_vivid_qwen3_native_modules,
    count_vivid_qwen3_blocks,
    is_vivid_qwen3_native_patched,
    is_vivid_qwen3_vision_patched,
)

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')

    if data_args.model_type == "qwen3vl":
        vivid_enabled = os.environ.get("QWEN3_VIVID_ENABLED", "1") != "0"
        vivid_native = os.environ.get("QWEN3_VIVID_NATIVE", "1") != "0"
        vivid_compact_tokens = int(os.environ.get("QWEN3_VIVID_COMPACT_TOKENS", "128"))
        vivid_min_keep_ratio = float(os.environ.get("QWEN3_VIVID_MIN_KEEP_RATIO", "0.0"))
        vivid_anchors = int(os.environ.get("QWEN3_VIVID_ANCHORS", "256"))
        vivid_topk = int(os.environ.get("QWEN3_VIVID_TOPK", "0"))
        vivid_enable_kv = os.environ.get("QWEN3_VIVID_ENABLE_KV", "1") != "0"
        vivid_aggregation = os.environ.get("QWEN3_VIVID_AGGREGATION", "learned")
        vivid_profile = os.environ.get("QWEN3_VIVID_PROFILE", "0") == "1"
        if vivid_native:
            patched = apply_vivid_qwen3_native_compact_patch(
                model,
                compact_tokens=vivid_compact_tokens,
                min_keep_ratio=vivid_min_keep_ratio,
                kv_anchors=vivid_anchors,
                kv_topk=vivid_topk,
                enable_kv_compression=vivid_enable_kv,
                aggregation_mode=vivid_aggregation,
                profile=vivid_profile,
                enabled=vivid_enabled,
            )
            vivid_blocks = count_vivid_qwen3_native_modules(model)
            vivid_is_patched = is_vivid_qwen3_native_patched(model)
        else:
            patched = apply_vivid_qwen3_vision_patch(
                model,
                num_anchors=vivid_anchors,
                topk=vivid_topk,
                enabled=vivid_enabled,
            )
            vivid_blocks = count_vivid_qwen3_blocks(model)
            vivid_is_patched = is_vivid_qwen3_vision_patched(model)
        strict = os.environ.get("QWEN3_VIVID_STRICT", "1") != "0"
        if vivid_enabled and strict and not vivid_is_patched:
            raise RuntimeError(
                "VIVID patch requested but not applied to Qwen3-VL."
            )

        if patched or vivid_blocks > 0:
            rank0_print(
                f"VIVID Qwen3 status: native={vivid_native}, patched_now={patched}, vivid_modules={vivid_blocks}, "
                f"compact_tokens={vivid_compact_tokens}, min_keep_ratio={vivid_min_keep_ratio}, anchors={vivid_anchors}, topk={vivid_topk}, "
                f"kv={vivid_enable_kv}, aggregation={vivid_aggregation}, enabled={vivid_enabled}"
            )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
