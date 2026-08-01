#!/usr/bin/env python3
"""
Evaluate models on ablated Psych-101 prompts.

Reads pre-ablated JSONL files produced by ablate_prompts.py and evaluates
each (condition, task) pair using the same SFTTrainer pipeline as eval_model.py.

Evaluates all 32 ablation experiments across five conditions:
  original, instruction_ablated, content_masked, history_only, choice_only

Usage:
    # Evaluate all conditions (requires ablate_prompts.py output)
    python eval_ablation.py --model socius/Qwentaur-0.6B-LoRA --ablated_data_dir ./ablated_data

    # Evaluate specific conditions only
    python eval_ablation.py --model socius/Qwentaur-8B --ablated_data_dir ./ablated_data \
        --conditions original content_masked history_only

    # Evaluate specific tasks only
    python eval_ablation.py --model socius/Qwentaur-8B --ablated_data_dir ./ablated_data \
        --tasks peterson2021using bahrami2020four

    # With 4-bit quantization
    python eval_ablation.py --model ./outputs/Llama-Centaur-1B-LoRA --ablated_data_dir ./ablated_data \
        --load_in_4bit
"""
import argparse
import os
from pathlib import Path


# ================================================================
# Pre-parse --backend and --model from argv BEFORE any heavy imports.
#
# Identical to eval_model.py — Unsloth monkey-patches transformers/trl
# at import time, which is required for the unsloth backend but breaks
# the transformers backend.
# ================================================================
def _pre_parse_backend():
    """Resolve backend from CLI args before importing heavy packages."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", default="auto", choices=["auto", "unsloth", "transformers"])
    pre.add_argument("--model", default="")
    args, _ = pre.parse_known_args()

    if args.backend != "auto":
        return args.backend

    model_lower = args.model.lower().rstrip("/")
    is_local = os.path.exists(args.model) if args.model else False
    if (is_local
        or model_lower.endswith("adapter")
        or model_lower.endswith("lora")
        or model_lower.startswith("unsloth/")):
        return "unsloth"
    return "transformers"


_BACKEND = _pre_parse_backend()

if _BACKEND == "unsloth":
    import unsloth  # noqa: F401 — must precede HF imports
    from unsloth import FastLanguageModel
else:
    FastLanguageModel = None  # not used in transformers backend

# Now safe to import HF packages (unsloth patches already applied, or skipped)
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import pandas as pd
import torch
import warnings
import numpy as np
from typing import List, Union, Optional, Any, Dict


# ================================================================
# DataCollatorForCompletionOnlyLM
# Copy-pasted from old trl version 0.8.6 (deprecated in new trl)
# IDENTICAL to eval_model.py — no changes
# ================================================================
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks instruction/prompt tokens in the labels, so that
    the loss is computed only on response (completion) tokens.

    Copied from trl 0.8.6 to ensure identical loss masking as original Centaur.
    See eval_model.py for full documentation.
    """
    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch


# ================================================================
# Helper function (response/instruction token IDs)
# IDENTICAL to eval_model.py
# ================================================================
def _detect_model_family(tokenizer) -> str:
    """
    Detect model family from tokenizer class/config rather than name string.
    
    Checks (in order):
        1. Tokenizer class name (e.g. Qwen2Tokenizer, LlamaTokenizer)
        2. model_type from tokenizer config (e.g. 'qwen2', 'llama')
        3. Fallback to name_or_path (least reliable, but better than crashing)
    
    Returns:
        'qwen' or 'llama'
    """
    # 1. Tokenizer class name (most reliable — survives renames and merges)
    cls_name = type(tokenizer).__name__.lower()
    if "qwen" in cls_name:
        return "qwen"
    if "llama" in cls_name:
        return "llama"
    
    # 2. model_type from tokenizer's init_kwargs or config
    model_type = getattr(tokenizer, "model_type", None)
    if model_type is None:
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        model_type = init_kwargs.get("model_type", "")
    model_type = str(model_type).lower()
    if "qwen" in model_type:
        return "qwen"
    if "llama" in model_type:
        return "llama"
    
    # 3. Fallback: name_or_path (least reliable — custom names may not contain family)
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        return "qwen"
    
    return "llama"  # default assumption


def get_response_template_ids(tokenizer):
    """Get token IDs for response template based on model family."""
    # Detect model family from tokenizer class/config (not name string)
    is_qwen = _detect_model_family(tokenizer) == "qwen"

    if is_qwen:
        l_id = tokenizer(" <<", add_special_tokens=False).input_ids
        r_id = tokenizer(">>", add_special_tokens=False).input_ids
    else:
        l_id = tokenizer(" <<").input_ids[1:]
        r_id = tokenizer(">>").input_ids[1:]

    return l_id, r_id


# ================================================================
# Task list — all 32 ablation experiments
# ================================================================
TASK_NAMES = [
    # Original 23
    "badham2017deficits",
    "bahrami2020four",
    "collsiöö2023MCPL",
    "feng2021dynamics",
    "frey2017cct",
    "garcia2023experiential",
    "gershman2018deconstructing",
    "krueger2022identifying",
    "lefebvre2017behavioural",
    "peterson2021using",
    "plonsky2018when",
    "sadeghiyeh2020temporal",
    "schulz2020finding",
    "somerville2017charting",
    "speekenbrink2008learning",
    "steingroever2015data",
    "waltz2020differential",
    "wilson2014humans",
    "wise2019acomputational",
    "wu2018generalisation",
    "wulff2018description",
    "wulff2018sampling",
    "xiong2023neural",
    # 9 additional (previously Tier 2/4, included for evaluation ablation)
    "flesch2018comparing",
    "gershman2020reward",
    "hilbig2014generalized",
    "kool2016when",
    "kool2017cost",
    "levering2020revisiting",
    "tomov2020discovery",
    "tomov2021multitask",
    "zorowitz2023data",
]

# Tasks with variable-length, multi-token responses (same as eval_model.py)
VARIABLE_LENGTH_TASKS = {
    'collsiöö2023MCPL',
    'garcia2023experiential',
    'krueger2022identifying',
    'wise2019acomputational',
    'wu2018generalisation',
}

CONDITIONS = [
    "original",
    "instruction_ablated",
    "content_masked",
    "history_only",
    "choice_only",
]


# ================================================================
# Custom metrics for variable-length, multi-token responses
# IDENTICAL to eval_model.py
# ================================================================
def preprocess_logits_for_metrics(logits, labels):
    """Compute per-response loss sums (not per-token)."""
    with torch.no_grad():
        logits = logits.cpu()
        labels = labels.cpu()

        labels = torch.cat((labels[0, 1:], -100 * torch.ones(1).long()), 0)
        logits = logits[0]

        ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

        total_loss = []
        item_loss = 0
        item_counter = 0

        for i in range(ce.shape[0]):
            if labels[i] != -100:
                item_loss += ce[i]
                item_counter += 1
            else:
                if item_counter != 0:
                    total_loss.append(item_loss)
                    item_loss = 0
                    item_counter = 0

        return torch.Tensor(total_loss)


def compute_metrics_per_response(pred):
    """Average over responses (not tokens) for fair comparison."""
    return {'custom_loss': pred.predictions.mean()}


# ================================================================
# Evaluate — modified from eval_model.py
# ================================================================
def evaluate(
    model_path: str,
    ablated_data_dir: str,
    output_dir: str = "./results",
    load_in_4bit: bool = False,
    device_map: str = None,
    max_seq_length: int = 32768,
    backend: str = "auto",
    conditions: List[str] = None,
    tasks: List[str] = None,
):
    """
    Evaluate a model on ablated Psych-101 prompts.

    Args:
        model_path: Local path or HuggingFace model ID
        ablated_data_dir: Directory containing JSONL files from ablate_prompts.py
        output_dir: Directory to save results CSV
        load_in_4bit: Use 4-bit quantization
        device_map: Device placement strategy
        max_seq_length: Maximum sequence length
        backend: 'auto', 'unsloth', or 'transformers'
        conditions: List of conditions to evaluate (default: all five)
        tasks: List of task names to evaluate (default: all 32)
    """
    if conditions is None:
        conditions = CONDITIONS
    if tasks is None:
        tasks = list(TASK_NAMES)

    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).name if os.path.exists(model_path) else model_path.replace("/", "-")

    print("\n" + "=" * 60)
    print("  Psych-101 Ablation Evaluation")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Ablated data: {ablated_data_dir}")
    print(f"  Conditions: {conditions}")
    print(f"  Tasks: {len(tasks)} experiments")
    print(f"  Load in 4bit: {load_in_4bit}")
    print(f"  Device map: {device_map or 'single GPU'}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Output: {output_dir}/{model_name}_ablation.csv")
    print("=" * 60)

    # ============================================
    # Resolve backend (same as eval_model.py)
    # ============================================
    if backend == "auto":
        model_path_lower = model_path.lower().rstrip("/")
        is_local = os.path.exists(model_path)
        if (is_local
            or model_path_lower.endswith("adapter")
            or model_path_lower.endswith("lora")
            or model_path_lower.startswith("unsloth/")):
            backend = "unsloth"
        else:
            backend = "transformers"
        print(f"  Auto-detected backend: {backend}")

    # ============================================
    # Load Model (same as eval_model.py)
    # ============================================
    print(f"\nLoading model (backend={backend})...")
    if backend == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )
    else:
        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            **quant_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            max_length=max_seq_length,
            truncation=True,
        )

    # ============================================
    # Setup tokenizer and collator (same as eval_model.py)
    # ============================================
    # Some models don't define a pad token.
    # The data collator requires one for batching, so fall back to eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    l_id, r_id = get_response_template_ids(tokenizer)

    model_family = _detect_model_family(tokenizer)
    print(f"  Model family: {model_family.capitalize()}")

    if not l_id or not r_id:
        raise ValueError("Tokenizer produced empty token lists. Check model compatibility.")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )

    # Store quantization state
    has_quantized_attr = hasattr(model, 'is_quantized')
    if has_quantized_attr:
        is_quantized = model.is_quantized

    # ============================================
    # Evaluate: loop over conditions, then tasks
    # ============================================
    print("\nEvaluating...")
    data = []
    total_evals = len(conditions) * len(tasks)
    eval_idx = 0

    with torch.no_grad():
        for condition in conditions:
            # ------------------------------------------
            # Load dataset for this condition
            # ------------------------------------------
            jsonl_path = os.path.join(ablated_data_dir, f"{condition}.jsonl")
            if not os.path.exists(jsonl_path):
                print(f"\n  Skipping condition '{condition}': {jsonl_path} not found")
                eval_idx += len(tasks)
                continue

            print(f"\n  Loading {condition}.jsonl...")
            dataset = load_dataset("json", data_files=jsonl_path)["train"]
            print(f"  Loaded {len(dataset):,} samples for condition '{condition}'")

            # ------------------------------------------
            # Evaluate each task under this condition
            # ------------------------------------------
            for task_name in tasks:
                eval_idx += 1

                eval_dataset = dataset.filter(
                    lambda example: example['experiment'].startswith(task_name)
                )

                if len(eval_dataset) == 0:
                    print(f"  [{eval_idx}/{total_evals}] {condition}/{task_name}: "
                          f"No samples found, skipping")
                    continue

                # ---- SFTTrainer setup (same as eval_model.py) ----
                if has_quantized_attr:
                    model.is_quantized = False

                is_variable_length = task_name in VARIABLE_LENGTH_TASKS

                if is_variable_length:
                    training_args = SFTConfig(
                        output_dir="eval_tmp",
                        per_device_eval_batch_size=1,
                        eval_accumulation_steps=1,
                        report_to="none",
                        dataset_text_field="text",
                        max_length=max_seq_length,
                        dataset_num_proc=1,
                    )
                    trainer = SFTTrainer(
                        model=model,
                        processing_class=tokenizer,
                        args=training_args,
                        train_dataset=eval_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=collator,
                        compute_metrics=compute_metrics_per_response,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                    )
                    if load_in_4bit:
                        trainer.args.bf16_full_eval = False
                        trainer.args.fp16_full_eval = False
                else:
                    training_args = SFTConfig(
                        output_dir="eval_tmp",
                        per_device_eval_batch_size=1,
                        report_to="none",
                        dataset_text_field="text",
                        max_length=max_seq_length,
                        dataset_num_proc=1,
                    )
                    trainer = SFTTrainer(
                        model=model,
                        processing_class=tokenizer,
                        args=training_args,
                        train_dataset=eval_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=collator,
                    )
                    if load_in_4bit:
                        trainer.args.bf16_full_eval = False
                        trainer.args.fp16_full_eval = False

                # ---- Evaluate ----
                result = trainer.evaluate()

                if has_quantized_attr:
                    model.is_quantized = is_quantized

                if is_variable_length:
                    loss = result['eval_custom_loss']
                else:
                    loss = result['eval_loss']

                print(f"  [{eval_idx}/{total_evals}] {condition}/{task_name}: "
                      f"loss={loss:.4f} (n={len(eval_dataset)})")
                data.append([task_name, condition, loss])

            # Free memory for this condition's dataset
            del dataset

    # ============================================
    # Save results
    # ============================================
    df = pd.DataFrame(data, columns=['task', 'condition', 'loss'])
    output_path = os.path.join(output_dir, f"{model_name}_ablation.csv")
    df.to_csv(output_path, index=False)

    # ============================================
    # Print summary
    # ============================================
    print("\n" + "=" * 70)
    print("  Ablation Results Summary")
    print("=" * 70)

    print(f"\n  {'Condition':25s} {'Mean Loss':>10s} {'Std':>8s} {'n':>5s}")
    print("  " + "-" * 52)
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        if cond_df.empty:
            continue
        print(f"  {cond:25s} {cond_df['loss'].mean():10.4f} {cond_df['loss'].std():8.4f} {len(cond_df):5d}")

    # Delta vs original
    if 'original' in conditions and len(df) > 0:
        orig = df[df['condition'] == 'original'].set_index('task')['loss']
        print(f"\n  Delta vs Original:")
        print(f"  {'Condition':25s} {'Mean Delta':>10s}")
        print("  " + "-" * 38)
        for cond in conditions:
            if cond == 'original':
                continue
            cond_losses = df[df['condition'] == cond].set_index('task')['loss']
            common = orig.index.intersection(cond_losses.index)
            if len(common) > 0:
                delta = (cond_losses[common] - orig[common]).mean()
                print(f"  {cond:25s} {delta:+10.4f}")

    print("\n" + "=" * 70)
    print(f"  Results saved to {output_path}")
    print("=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on ablated Psych-101 prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all conditions
  python eval_ablation.py --model socius/Qwentaur-8B --ablated_data_dir ./ablated_data

  # Specific conditions
  python eval_ablation.py --model socius/Qwentaur-8B --ablated_data_dir ./ablated_data \\
      --conditions original content_masked history_only

  # Specific tasks
  python eval_ablation.py --model socius/Qwentaur-8B --ablated_data_dir ./ablated_data \\
      --tasks peterson2021using bahrami2020four

  # Local adapter with 4-bit
  python eval_ablation.py --model ./outputs/Llama-Centaur-1B-LoRA --ablated_data_dir ./ablated_data \\
      --load_in_4bit
        """
    )

    # Same args as eval_model.py
    parser.add_argument("--model", type=str, required=True,
                        help="Local path or HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results CSV (default: ./results)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Enable 4-bit quantization")
    parser.add_argument("--device_map", type=str, default=None,
                        help="Device map: 'auto', 'balanced', or None (default: None = single GPU)")
    parser.add_argument("--max_seq_length", type=int, default=32768,
                        help="Maximum sequence length (default: 32768)")
    parser.add_argument("--backend", type=str, choices=["auto", "unsloth", "transformers"],
                        default="auto",
                        help="Model loading backend (default: auto)")

    # New args for ablation evaluation
    parser.add_argument("--ablated_data_dir", type=str, required=True,
                        help="Directory containing JSONL files from ablate_prompts.py")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=CONDITIONS,
                        help="Conditions to evaluate (default: all five)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names to evaluate (default: all 32)")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        ablated_data_dir=args.ablated_data_dir,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        max_seq_length=args.max_seq_length,
        backend=args.backend,
        conditions=args.conditions,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
