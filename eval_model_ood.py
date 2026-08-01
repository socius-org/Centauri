#!/usr/bin/env python3
"""
Evaluate models on the Psych-201-RT test set (out-of-distribution: 18 held-out
experiments never seen in training).

Copy of results/Psych-201 (NLL)/eval_model.py for the ablation grids
(ADDITIONAL_EXPERIMENTS.md), with the get_response_template_ids fix so the
template IDs are correct for ALL families (Llama, Qwen3, SmolLM2/3, OLMo) --
the original's [1:] BOS-strip crashes on tokenizers that don't prepend BOS.

Drive it over the rank sweep / dataset-size adapters with:
    python schedule_eval_runs.py --family smollm \
        --eval_script ./eval_model_ood.py --output_dir ./eval_results_ood/smoltaur

Supports both LoRA/QLoRA adapters (via Unsloth) and merged HuggingFace models
(via transformers).

Works with:
- Local saved LoRA/QLoRA adapters (e.g., ./outputs/Llama-Centaur-1B-LoRA)
- LoRA adapters from HuggingFace (e.g., socius/Llama-Centaur-8B-LoRA) - automatically loads relevant base models
- Full merged models from HuggingFace (e.g., socius/Llama-Qwentaur-14B)
- All four model families: Llama, Qwen3, SmolLM2/SmolLM3, OLMo

Note on local models:
    Local saved models are typically LoRA/QLoRA adapters. When --backend is set to "auto"
    (the default), local paths default to Unsloth's FastLanguageModel for loading.
    For HuggingFace Hub models, auto-detection checks the model name for "adapter", "LoRA",
    or the "unsloth/" prefix to decide. 

Usage:
    # Evaluate local adapter model (auto-load with unsloth's FastLanguageModel)
    python eval_model.py --model ./outputs/Llama-Centaur-1B-LoRA
    
    # Evaluate merged HuggingFace model (auto-load with transformers' AutoModelForCausalLM)
    python eval_model.py --model socius/Llama-Centaur-1B

    # Evaluate merged HuggingFace model (with unsloth FastLanguageModel backend) 
"""
import argparse
import os
from pathlib import Path


# ================================================================
# Pre-parse --backend and --model from argv BEFORE any heavy imports.
# 
# Unsloth monkey-patches transformers/trl at import time.
# This is required when using the unsloth backend (and unsloth
# docs recommends importing it before any HF packages), but it breaks
# the transformers backend — the patched Trainer.evaluation_loop
# tries to cast quantized models to fp16, which bitsandbytes rejects.
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
# This ensures identical loss calculation as original Centaur 
# Masks loss on instruction tokens, only trains on response tokens
# ================================================================
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks instruction/prompt tokens in the labels, so that
    the loss is computed only on response (completion) tokens.
    
    In Psych-101, each sequence interleaves instructions and responses in the format:
    
        >> instruction_1 << response_1 >> instruction_2 << response_2 >> ...
    
    where ">>" marks the start of an instruction and "<<" marks the start of a
    response. This collator uses the token IDs of these delimiters to identify
    which tokens are instructions and which are responses, then sets all
    instruction token labels to -100 (PyTorch's ignore index). This way,
    cross-entropy loss is only computed on the model's predictions for response
    tokens — the actual behavioral choices.
    
    Two modes of operation:
        1. response_template only (instruction_template=None):
           Finds the last occurrence of the response delimiter and masks
           everything before it. Suitable for single-response sequences.
        
        2. Both response_template and instruction_template:
           Finds all instruction/response delimiter pairs and masks each
           instruction span individually. Required for multi-turn sequences
           (multiple instruction-response pairs in one sample), which is the
           standard format in Psych-101.
    
    Copied from trl 0.8.6 (deprecated in later versions) to ensure identical
    loss masking behavior as the original Centaur evaluation code:
    https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test_adapter.py
    https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/main/test_adapter_custom_metrics.py 
    
    Args:
        response_template: String or token ID list for the response delimiter ("<<").
        instruction_template: String or token ID list for the instruction delimiter (">>").
        mlm: Must be False (causal LM, not masked LM).
        ignore_index: Label value for masked tokens (default: -100, PyTorch convention).
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
    """
    Get token IDs for the response/instruction templates.

    Always encoded with add_special_tokens=False: the templates are searched
    for inside already-tokenized sequences, so they must never carry BOS.
    For tokenizers that do prepend BOS (Llama 3.x) this is equivalent to the
    old [1:] strip, and it matches the old Qwen path exactly. For tokenizers
    that don't (SmolLM2/SmolLM3, OLMo) the [1:] strip dropped a real token --
    ">>" is a single token there, leaving an empty template list.

    Returns:
        tuple: (response_template_ids, instruction_template_ids)
    """
    l_id = tokenizer(" <<", add_special_tokens=False).input_ids
    r_id = tokenizer(">>", add_special_tokens=False).input_ids
    return l_id, r_id


# ================================================================
# Task list (same as original Centaur evaluation)
# ================================================================
TASK_NAMES = [
    "anllo2024weird",
    "bavard2021range",
    "busch2024navon",
    "busch2024stroop",
    "castrorodrigues2022twostep",
    "fan2022trait",
    "franke2024bayesian",
    "frankedegen2016reasoning",
    "guenther2020ts",
    "guenther2023grammaticality",
    "palminteri2017confirmation",
    "rutledge2023happiness",
    "shahar2019twosteptask",
    "spektor2024lossaversion",
    "tsvilodub2023xorsome",
    "vandendriessche2022depression",
    "xu2023augmenting",
    "zika2023traitanxiety",
]

# Tasks with variable-length, multi-token responses
VARIABLE_LENGTH_TASKS = {
    'rutledge2023happiness',
    'tsvilodub2023xorsome',
    'zika2023traitanxiety',
}


# ================================================================
# Custom metrics for variable-length, multi-token responses
# ================================================================
def preprocess_logits_for_metrics(logits, labels):
    """
    Compute per-response loss sums (not per-token).
    This ensures each behavioral choice counts equally regardless of token length.
    
    For variable-length tasks, a single sequence may contain multiple responses of
    different token lengths (e.g., "<<[1, 2]>>" vs "<<0>>"). Standard per-token
    averaging would bias toward longer responses. Instead, we:
      1. Compute per-token cross-entropy losses
      2. Sum the losses within each response (contiguous non-ignored span)
      3. Return one summed loss per response
    These sums are later averaged across responses by compute_metrics_per_response.
    """
    with torch.no_grad():
        # Move to CPU to avoid GPU memory accumulation during eval
        logits = logits.cpu()
        labels = labels.cpu()
        
        # Shift labels left by 1 for autoregressive alignment:
        # position i's logits should predict the token at position i+1.
        # Pad the end with -100 (ignore index) since there's no next token.
        # PyTorch Convention: When you pass -100 to `cross_entropy`, any label with value -100 is skipped in the computation. 
        labels = torch.cat((labels[0, 1:], -100 * torch.ones(1).long()), 0)
        logits = logits[0]  # Remove batch dimension (batch_size=1)
        
        # Compute per-token cross-entropy (no reduction, one loss per position)
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        
        # Accumulate losses per response:
        # - Non-ignored tokens (labels != -100) belong to a response → sum their CE
        # - Ignored tokens (labels == -100) are boundaries between responses
        total_loss = []     # Will hold one summed loss per response
        item_loss = 0       # Running sum of CE within current response
        item_counter = 0    # Number of tokens in current response (for boundary detection)
        
        for i in range(ce.shape[0]):
            if labels[i] != -100:
                # Inside a response: accumulate the token's cross-entropy
                item_loss += ce[i]
                item_counter += 1
            else:
                # Hit an ignored token (instruction/padding boundary)
                if item_counter != 0:
                    # End of a response: save the summed loss and reset
                    total_loss.append(item_loss)
                    item_loss = 0
                    item_counter = 0
        
        # Return tensor of per-response summed losses
        return torch.Tensor(total_loss)


def compute_metrics_per_response(pred):
    """
    Average over responses (not tokens) for fair comparison.
    
    IMPORTANT: Returns 'custom_loss' (not 'eval_loss') because the HF Trainer
    unconditionally overwrites metrics["eval_loss"] with its own default per-token
    average after calling compute_metrics(). Using 'custom_loss' ensures our value
    survives as 'eval_custom_loss' in the results dict (the Trainer auto-prefixes
    keys that don't already start with 'eval_').
    
    See: test_adapter_custom_metrics.py in the original Centaur codebase.
    """
    return {'custom_loss': pred.predictions.mean()}


def evaluate(
    model_path: str,
    output_dir: str = "./results",
    load_in_4bit: bool = False,
    device_map: str = None,
    max_seq_length: int = 32768,
    backend: str = "auto",
):
    """
    Evaluate a fine-tuned model on Psych-201-RT test set.
    
    Args:
        model_path: Local path or HuggingFace model ID
        output_dir: Directory to save results CSV
        load_in_4bit: Use 4-bit quantization for evaluation
        device_map: Device placement strategy ('auto', 'balanced', or None)
        max_seq_length: Maximum sequence length
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model name for output file
    model_name = Path(model_path).name if os.path.exists(model_path) else model_path.replace("/", "-")
    
    print("\n" + "=" * 60)
    print("  Psych-201-RT Evaluation")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Load in 4bit: {load_in_4bit}")
    print(f"  Device map: {device_map or 'single GPU'}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Output: {output_dir}/{model_name}.csv")
    print(f"  Tasks: {len(TASK_NAMES)}")
    print("=" * 60)
    
    # ============================================
    # Resolve backend
    # ============================================
    if backend == "auto":
        model_path_lower = model_path.lower().rstrip("/")
        is_local = os.path.exists(model_path)
        if (is_local
            or model_path_lower.endswith("adapter")
            or model_path_lower.endswith("lora")
            or model_path_lower.startswith("unsloth/")):
            # Local paths are almost always LoRA/QLoRA adapters (unless you have abundance of GPUs and storage), so default to unsloth.
            # For HF Hub models, match names ending in "adapter"/"lora" or from "unsloth/".
            backend = "unsloth"
        else:
            backend = "transformers"
        print(f"  Auto-detected backend: {backend}")
    
    # ============================================
    # Load Model
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
    # Setup tokenizer and collator
    # ============================================
    # Some models (e.g. Qwen) don't define a pad token.
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
    
    # ============================================
    # Load dataset
    # ============================================
    print("\nLoading Psych-201-RT test dataset...")
    dataset = load_dataset("socius/Psych-201-RT")
    print(f"  Total samples: {len(dataset['test']):,}")
    
    # Store quantization state (only if attribute exists)
    has_quantized_attr = hasattr(model, 'is_quantized')
    if has_quantized_attr:
        is_quantized = model.is_quantized
    
    # ============================================
    # Evaluate each task
    # ============================================
    print("\nEvaluating tasks...")
    data = []
    
    with torch.no_grad():
        for i, task_name in enumerate(TASK_NAMES):
            # Filter dataset for this task
            eval_dataset = dataset['test'].filter(
                lambda example: example['experiment'].startswith(task_name)
            )
            
            if len(eval_dataset) == 0:
                print(f"  [{i+1}/{len(TASK_NAMES)}] {task_name}: No samples found, skipping")
                continue
            
            # Temporarily disable quantization flag so SFTTrainer init doesn't
            # refuse to create a trainer for a quantized model without adapters.
            # (The actual .to(dtype) crash is handled separately below via bf16_full_eval.)
            if has_quantized_attr:
                model.is_quantized = False
            
            # trl 0.24+ uses SFTConfig instead of TrainingArguments
            is_variable_length = task_name in VARIABLE_LENGTH_TASKS
            
            if is_variable_length:
                # Variable-length tasks: sum log-likelihoods within each response,
                # then average across responses (not tokens) for fair comparison.
                training_args = SFTConfig(
                    output_dir="eval_tmp",              # Required by HF Trainer but unused (eval-only)
                    per_device_eval_batch_size=1,        # One sample at a time (sequences are full-length)
                    eval_accumulation_steps=1,           # Accumulate predictions on CPU every step to avoid OOM
                    report_to="none",                    # Disable wandb/tensorboard logging
                    dataset_text_field="text",           # Column name in the dataset
                    max_length=max_seq_length,           # Safety cap matching model context window (None = no truncation)
                    dataset_num_proc=1,                  # Single-process tokenization (avoids multiprocessing issues)
                    # Note: bf16/fp16 are NOT set here because Unsloth overrides them
                    # during SFTTrainer.__init__. See the post-init override below.
                )
                
                trainer = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    args=training_args,
                    train_dataset=eval_dataset,          # Required by SFTTrainer even for eval-only
                    eval_dataset=eval_dataset,
                    data_collator=collator,              # Masks instruction tokens; loss only on responses
                    compute_metrics=compute_metrics_per_response,        # Average across per-response sums
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # Sum CE within each response
                )
                # Unsloth silently sets bf16=True and bf16_full_eval=True during init,
                # which causes evaluation_loop to call model.to(dtype=bf16) — crashing
                # on bitsandbytes-quantized base models. Override after creation.
                if load_in_4bit:
                    trainer.args.bf16_full_eval = False
                    trainer.args.fp16_full_eval = False
            else:
                # Standard tasks: use HF Trainer's default per-token cross-entropy averaging.
                training_args = SFTConfig(
                    output_dir="eval_tmp",              # Required by HF Trainer but unused (eval-only)
                    per_device_eval_batch_size=1,        # One sample at a time (sequences are full-length)
                    report_to="none",                    # Disable wandb/tensorboard logging
                    dataset_text_field="text",           # Column name in the dataset
                    max_length=max_seq_length,           # Safety cap matching model context window (None = no truncation)
                    dataset_num_proc=1,                  # Single-process tokenization (avoids multiprocessing issues)
                    # Note: bf16/fp16 are NOT set here because Unsloth overrides them
                    # during SFTTrainer.__init__. See the post-init override below.
                )
                
                trainer = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    args=training_args,
                    train_dataset=eval_dataset,          # Required by SFTTrainer even for eval-only
                    eval_dataset=eval_dataset,
                    data_collator=collator,              # Masks instruction tokens; loss only on responses
                )
                # Unsloth silently sets bf16=True and bf16_full_eval=True during init,
                # which causes evaluation_loop to call model.to(dtype=bf16) — crashing
                # on bitsandbytes-quantized base models. Override after creation.
                if load_in_4bit:
                    trainer.args.bf16_full_eval = False
                    trainer.args.fp16_full_eval = False
            
            # Evaluate
            result = trainer.evaluate()
            
            # Restore quantization flag
            if has_quantized_attr:
                model.is_quantized = is_quantized
            
            # Variable-length tasks: read 'eval_custom_loss' (from compute_metrics_per_response)
            # Standard tasks: read 'eval_loss' (Trainer's default per-token average)
            #
            # Why the distinction? The HF Trainer unconditionally sets metrics["eval_loss"]
            # to its own per-token-averaged CE loss. For variable-length tasks, this per-token
            # average is incorrect (it biases toward longer responses). Our custom pipeline
            # (preprocess_logits_for_metrics → compute_metrics_per_response) computes a
            # per-response average instead, stored under 'eval_custom_loss'.
            if is_variable_length:
                loss = result['eval_custom_loss']
            else:
                loss = result['eval_loss']
            
            print(f"  [{i+1}/{len(TASK_NAMES)}] {task_name}: loss={loss:.4f} (n={len(eval_dataset)}, variable_length={is_variable_length})")
            data.append([task_name, loss])
    
    # ============================================
    # Save results
    # ============================================
    df = pd.DataFrame(data, columns=['task', 'loss'])
    output_path = os.path.join(output_dir, f"{model_name}.csv")
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Mean loss: {df['loss'].mean():.4f}")
    print(f"  Std loss:  {df['loss'].std():.4f}")
    print(f"  Min loss:  {df['loss'].min():.4f} ({df.loc[df['loss'].idxmin(), 'task']})")
    print(f"  Max loss:  {df['loss'].max():.4f} ({df.loc[df['loss'].idxmax(), 'task']})")
    print("=" * 60)
    print(f"\n✓ Results saved to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned adapter on Psych-201-RT test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate local adapter model (auto-detected as unsloth)
  python eval_model.py --model ./outputs/Llama-Centaur-1B-LoRA
  
  # Evaluate socius HuggingFace model
  python eval_model.py --model socius/Qwentaur-8B --output_dir ./RT_results
  
  # Evaluate marcelbinz HuggingFace model
  python eval_model.py --model marcelbinz/Llama-3.1-Centaur-70B-adapter

Note:
  For multi-GPU inference with merged models (not adapters), 
  --backend transformers with --device_map auto is more stable than unsloth.
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results CSV (default: ./results)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit quantization (uses less memory but may affect accuracy)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map: 'auto', 'balanced' (multi-GPU), or None (default: None = single GPU)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32768,
        help="Maximum sequence length (default: 32768)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "unsloth", "transformers"],
        default="auto",
        help="Model loading backend. 'auto' uses unsloth for adapter/LoRA models, transformers otherwise (default: auto)"
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        max_seq_length=args.max_seq_length,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()