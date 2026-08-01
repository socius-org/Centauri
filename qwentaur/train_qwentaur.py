# train_qwentaur.py
"""
Qwentaur: SFT Training for Qwen3 Models on Psych-101

Replicates the Centaur paper methodology using Qwen3 base models.
Includes Weights & Biases (wandb) integration for experiment tracking.

Usage:
    # Basic training
    python train_qwentaur.py --size 0.6B
    
    # With wandb tracking
    python train_qwentaur.py --size 0.6B --wandb --wandb_project qwentaur-rank-and-datasize --wandb_run qwen-0.6b-run1
    
    # With 4-bit quantization
    python train_qwentaur.py --size 14B --load_in_4bit

"""

import unsloth 
# unsloth recommends importing unsloth pacakge before importing hf packages (e.g., transformers)
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import argparse
import os
from datasets import load_dataset
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_psych101_fraction
import torch
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np
from typing import List, Union, Optional, Any, Dict

# ================================================================
# DataCollatorForCompletionOnlyLM
# Copy-pasted from old trl version 0.8.6 (deprecated in new trl)
# This ensures identical loss calculation as original Centaur code
# Masks loss on instruction tokens, only trains on response tokens
# ================================================================
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
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

# ============================================
# Model Configurations & Fixed Seed
# We use `-Base` (not `-Instruct` or `-GGUF`)
# * `-GGUF` models are designed for inference only 
# * `-Instruct` models are post-trained to be "helpful assistant"
# (starting with instruct means doing SFT on top of previous SFT)
# ============================================
QWEN_CONFIGS = {
    "0.6B": {
        "model_name": "unsloth/Qwen3-0.6B-Base",
        "lora_rank": 16,
    },
    "1.7B": {
        "model_name": "unsloth/Qwen3-1.7B-Base",
        "lora_rank": 16,
    },
    "4B": {
        "model_name": "unsloth/Qwen3-4B-Base",
        "lora_rank": 16,
    },
    "8B": {
        "model_name": "unsloth/Qwen3-8B-Base",
        "lora_rank": 16,
    },
    "14B": {
        "model_name": "unsloth/Qwen3-14B-Base",
        "lora_rank": 16,
    },
}

SEED = 3407

# ============================================
# Batch Size Configurations
# ============================================
def get_batch_config(size: str, load_in_4bit: bool, num_gpus: int):
    """
    Get optimal batch size and gradient accumulation based on model size
    and quantization, based on 1 x A100 GPU.
    """
    if load_in_4bit:
        # 4-bit quantization - fits on fewer GPUs
        configs = {
            "0.6B": {"batch_size": 4, "grad_accum": 8, "device_map": "auto"},
            "1.7B": {"batch_size": 2, "grad_accum": 16, "device_map": "auto"},
            "4B": {"batch_size": 2, "grad_accum": 16, "device_map": "auto"},
            "8B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
            "14B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
        }
    else:
        # bfloat16 - needs more memory
        configs = {
            "0.6B": {"batch_size": 2, "grad_accum": 16, "device_map": "auto"},
            "1.7B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
            "4B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
            "8B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
            "14B": {"batch_size": 1, "grad_accum": 32, "device_map": "auto"},
        }
    
    config = configs[size]
    
    # Use auto device_map for multi-GPU
    if num_gpus > 1:
        config["device_map"] = "auto"
    
    return config


def train(
    size: str,
    output_dir: str = None,
    load_in_4bit: bool = False,
    num_gpus: int = 1,
    # ablation arguments
    data_fraction: float = 1.0,
    indices_dir: str = "./psych101_fractions",
    lora_rank: int = None,
    # wandb arguments
    use_wandb: bool = False,
    wandb_project: str = "qwentaur-rank-and-datasize",
    wandb_run_name: str = None,
    wandb_entity: str = None,
    wandb_tags: List[str] = None,
    wandb_run_id: str = None,
    # crash recovery
    resume_from_checkpoint: str = None,
):
    """
    Train Qwentaur model on Psych-101 dataset.

    Args:
        size: Model size ("0.6B", "1.7B", "4B", "8B", "14B")
        output_dir: Output directory (default: ./outputs/qwentaur-{size})
        load_in_4bit: Use 4-bit quantization (QLoRA)
        num_gpus: Number of GPUs to use
        data_fraction: Fraction of Psych-101 to train on. 1.0 uses the full
            split (existing behaviour). Smaller values require nested
            stratified indices from build_dataset_fractions.py.
        indices_dir: Where to find fractions_indices.json (only used when
            data_fraction < 1.0).
        lora_rank: Override the default LoRA rank (16 for every size in
            QWEN_CONFIGS). With rsLoRA, lora_alpha is kept equal to rank.
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (default: auto-generated)
        wandb_entity: W&B entity (team/username)
        wandb_tags: List of tags for the run
    """

    if size not in QWEN_CONFIGS:
        raise ValueError(f"Invalid size: {size}. Choose from {list(QWEN_CONFIGS.keys())}")

    model_config = QWEN_CONFIGS[size]
    batch_config = get_batch_config(size, load_in_4bit, num_gpus)

    # Resolve LoRA rank: CLI override > model_config default.
    if lora_rank is None:
        lora_rank = model_config["lora_rank"]

    if output_dir is None:
        quant_suffix = "-4bit" if load_in_4bit else "-bf16"
        rank_suffix = f"-r{lora_rank}" if lora_rank != model_config["lora_rank"] else ""
        frac_suffix = f"-f{data_fraction:g}" if abs(data_fraction - 1.0) > 1e-9 else ""
        output_dir = f"./outputs/qwentaur-{size.lower()}{rank_suffix}{frac_suffix}{quant_suffix}"

    # Training hyperparameters
    learning_rate = 5e-5
    
    # ============================================
    # Initialize wandb
    # ============================================
    if use_wandb:
        try:
            import wandb
            
            # Auto-generate run name if not provided
            if wandb_run_name is None:
                quant_str = "4bit" if load_in_4bit else "bf16"
                wandb_run_name = f"qwen-{size.lower()}-{quant_str}"
            
            # Resuming a crashed run: reattach to the existing wandb run so
            # the loss curve continues at the checkpoint's global step
            # instead of starting a new run. resume="must" fails loudly if
            # the id does not exist (a silent new run would fork history).
            resume_kwargs = {}
            if wandb_run_id is not None:
                resume_kwargs = {"id": wandb_run_id, "resume": "must"}

            # Initialize wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                **resume_kwargs,
                tags=wandb_tags or [f"qwen-{size}", "qwentaur", "psych-101"],
                config={
                    "model_family": "qwen3",
                    "model_size": size,
                    "model_name": model_config["model_name"],
                    "load_in_4bit": load_in_4bit,
                    "lora_rank": lora_rank,
                    "data_fraction": data_fraction,
                    "indices_dir": indices_dir,
                    "learning_rate": learning_rate,
                    "batch_size": batch_config["batch_size"],
                    "gradient_accumulation_steps": batch_config["grad_accum"],
                    "effective_batch_size": batch_config["batch_size"] * batch_config["grad_accum"],
                    "max_seq_length": 32768,
                    "num_gpus": num_gpus,
                    "seed": SEED,
                },
            )
            print(f"\n✓ Weights & Biases initialized: {wandb_project}/{wandb_run_name}")
            
        except ImportError:
            print("\n⚠ wandb not installed. Run: pip install wandb")
            print("  Continuing without wandb logging...")
            use_wandb = False
    
    print("\n" + "=" * 60)
    print(f"  Qwentaur-{size}")
    print("=" * 60)
    print(f"  Model: {model_config['model_name']}")
    print(f"  Output: {output_dir}")
    print(f"  LoRA rank: {lora_rank}  (alpha = rank, rsLoRA)")
    print(f"  Data fraction: {data_fraction:g}")
    print(f"  Indices dir: {indices_dir}")
    print(f"  Load in 4bit: {load_in_4bit}")
    print(f"  Max seq length: 32768")
    print(f"  Batch size: {batch_config['batch_size']}")
    print(f"  Gradient accumulation: {batch_config['grad_accum']}")
    print(f"  Effective batch size: {batch_config['batch_size'] * batch_config['grad_accum']}")
    print(f"  Num GPUs: {num_gpus}")
    print(f"  Device map: {batch_config['device_map']}")
    print(f"  Seed: {SEED}")
    print(f"  Wandb: {'Enabled' if use_wandb else 'Disabled'}")
    print("=" * 60)
    
    # ============================================
    # Load Dataset
    # ============================================
    print("\nLoading Psych-101 dataset...")
    train_dataset = load_psych101_fraction(
        fraction=data_fraction,
        seed=SEED,
        indices_dir=indices_dir,
    )
    print(f"  Train: {len(train_dataset):,} samples  (fraction={data_fraction:g})")
    # In-training eval is disabled for the ablation runs (eval_strategy="no").
    # Final paper-grade evaluation against marcelbinz/Psych-101-test happens
    # post-hoc via ../eval_model.py against every saved adapter.
    
    # ============================================
    # Load Model
    # ============================================
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["model_name"],
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        device_map=batch_config["device_map"],
    )
    
    # ============================================
    # Add LoRA Adapters
    # * LoRA rank is the dimensionality of the low-rank matrices that approximate weight updates.
    # * LoRA alpha (scaling factor) is a multiplier that controls how much the LoRA adaptation influences 
    # the output relative to the frozen base weights. Higher alpha means the adapter has more impact, and it's typically set equal to rank or 2× rank.
    #
    # `rslora`: rank-stabilized LoRA 
    # * LoRA alpha is typically 2x rank, but we set lora_alpha = lora_rank since we are using rslora
    # * Standard LoRA: output = base_output + adapter_output * (alpha / r)
    # * rsLoRA: output = base_output + adapter_output * (alpha / sqrt(r))
    # * With rsLoRA, we set alpha = r, since the sqrt scaling amplifies the adapter
    # ============================================
    print(f"Adding LoRA adapters (r={lora_rank})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=True, # rank-stabilized LoRA (rsLoRA)
        loftq_config=None,
    )
    model.print_trainable_parameters()
    
    # ============================================
    # Tokenizer Setup
    # Note: Qwen3 doesn't add BOS token by default
    # ============================================
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    
    # ============================================
    # Data Collator
    # IMPORTANT: Use add_special_tokens=False for Qwen3
    # (Qwen3 doesn't add BOS, so [1:] trick doesn't work)
    # ============================================
    l_id = tokenizer(" <<", add_special_tokens=False).input_ids
    r_id = tokenizer(">>", add_special_tokens=False).input_ids
    
    # print(f"\nResponse template tokens: {l_id} -> '{tokenizer.decode(l_id)}'")
    # print(f"Instruction template tokens: {r_id} -> '{tokenizer.decode(r_id)}'")
    
    if not l_id:
        raise ValueError("Response template ' <<' produced empty token list!")
    if not r_id:
        raise ValueError("Instruction template '>>' produced empty token list!")
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,
    )
    
    # ============================================
    # Training Arguments
    # ============================================
    training_args = UnslothTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_config["batch_size"],
        gradient_accumulation_steps=batch_config["grad_accum"],
        num_train_epochs=1,
        learning_rate=learning_rate,
        embedding_learning_rate=learning_rate / 10,
        warmup_ratio=0.05,  # ~5% of training, auto-scales with data_fraction
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_strategy="steps",
        logging_steps=10,  # More frequent logging for wandb
        log_level="info",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        # Wandb integration
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_run_name if use_wandb else None,
    )
    
    # ============================================
    # Trainer
    # ============================================
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=32768,
        dataset_num_proc=8,
        data_collator=collator,
        args=training_args,
    )
    
    # ============================================
    # Train
    # ============================================
    if resume_from_checkpoint:
        print(f"\nResuming training from {resume_from_checkpoint}...")
    else:
        print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # ============================================
    # Save
    # ============================================
    print("\nSaving model...")
    trainer.save_model()
    
    # ============================================
    # Finish wandb run
    # ============================================
    if use_wandb:
        import wandb
        wandb.finish()
        print("✓ Wandb run finished")
    
    print(f"\nQwentaur-{size} training complete!")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwentaur (Qwen3) on Psych-101",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_qwentaur.py --size 0.6B
  
  # With 4-bit quantization (recommended for 8B/14B)
  python train_qwentaur.py --size 8B --load_in_4bit
  
  # With wandb tracking
  python train_qwentaur.py --size 0.6B --wandb --wandb_project my-project
  
  # Full example with all options
  python train_qwentaur.py --size 14B --load_in_4bit --num_gpus 2 \\
      --wandb --wandb_project qwentaur-rank-and-datasize --wandb_run qwen-14b-experiment \\
      --wandb_tags experiment1 baseline
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        choices=["0.6B", "1.7B", "4B", "8B", "14B"],
        help="Model size: 0.6B, 1.7B, 4B, 8B, or 14B"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./outputs/qwentaur-{size})"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA). Recommended for 8B and 14B models."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )

    # Ablation arguments
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Fraction of Psych-101 to train on (see build_dataset_fractions.py). "
             "1.0 = full split (default, identical to original behaviour)."
    )
    parser.add_argument(
        "--indices_dir",
        type=str,
        default="./psych101_fractions",
        help="Where to read fractions_indices.json from "
             "(only used when --data_fraction < 1.0)."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="Override LoRA rank (default: 16 for every size). "
             "With rsLoRA, lora_alpha is kept equal to rank."
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwentaur-rank-and-datasize",
        help="Wandb project name (default: qwentaur-rank-and-datasize)"
    )
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (team or username)"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="Wandb tags for the run (space-separated)"
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Existing wandb run id to resume (e.g. 28m8bu7y). Use together "
             "with --resume_from_checkpoint so the curve continues in the "
             "same run instead of starting a new one."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a COMPLETE checkpoint dir (must contain "
             "trainer_state.json and optimizer.pt) to resume training from, "
             "e.g. ./outputs/.../checkpoint-1000."
    )

    args = parser.parse_args()

    train(
        size=args.size,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        num_gpus=args.num_gpus,
        data_fraction=args.data_fraction,
        indices_dir=args.indices_dir,
        lora_rank=args.lora_rank,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        wandb_entity=args.wandb_entity,
        wandb_tags=args.wandb_tags,
        wandb_run_id=args.wandb_run_id,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )