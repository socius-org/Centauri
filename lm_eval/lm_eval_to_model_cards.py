#!/usr/bin/env python3
"""
Convert lm_eval results to HuggingFace model cards.

This script groups JSON files by model name, then creates comprehensive outputs
for each model including both model-index YAML and .eval_results/ files.

Usage:
    # Process all models, merge their benchmarks, output to directory
    python lm_eval_to_model_cards.py lm_eval/cogsoc/ lm_eval/metabench/ --output-dir model_cards/
    
    # Example result:
    # model_cards/
    # ├── Llama-Centaur-1B/
    # │   ├── README_model_card.yaml      (model-index format for README.md)
    # │   └── .eval_results/              (new HF leaderboard format)
    # │       ├── stanfordnlp_coqa.yaml
    # │       ├── ybisk_piqa.yaml
    # │       └── ...
    # ├── Llama-Centaur-8B/
    # │   ├── README_model_card.yaml
    # │   └── .eval_results/
    # └── Qwentaur-14B/
    #     ├── README_model_card.yaml
    #     └── .eval_results/
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


# Map lm_eval task names to HuggingFace dataset identifiers
# Prioritizes original/canonical dataset sources on HuggingFace Hub
# (lm_eval internal paths shown in comments for reference)
DATASET_MAPPING = {
    # Ethics benchmarks
    'ethics_cm': 'hendrycks/ethics',  # lm_eval uses: EleutherAI/hendrycks_ethics
    'ethics_deontology': 'hendrycks/ethics',  # lm_eval uses: EleutherAI/hendrycks_ethics
    'ethics_justice': 'hendrycks/ethics',  # lm_eval uses: EleutherAI/hendrycks_ethics
    'ethics_utilitarianism': 'hendrycks/ethics',  # lm_eval uses: EleutherAI/hendrycks_ethics
    'ethics_virtue': 'hendrycks/ethics',  # lm_eval uses: EleutherAI/hendrycks_ethics
    
    # Language modeling
    'lambada_openai': 'EleutherAI/lambada_openai',  # Original source
    'lambada_standard': 'cimec/lambada',  # Original source
    
    # Reasoning
    'logiqa': 'EleutherAI/logiqa',  # Original source
    'piqa': 'ybisk/piqa',  # lm_eval uses: baber/piqa
    'social_iqa': 'allenai/social_i_qa',  # Original source
    
    # QA
    'coqa': 'stanfordnlp/coqa',  # lm_eval uses: EleutherAI/coqa
    
    # Planning benchmarks (ACP Bench)
    'acp_app_bool': 'ibm-research/acp_bench',
    'acp_app_mcq': 'ibm-research/acp_bench',
    'acp_areach_bool': 'ibm-research/acp_bench',
    'acp_areach_mcq': 'ibm-research/acp_bench',
    'acp_just_bool': 'ibm-research/acp_bench',
    'acp_just_mcq': 'ibm-research/acp_bench',
    'acp_land_bool': 'ibm-research/acp_bench',
    'acp_land_mcq': 'ibm-research/acp_bench',
    'acp_prog_bool': 'ibm-research/acp_bench',
    'acp_prog_mcq': 'ibm-research/acp_bench',
    'acp_reach_bool': 'ibm-research/acp_bench',
    'acp_reach_mcq': 'ibm-research/acp_bench',
    'acp_val_bool': 'ibm-research/acp_bench',
    'acp_val_mcq': 'ibm-research/acp_bench',
    
    # EQ-Bench
    'eq_bench': 'pbevan11/EQ-Bench',  # Original source (canonical)
    
    # Metabench suite (all variants use same dataset)
    'metabench': 'HCAI/metabench',
    'metabench_arc': 'HCAI/metabench',
    'metabench_gsm8k': 'HCAI/metabench',
    'metabench_hellaswag': 'HCAI/metabench',
    'metabench_mmlu': 'HCAI/metabench',
    'metabench_truthfulqa': 'HCAI/metabench',
    'metabench_winogrande': 'HCAI/metabench',
    
    # TinyBenchmarks suite
    'tinyBenchmarks': 'tinyBenchmarks',  # Group task, usually no metrics
    'tinyArc': 'tinyBenchmarks/tinyAI2_arc',
    'tinyGSM8k': 'tinyBenchmarks/tinyGSM8k',
    'tinyHellaswag': 'tinyBenchmarks/tinyHellaswag',
    'tinyMMLU': 'tinyBenchmarks/tinyMMLU',
    'tinyTruthfulQA': 'tinyBenchmarks/tinyTruthfulQA',
    'tinyWinogrande': 'tinyBenchmarks/tinyWinogrande',
    
    # MMLU and variants
    'mmlu': 'cais/mmlu',
    
    # GSM8K
    'gsm8k': 'openai/gsm8k',  # lm_eval may use: gsm8k
    
    # ARC
    'arc_easy': 'allenai/ai2_arc',  # lm_eval uses: ai2_arc
    'arc_challenge': 'allenai/ai2_arc',  # lm_eval uses: ai2_arc
    
    # HellaSwag
    'hellaswag': 'Rowan/hellaswag',  # lm_eval uses: hellaswag
    
    # WinoGrande
    'winogrande': 'allenai/winogrande',  # lm_eval uses: winogrande
    
    # TruthfulQA
    'truthfulqa_mc1': 'truthfulqa/truthful_qa',  # lm_eval uses: truthful_qa
    'truthfulqa_mc2': 'truthfulqa/truthful_qa',  # lm_eval uses: truthful_qa
}

# Task type mapping for model-index
TASK_TYPE_MAPPING = {
    # Ethics
    'ethics_cm': 'text-classification',
    'ethics_deontology': 'text-classification',
    'ethics_justice': 'text-classification',
    'ethics_utilitarianism': 'text-classification',
    'ethics_virtue': 'text-classification',
    
    # Language modeling
    'lambada_openai': 'text-generation',
    'lambada_standard': 'text-generation',
    
    # Reasoning
    'logiqa': 'multiple-choice',
    'piqa': 'multiple-choice',
    'social_iqa': 'multiple-choice',
    
    # QA
    'coqa': 'question-answering',
    
    # EQ-Bench
    'eq_bench': 'text-generation',
    
    # Metabench suite
    'metabench': 'text-generation',
    'metabench_arc': 'multiple-choice',
    'metabench_gsm8k': 'text-generation',
    'metabench_hellaswag': 'multiple-choice',
    'metabench_mmlu': 'multiple-choice',
    'metabench_truthfulqa': 'multiple-choice',
    'metabench_winogrande': 'multiple-choice',
    
    # TinyBenchmarks suite
    'tinyBenchmarks': 'text-generation',
    'tinyArc': 'multiple-choice',
    'tinyGSM8k': 'text-generation',
    'tinyHellaswag': 'multiple-choice',
    'tinyMMLU': 'multiple-choice',
    'tinyTruthfulQA': 'multiple-choice',
    'tinyWinogrande': 'multiple-choice',
    
    # Standard benchmarks
    'mmlu': 'multiple-choice',
    'gsm8k': 'text-generation',
    'arc_easy': 'multiple-choice',
    'arc_challenge': 'multiple-choice',
    'hellaswag': 'multiple-choice',
    'winogrande': 'multiple-choice',
    'truthfulqa_mc1': 'multiple-choice',
    'truthfulqa_mc2': 'multiple-choice',
}

# Primary metrics to extract (ignoring stderr versions)
PRIMARY_METRICS = {
    'acc': 'Accuracy',
    'acc_norm': 'Normalized Accuracy',
    'exact_match': 'Exact Match',
    'em': 'Exact Match',
    'f1': 'F1 Score',
    'eqbench': 'EQ-Bench Score',
    'perplexity': 'Perplexity',
}


def find_json_files(paths: List[str]) -> List[Path]:
    """Find all JSON files from given paths (files or directories)."""
    json_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file() and path.suffix == '.json':
            json_files.append(path)
        elif path.is_dir():
            # Recursively find all .json files
            json_files.extend(path.rglob('*.json'))
        else:
            print(f"Warning: {path_str} is not a valid file or directory", file=sys.stderr)
    
    return sorted(json_files)


def extract_model_name(json_path: Path) -> Optional[str]:
    """Extract model name from JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        model_name = data.get('config', {}).get('model_args', {}).get('pretrained')
        if not model_name:
            model_name = data.get('model_name')
        
        return model_name
    except Exception as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        return None


def group_files_by_model(json_files: List[Path]) -> Dict[str, List[Path]]:
    """Group JSON files by their model name."""
    model_groups = defaultdict(list)
    
    for json_file in json_files:
        model_name = extract_model_name(json_file)
        if model_name:
            model_groups[model_name].append(json_file)
        else:
            print(f"Warning: Could not extract model name from {json_file}, skipping", file=sys.stderr)
    
    return dict(model_groups)


def merge_model_results(json_paths: List[Path], model_name: str) -> Optional[Dict]:
    """Merge multiple lm_eval JSON outputs for one model into unified model-index."""
    
    # Collect all tasks and their metrics across all JSON files
    all_tasks = defaultdict(dict)
    
    for json_path in json_paths:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}", file=sys.stderr)
            continue
        
        results_dict = data.get('results', {})
        
        # Merge results
        for task_name, task_results in results_dict.items():
            # Store all metrics for this task
            all_tasks[task_name].update(task_results)
    
    if not all_tasks:
        print(f"Error: No results found for {model_name}", file=sys.stderr)
        return None
    
    # Group tasks by dataset
    dataset_tasks = defaultdict(list)
    
    for task_name, task_results in all_tasks.items():
        # Get dataset identifier
        dataset_id = DATASET_MAPPING.get(task_name, task_name)
        
        # Extract metrics (skip stderr versions)
        metrics = []
        for metric_key, metric_value in task_results.items():
            if metric_key == 'alias':
                continue
            
            # Skip stderr metrics
            if '_stderr' in metric_key:
                continue
            
            # Parse metric name
            if ',' in metric_key:
                metric_name, _ = metric_key.split(',', 1)
            else:
                metric_name = metric_key
            
            # Skip if metric_name is empty, whitespace-only, or metric_value is None/invalid
            if not metric_name or not metric_name.strip() or metric_value is None:
                continue
            
            # Get human-readable metric name
            metric_display = PRIMARY_METRICS.get(metric_name, metric_name.replace('_', ' ').title())
            
            # Skip if display name is empty or whitespace-only
            if not metric_display or not metric_display.strip():
                continue
            
            # Convert to percentage for accuracy/match metrics
            if metric_name in ['acc', 'acc_norm', 'exact_match', 'em', 'f1']:
                metric_value = round(metric_value * 100, 2)
            elif metric_name == 'perplexity':
                metric_value = round(metric_value, 2)
            elif metric_name == 'eqbench':
                metric_value = round(metric_value, 2)
            
            metrics.append({
                'name': f"{metric_display} ({task_name})" if 'acp_' in task_name or 'ethics_' in task_name else metric_display,
                'type': metric_name,
                'value': metric_value
            })
        
        if metrics:
            dataset_tasks[dataset_id].extend(metrics)
    
    # Build unified model-index structure
    model_index = []
    
    for dataset_id, metrics in dataset_tasks.items():
        # Skip if no valid metrics
        if not metrics:
            continue
        # Determine task type
        task_type = 'text-generation'  # default
        for task_name in all_tasks.keys():
            if DATASET_MAPPING.get(task_name) == dataset_id:
                task_type = TASK_TYPE_MAPPING.get(task_name, 'text-generation')
                break
        
        result = {
            'task': {
                'type': task_type,
            },
            'dataset': {
                'name': dataset_id.split('/')[-1].replace('_', ' ').title(),
                'type': dataset_id,
            },
            'metrics': metrics
        }
        
        model_index.append(result)
    
    return {
        'model_name': model_name.split('/')[-1] if '/' in model_name else model_name,
        'model_path': model_name,
        'model_index': [{
            'name': model_name.split('/')[-1] if '/' in model_name else model_name,
            'results': model_index
        }],
        'source_files': [str(p) for p in json_paths]
    }


def format_yaml(data: dict, indent: int = 0) -> str:
    """Format dictionary as YAML with proper indentation."""
    lines = []
    prefix = '  ' * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_yaml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}- ")
                    # Format dict items with proper alignment
                    dict_lines = format_yaml(item, indent + 1).rstrip().split('\n')
                    lines.append(dict_lines[0].replace(prefix + '  ', prefix + '  '))
                    for line in dict_lines[1:]:
                        lines.append(line)
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            if isinstance(value, str):
                lines.append(f"{prefix}{key}: {value}")
            else:
                lines.append(f"{prefix}{key}: {value}")
    
    return '\n'.join(lines)


def write_model_outputs(output_dir: Path, model_data: Dict):
    """Write both model-index YAML and .eval_results/ files for a model."""
    # Create model subfolder
    model_name = model_data['model_name'].replace('/', '_').replace(' ', '_')
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Write model-index YAML (for README.md)
    model_card_path = model_dir / 'README_model_card.yaml'
    write_model_card_yaml(model_card_path, model_data)
    
    # 2. Write .eval_results/ files
    eval_results_dir = model_dir / '.eval_results'
    eval_results_dir.mkdir(exist_ok=True)
    write_eval_results_files(eval_results_dir, model_data)
    
    return model_dir


def write_model_card_yaml(output_path: Path, model_data: Dict):
    """Write model card YAML (model-index format) to file."""
    with open(output_path, 'w') as f:
        f.write("---\n")
        
        # Only write model-index
        model_index_data = {'model-index': model_data['model_index']}
        f.write(format_yaml(model_index_data))
        f.write("\n---\n\n")
        
        # Add comments
        f.write(f"# {model_data['model_name']}\n\n")
        f.write(f"Model-index format for {model_data['model_path']}\n\n")
        f.write("Copy the YAML section above (between the `---` markers) to the top of your model card README.md\n")


def write_yaml_list_item(f, d: dict):
    """Write a dictionary as a YAML list item with correct indentation."""
    items = list(d.items())
    
    if not items:
        f.write('- {}\n')
        return
    
    # First key-value pair on the same line as the dash
    first_key, first_value = items[0]
    
    if isinstance(first_value, dict):
        f.write(f'- {first_key}:\n')
        # Nested content indented 4 spaces from line start (2 for list + 2 for nesting)
        for nested_key, nested_value in first_value.items():
            f.write(f'    {nested_key}: {nested_value}\n')
    else:
        f.write(f'- {first_key}: {first_value}\n')
    
    # Remaining key-value pairs indented 2 spaces (to align under the dash)
    for key, value in items[1:]:
        if isinstance(value, dict):
            f.write(f'  {key}:\n')
            for nested_key, nested_value in value.items():
                f.write(f'    {nested_key}: {nested_value}\n')
        else:
            f.write(f'  {key}: {value}\n')


def write_eval_results_files(eval_results_dir: Path, model_data: Dict):
    """Write .eval_results/*.yaml files (new HuggingFace format)."""
    
    # Group metrics by dataset
    dataset_results = defaultdict(list)
    
    for result in model_data['model_index'][0]['results']:
        dataset_id = result['dataset']['type']
        metrics = result['metrics']
        
        # Check if this has subtasks (e.g., ethics_cm, acp_app_bool)
        has_subtasks = any('(' in m['name'] and ')' in m['name'] for m in metrics)
        
        if has_subtasks:
            # Create one entry per subtask
            for metric in metrics:
                task_id = extract_task_id_from_metric(metric)
                if task_id:
                    entry = {
                        'dataset': {
                            'id': dataset_id,
                            'task_id': task_id
                        },
                        'value': metric['value']
                    }
                    dataset_results[dataset_id].append(entry)
        else:
            # Single task - select primary metric
            primary_metric = select_primary_metric(metrics)
            
            if primary_metric:
                entry = {
                    'dataset': {
                        'id': dataset_id,
                    },
                    'value': primary_metric['value']
                }
                
                dataset_results[dataset_id].append(entry)
    
    # Write one file per dataset
    for dataset_id, entries in dataset_results.items():
        # Create safe filename from dataset ID
        safe_filename = dataset_id.replace('/', '_').replace('-', '_') + '.yaml'
        filepath = eval_results_dir / safe_filename
        
        with open(filepath, 'w') as f:
            # Write each entry as a YAML list item
            for entry in entries:
                write_yaml_list_item(f, entry)


def select_primary_metric(metrics: list) -> dict:
    """Select the primary/most important metric from a list."""
    # Priority order for metric types
    priority = [
        'acc_norm',      # Normalized accuracy (preferred)
        'exact_match',   # Exact match (for QA tasks)
        'em',           # Exact match (alternative)
        'f1',           # F1 score (for QA tasks)
        'acc',          # Accuracy
        'eqbench',      # EQ-Bench score
        'perplexity',   # Perplexity
    ]
    
    # Try to find metric by priority
    for metric_type in priority:
        for metric in metrics:
            if metric['type'] == metric_type:
                return metric
    
    # Fallback: return first metric
    return metrics[0] if metrics else None


def extract_task_id_from_metric(metric: dict) -> str:
    """Extract task ID from a single metric's name if it has a subtask identifier."""
    # For tasks with subtask identifiers in the name
    # e.g., "Accuracy (ethics_cm)" -> task_id = "ethics_cm"
    # e.g., "Exact Match (acp_app_bool)" -> task_id = "acp_app_bool"
    
    name = metric['name']
    if '(' in name and ')' in name:
        # Extract text between parentheses
        task_id = name[name.index('(')+1:name.index(')')]
        return task_id
    
    return None


def write_dict_as_yaml(f, d: dict, indent: int = 0):
    """Write a dictionary as YAML with proper indentation."""
    prefix = '  ' * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            f.write(f"{prefix}{key}:\n")
            write_dict_as_yaml(f, value, indent + 1)
        elif isinstance(value, (int, float)):
            f.write(f"{prefix}{key}: {value}\n")
        else:
            f.write(f"{prefix}{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert lm_eval results to HuggingFace model cards (both model-index and .eval_results/ formats).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models in directories, output to subfolders
  python lm_eval_to_model_cards.py lm_eval/cogsoc/ lm_eval/metabench/ --output-dir model_cards/
  
  # Result:
  # model_cards/
  # ├── Llama-Centaur-1B/
  # │   ├── README_model_card.yaml      (paste into README.md)
  # │   └── .eval_results/              (upload to model repo)
  # │       ├── stanfordnlp_coqa.yaml
  # │       ├── ybisk_piqa.yaml
  # │       └── ...
  # ├── Llama-Centaur-8B/
  # └── Qwentaur-14B/
        """
    )
    
    parser.add_argument('paths', nargs='+', help='JSON files or directories containing JSON files')
    parser.add_argument('--output-dir', '-d', required=True, help='Output directory for model card YAML files')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Find all JSON files
    json_files = find_json_files(args.paths)
    
    if not json_files:
        print("Error: No JSON files found", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(json_files)} JSON file(s)", file=sys.stderr)
    
    # Group files by model
    model_groups = group_files_by_model(json_files)
    
    if not model_groups:
        print("Error: Could not group files by model", file=sys.stderr)
        sys.exit(1)
    
    if not args.quiet:
        print(f"\nGrouped into {len(model_groups)} model(s):", file=sys.stderr)
        for model_name, files in model_groups.items():
            print(f"  {model_name}: {len(files)} file(s)", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    results = []
    for model_name, model_files in model_groups.items():
        if not args.quiet:
            print(f"Processing {model_name}...", file=sys.stderr)
        
        # Merge all results for this model
        merged_data = merge_model_results(model_files, model_name)
        
        if merged_data:
            # Write both model-index and .eval_results/ files
            model_dir = write_model_outputs(output_dir, merged_data)
            
            results.append((model_name, model_dir, len(model_files)))
            
            if not args.quiet:
                print(f"  ✓ Created {model_dir}/", file=sys.stderr)
                print(f"    - README_model_card.yaml (model-index format)", file=sys.stderr)
                print(f"    - .eval_results/*.yaml ({len(list((model_dir / '.eval_results').glob('*.yaml')))} files)", file=sys.stderr)
    
    # Summary
    if not args.quiet:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"✓ Successfully processed {len(results)} model(s):", file=sys.stderr)
        for model_name, model_dir, num_files in results:
            print(f"  {model_name}: merged {num_files} file(s) → {model_dir}/", file=sys.stderr)
        print(f"\nOutputs saved to: {output_dir}/", file=sys.stderr)
        print(f"\nFor each model, created:", file=sys.stderr)
        print(f"  - README_model_card.yaml (paste into README.md)", file=sys.stderr)
        print(f"  - .eval_results/*.yaml (upload to model repo)", file=sys.stderr)


if __name__ == "__main__":
    main()
