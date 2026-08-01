# Metabench Evaluation Results

This directory contains evaluation results on the **metabench** benchmark suite, a collection of general capability benchmarks used to assess whether cognitive fine-tuning degrades performance on standard language model evaluations.

## Overview

The metabench suite (Kipnis et al., 2025) provides a sparse but representative sample of reasoning and knowledge benchmarks from the machine learning literature. We use these benchmarks to quantify the "capability tax" incurred by fine-tuning language models on human behavioural data from psychological experiments.

## Models Evaluated

### Qwen Family

| Model | Parameters | Type | Description |
|-------|------------|------|-------------|
| Qwen3-0.6B | 0.6B | Base | Qwen3 base model |
| Qwen3-1.7B | 1.7B | Base | Qwen3 base model |
| Qwen3-4B | 4B | Base | Qwen3 base model |
| Qwen3-8B | 8B | Base | Qwen3 base model |
| Qwen3-14B | 14B | Base | Qwen3 base model |
| Qwentaur-0.6B | 0.6B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Qwentaur-1.7B | 1.7B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Qwentaur-4B | 4B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Qwentaur-8B | 8B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Qwentaur-14B | 14B | Fine-tuned | Cognitive fine-tuned on Psych-101 |

### Llama Family

| Model | Parameters | Type | Description |
|-------|------------|------|-------------|
| Llama-3.2-1B | 1B | Base | Llama 3.2 base model |
| Llama-3.2-3B | 3B | Base | Llama 3.2 base model |
| Llama-3.1-8B | 8B | Base | Llama 3.1 base model |
| Centaur-1B | 1B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Centaur-3B | 3B | Fine-tuned | Cognitive fine-tuned on Psych-101 |
| Centaur-8B | 8B | Fine-tuned | Cognitive fine-tuned on Psych-101 |

## Benchmark Descriptions

### ARC (AI2 Reasoning Challenge)

A dataset of genuine grade-school level, multiple-choice science questions assembled to encourage research in advanced question-answering. The Challenge Set contains only questions answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. The dataset comprises 7,787 questions partitioned into Challenge and Easy sets, testing scientific reasoning abilities that go beyond simple information retrieval.

- **Task type:** Multiple-choice question answering
- **Metric:** `acc_norm` (length-normalised accuracy)
- **Hugging Face:** [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)
- **Paper:** Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457). *arXiv:1803.05457*

### GSM8K (Grade School Math 8K)

A dataset of 8,500 high-quality, linguistically diverse grade school maths word problems created by human problem writers. Each problem requires between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations. The dataset tests multi-step mathematical reasoning, requiring models to chain together multiple operations to arrive at the correct answer.

- **Task type:** Mathematical reasoning with chain-of-thought
- **Metric:** `exact_match` (flexible extract)
- **Hugging Face:** [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- **Paper:** Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). *arXiv:2110.14168*

### HellaSwag

A challenge dataset for evaluating commonsense natural language inference. Given an event description such as "A woman sits at a piano," a machine must select the most likely follow-up from four choices. The dataset uses Adversarial Filtering (AF) to create challenging distractors that are grammatical and topical but ultimately incorrect. Human performance reaches approximately 95%, whilst state-of-the-art models at the time of release achieved only around 48%, demonstrating a significant gap in commonsense understanding.

- **Task type:** Sentence completion / commonsense NLI
- **Metric:** `acc_norm` (length-normalised accuracy)
- **Hugging Face:** [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag)
- **Paper:** Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019). [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830). *ACL 2019*

### MMLU (Massive Multitask Language Understanding)

A comprehensive benchmark covering 57 subjects across STEM, humanities, social sciences, and more. Subjects range from elementary mathematics to professional law and medicine, testing both world knowledge and problem-solving ability at varying difficulty levels. The benchmark includes tasks spanning abstract algebra, anatomy, astronomy, business ethics, clinical knowledge, computer science, econometrics, formal logic, global facts, jurisprudence, machine learning, moral scenarios, nutrition, philosophy, professional accounting, public relations, sociology, virology, and world religions, amongst others.

- **Task type:** Multiple-choice question answering
- **Metric:** `acc` (accuracy)
- **Hugging Face:** [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)
- **Paper:** Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300). *ICLR 2021*

### TruthfulQA

A benchmark designed to measure whether a language model generates truthful answers to questions. The benchmark comprises 817 questions spanning 38 categories, including health, law, finance, politics, and conspiracies. Questions are crafted such that some humans would answer falsely due to misconceptions or false beliefs—the benchmark specifically targets "imitative falsehoods" that models might learn from human-generated text. The MC1 (single-true) task variant requires selecting the single correct answer from multiple choices.

- **Task type:** Truthfulness evaluation
- **Metric:** `acc` (accuracy on MC1 single-true task)
- **Hugging Face:** [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa)
- **Paper:** Lin, S., Hilton, J., & Evans, O. (2022). [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958). *ACL 2022*

### Winogrande

A large-scale dataset of 44,000 problems inspired by the original Winograd Schema Challenge (WSC), designed to test commonsense reasoning. Problems require resolving pronoun references in sentences where the correct referent depends on world knowledge rather than syntactic cues. The dataset was constructed using a novel crowdsourcing procedure with careful adversarial filtering to reduce annotation artefacts. For example, given "The trophy doesn't fit in the suitcase because it is too big," the model must determine whether "it" refers to the trophy or the suitcase.

- **Task type:** Pronoun resolution / commonsense reasoning
- **Metric:** `acc` (accuracy)
- **Hugging Face:** [`allenai/winogrande`](https://huggingface.co/datasets/allenai/winogrande)
- **Paper:** Sakaguchi, K., Le Bras, R., Bhagavatula, C., & Choi, Y. (2020). [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641). *AAAI 2020*

## Evaluation Command

All evaluations were conducted using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.

```bash
lm_eval --model hf \
    --model_args pretrained=<model_path>,dtype=bfloat16 \
    --tasks metabench_arc,metabench_gsm8k,metabench_hellaswag,metabench_mmlu,metabench_truthfulqa,metabench_winogrande \
    --batch_size auto \
    --output_path ./results/<model_name>_metabench.json
```

For quantised models:

```bash
lm_eval --model hf \
    --model_args pretrained=<model_path>,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16 \
    --tasks metabench_arc,metabench_gsm8k,metabench_hellaswag,metabench_mmlu,metabench_truthfulqa,metabench_winogrande \
    --batch_size auto \
    --output_path ./results/<model_name>_metabench.json
```

## File Structure

```
metabench/
├── README.md
├── Qwen3-0.6B_metabench.json
├── Qwen3-1.7B_metabench.json
├── Qwen3-4B_metabench.json
├── Qwen3-8B_metabench.json
├── Qwen3-14B_metabench.json
├── Qwentaur-0.6B_metabench.json
├── Qwentaur-1.7B_metabench.json
├── Qwentaur-4B_metabench.json
├── Qwentaur-8B_metabench.json
├── Qwentaur-14B_metabench.json
├── Llama-3.2-1B_metabench.json
├── Llama-3.2-3B_metabench.json
├── Llama-3.1-8B_metabench.json
├── Centaur-1B_metabench.json
├── Centaur-3B_metabench.json
└── Centaur-8B_metabench.json
```

## References

- Kipnis, A., Voudouris, K., Schulze Buschoff, L. M., & Schulz, E. (2025). metabench: A sparse benchmark of reasoning and knowledge in large language models. *ICLR 2025*.
- Binz, M., et al. (2025). A foundation model to predict and capture human cognition. *Nature, 644*, 1002–1009.
- Gao, L., et al. (2023). A framework for few-shot language model evaluation. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
