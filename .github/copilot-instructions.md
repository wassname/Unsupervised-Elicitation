# Copilot Instructions for Unsupervised-Elicitation Project

## Project Overview
Simplified fork of [Unsupervised Elicitation](https://github.com/Jiaxin-Wen/Unsupervised-Elicitation) implementing Internal Coherence Maximization (ICM) - unsupervised label generation via in-context metalearning + consistency constraints + mutual predictability + simulated annealing.

**Key innovation**: No "leading the witness" prompts (removed "find truth"/"which is helpful"). Pure pattern completion for unsupervised elicitation.

Key hypothesis:
- LLM's logprobs are an internal only, non-calibrated measure of confidence. But with N-shots they are least use in context-learning
- But we can compare between predictions to get external measures of confidence

**Main file**: `src/simple_icm.py` - simplified ICM using OpenRouter API (async, logprobs). Reference implementations in `docs/`:

Reference files
- `docs/Unsupervised-Elicitation-orig_code/src/experiments/ICM.py` + `ICM_tools.py` - original paper code
- `docs/icm-replication/icm/core.py` - transformers-based replication
- `docs/paper.tex` - original paper

**Data flow**: Load dataset → Random init (num_seed examples) → Iterative predict/fix → Simulated annealing acceptance → Output labeled parquet

## Directory Structure
```
src/
  simple_icm.py          # Main entry point - edit Config dataclass for params
  data/                  # Dataset loaders with consistency_id/key for grouping
    truthfulqa.py
    daily_dilemmas.py

nbs/                     # VSCode notebooks (.py with %% cells) for analysis
  02_analyse_*.py        # Load outputs/, generate plots
  03_eval_tqa.py

outputs/icm/{dataset}/   # Results per run
  icm_final_labels.parquet
  icm_config.json
  icm_progress.png       # plot of accuracy vs iteration, and energy vs iteration
  reasoning.txt          # Model's explanations (if verbose=2)

docs/
  Unsupervised-Elicitation-orig_code/  # Original paper implementation
    src/experiments/ICM.py             # Main original logic
    src/experiments/ICM_tools.py       # Consistency fixing
  icm-replication/                     # Transformers-based replication
    icm/core.py
  paper.tex                            # Reference for algorithm details
```

## Non-Obvious Patterns
- **Setup**: `uv sync` then `uv run src/simple_icm.py` (NOT pip/conda)
- **Consistency groups**: consistency_id groups questions; consistency_key distinguishes paraphrases (must agree) vs contradictions (must oppose)
- **Reference original**: When confused, check docs/Unsupervised-Elicitation-orig_code/src/experiments/ICM.py for paper's actual implementation
- **Debug predictions**: Set verbose=2 in predict_label() → logs to outputs/*/reasoning.txt
- **Energy function**: Simplified from original - no LLM meta-reasoning in consistency fixing, just greedy k-flip enumeration
