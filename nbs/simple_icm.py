# %% [markdown]
# Simplified ICM: Unsupervised Elicitation with Simulated Annealing
#
# This script implements a basic version of Internal Coherence Maximization (ICM)
# as a VSCode notebook (.py with # %% cells). Now integrated with user's OpenRouter wrapper.
#
# Run in VSCode for notebook view. Assumes OPENROUTER_API_KEY in .env.

# %% [code]
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
from anycache import anycache
from functools import lru_cache
import os, sys
import pandas as pd
from dataclasses import dataclass, asdict
import dotenv
from loguru import logger
from openrouter_wrapper.logprobs import openrouter_completion_wlogprobs, get_logprobs_choices, LogprobsNotSupportedError  # User's wrapper
from typing import List, Tuple
import asyncio
from aiocache import SimpleMemoryCache, cached, Cache

cache = SimpleMemoryCache()

try:
    from IPython import get_ipython
    if get_ipython() is not None:  # In Jupyter/VS Code notebook
        import nest_asyncio
        nest_asyncio.apply()
except ImportError:
    pass  # Not in notebook or IPython not available


dotenv.load_dotenv()


# Setup loguru
logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {message}", colorize=True, level="INFO")

# %% [code]

@dataclass
class Config:
    alpha: float = 30.0
    initial_t: float = 10.0
    final_t: float = 0.01
    decay_rate: float = 0.99
    beta: float = 2.0
    num_seed: int = 8
    max_iters: int = 950  # Small for demo; increase for more
    n_shots: int = 16  # Number of in-context examples
    model_id: str = "meta-llama/llama-3.1-8b-instruct"  # Logprobs supported
    provider_whitelist: Tuple[str] = None  # None to let OpenRouter choose
    out_dir: Path = Path("../outputs/icm")  # Directory to save outputs
    log_interval: int = 50  # Log progress every N iterations

C = Config(
    model_id="qwen/qwen3-235b-a22b-2507",
    provider_whitelist=[ 'Chutes','Nebius',], 
)
C.out_dir.mkdir(parents=True, exist_ok=True)

# C = Config(
#     model="qwen/qwen3-30b-a3b-instruct-2507",
#     provider_whitelist=[ 'Chutes','Nebius',],
# )

# C = Config(
#     model="meta-llama/llama-3.1-70b-instruct",
#     provider_whitelist=[ 'Cerebras','Nebius',],
# )
# C = Config(
#     model="meta-llama/llama-3.1-8b-instruct",
#     provider_whitelist=('Cerebras','Nebius',),
# )

logger.info(f"Config: {C}")


# %% [code]
from datasets import load_dataset

# Load larger HF dataset: Yik/truthfulQA-bool
dataset = load_dataset("wassname/truthful_qa_v2", split="validation", name="binary")
# semantic_anchor = "0 is false, 1 is true."
data = []
group_id = 0
groups = {}
for idx, item in enumerate(dataset):
    claim = f"Question: {item['Question']}\nClaim: {item['Answer']}\nI think this claim is"
    label = item['label']
    consistency_id = item['question_id']
    
    
    example = {
        'uid': idx,
        'prompt': claim,
        'vanilla_label': label,
        'consistency_id': consistency_id,
        'consistency_key': 'A' if label == 1 else 'B'
    }
    data.append(example)

# HACK: Limit to small batch for demo
# data = data[:512]
logger.info(f"Loaded {len(data)} examples from TruthfulQA-bool")

# %% [code]
# Initialize: Random labels for first num_seed, None for others
def initialize_data(data, config):
    demonstrations = {item['uid']: item.copy() for item in data}
    labeled_uids = random.sample(list(demonstrations.keys()), min(config.num_seed, len(data)))
    for uid in demonstrations:
        demonstrations[uid]['label'] = None
        demonstrations[uid]['score'] = 0.0  # Will store prediction score
        if uid in labeled_uids:
            demonstrations[uid]['label'] = random.choice([0, 1])
    return demonstrations

demonstrations = initialize_data(data, C)
logger.info("Initialized labels: {}", {k: v['label'] for k, v in demonstrations.items() if v['label'] is not None})

# %% [code]
# Predict label using in-context prompting

def print_messages(messages):
    return "\n".join([f"**{m['role'].upper()}**: {m['content']}" for m in messages])

@cached(cache)
async def cached_openrouter_completion_wlogprobs(*args, **kwargs):
    # TODO: Async caching (e.g., with aiocache)
    return await openrouter_completion_wlogprobs(*args, **kwargs)

async def predict_label(example_uid, full_demos, labeled_demos, config=C, verbose=False):
    # simplified from https://github.com/Jiaxin-Wen/Unsupervised-Elicitation/blob/master/src/experiments/ICM.py#L254
    # Sort by consistency_key for relevance, limit to 16
    relevant_demos = sorted(
        [demo for uid, demo in labeled_demos.items() if uid != example_uid],
        key=lambda x: x.get('consistency_key', '')
    )[:C.n_shots]
    # TODO: Use all/subsampled demos like original (group/sort by consistency_id for relevance), handling context limits (high value, med complexity).

    # FIXME: some LLM's have a positional bias, so we should randomize order

    instruction = "Carefully examine the noisy example pairs of (Candidate, Set). Then as consistently and coherently as possible complete the pattern, you may provide reasoning after.\n\n"
    # if verbose>1:
    #     instruction = "Examine the examples below and determine the category 0 or 1 for the final claim based on the pattern. After

    # Randomize order to avoid positional bias
    random.shuffle(relevant_demos)

    messages = [
        {"role": "system", "content": instruction}
    ]

    for demo in relevant_demos:
        messages.append({"role": "user", "content": demo['prompt']})
        messages.append({"role": "assistant", "content": str(demo['label'])})

    target_demo = full_demos[example_uid]
    messages.append({"role": "user", "content": target_demo['prompt']})

    if verbose:
        logger.info(f"Predicting for UID {example_uid} with {len(relevant_demos)} shots:")
        logger.info(print_messages(messages[:3]))  # Log first few for brevity
        if verbose > 1:
            logger.info(print_messages(messages))

    try:
        completion = await cached_openrouter_completion_wlogprobs(
            model_id=config.model_id,
            messages=messages,
            max_completion_tokens=60 if verbose else 4,
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
            provider_whitelist=config.provider_whitelist if config.provider_whitelist else None
        )

        # Extract logprobs for '0' and '1'
        choice_logp_dict, logp_dict = get_logprobs_choices(completion, choices=['A', 'B'], regex='[AB]')


        if verbose:
            logger.info(f"Predicted: {pred}, logprob: {lprob:.4f}")

        return pred, lprob
    except Exception as e:
        logger.error(f"Error in predict_label for UID {example_uid}: {e}")
        return random.choice([0, 1]), 0.0

# %% [code]
# Compute energy and metrics
def compute_energy(demos, config=C):
    labeled = [d for d in demos.values() if d['label'] is not None]
    if not labeled:
        return 0.0
    avg_lprob =  np.mean([d['score'] for d in labeled])
    # Count inconsistencies
    """Counts inconsistencies: same consistency_key must have same label (paraphrases agree);
    different keys in group must have opposite labels (assumes contradictions, like original TruthfulQA groups).
    Penalizes each violation. Handles multiples flexibly."""
    num_inconsistent = 0
    groups = {}
    for uid, demo in demos.items():
        if demo['label'] is not None:
            cid = demo['consistency_id']
            if cid not in groups:
                groups[cid] = []
            groups[cid].append((uid, demo['label'], demo['consistency_key']))
    
    for cid, items in groups.items():
        key_groups = {}
        for uid, label, key in items:
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(label)
        
        # Same key must agree
        for key, labels in key_groups.items():
            if len(set(labels)) > 1:
                num_inconsistent += len(labels) - 1  # Penalize differing labels in same key
        
        # Different keys must oppose (assume contradictory)
        all_labels = [label for _, label, _ in items]
        if len(set(all_labels)) < len(key_groups):  # Not all oppose if num unique labels < num keys
            num_inconsistent += max(0, len(items) - len(set(all_labels)))
    
    energy = config.alpha * avg_lprob - num_inconsistent
    accuracy = np.mean([d['label'] == d['vanilla_label'] for d in labeled])
    return energy, {
        'avg_prob': avg_lprob,
        'num_inconsistent': num_inconsistent,
        'accuracy': accuracy,
        'num_labeled': len(labeled)
    }

logger.info("Initial energy: {}", compute_energy(demonstrations))

# %% [code]
async def fix_inconsistencies_simple(demos, config=C, max_fixes=5):
    """Simple consistency fix: for inconsistent pairs, enumerate label combos, re-predict, pick max energy."""
    for fix_iter in range(max_fixes):
        # Find inconsistent pairs
        groups = {}
        for uid, demo in demos.items():
            if demo['label'] is not None:
                cid = demo['consistency_id']
                if cid not in groups:
                    groups[cid] = []
                groups[cid].append(uid)
        
        # Find first inconsistent pair
        inconsistent_pair = None
        for cid, uids in groups.items():
            labels = [demos[uid]['label'] for uid in uids if demos[uid]['label'] is not None]
            if len(set(labels)) > 1:  # Inconsistent
                inconsistent_pair = (uids[0], uids[1])
                break
        
        if inconsistent_pair is None:
            break  # No more inconsistencies or not yet enough labels to have any effect
        
        uid1, uid2 = inconsistent_pair
        
        # Enumerate all 4 label combinations and pick max energy
        # FIXME doesn't this call predict_label twice per option?
        options = [(0, 0), (0, 1), (1, 0), (1, 1)]
        best_energy = float('-inf')
        best_option = None
        
        for label1, label2 in options:
            temp_demos = demos.copy()
            temp_demos[uid1]['label'] = label1
            temp_demos[uid2]['label'] = label2
            
            # Re-predict labels with new context to update scores
            current_labeled = {k: v for k, v in temp_demos.items() if v['label'] is not None}
            new_label1, score1 = await predict_label(uid1, temp_demos, current_labeled, config)
            new_label2, score2 = await predict_label(uid2, temp_demos, current_labeled, config)
            temp_demos[uid1]['score'] = score1
            temp_demos[uid2]['score'] = score2
            
            energy, _ = compute_energy(temp_demos, config)
            
            if energy > best_energy:
                best_energy = energy
                best_option = (label1, label2, score1, score2)
        
        # Apply best option
        demos[uid1]['label'] = best_option[0]
        demos[uid1]['score'] = best_option[2]
        demos[uid2]['label'] = best_option[1]
        demos[uid2]['score'] = best_option[3]
    
    return demos

# %% [code]
# Main simulated annealing loop
async def run_icm(demonstrations, config=C):
    energies = []
    accuracies = []
    
    # Fix any initial inconsistencies from random initialization
    demonstrations = await fix_inconsistencies_simple(demonstrations, config)
    
    current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
    old_energy, old_metrics = compute_energy(demonstrations, config)
    
    # Set initial scores for seeds by predicting (using current labels)
    initial_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
    for uid in initial_labeled:
        pred_label, score = await predict_label(uid, demonstrations, initial_labeled, config, verbose=False)
        demonstrations[uid]['score'] = score  # Keep original label, update score
        initial_labeled[uid]['score'] = score
    old_energy, old_metrics = compute_energy(demonstrations, config)
    logger.info("Initial scores set. Updated energy: {}", old_energy)
    
    for iter in range(config.max_iters):
        # Weighted sampling for inconsistent groups
        groups = {}
        all_uids = list(demonstrations.keys())
        for uid in all_uids:
            cid = demonstrations[uid]['consistency_id']
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(uid)
        
        weights = [0.1 for _ in all_uids]  # Base low
        
        for cid, group_uids in groups.items():
            labeled_labels = [demonstrations[uid]['label'] for uid in group_uids if demonstrations[uid]['label'] is not None]
            num_labeled = len(labeled_labels)
            num_unlabeled = len(group_uids) - num_labeled
            
            inconsistency = 0
            if num_labeled > 0:
                unique_labels = set(labeled_labels)
                inconsistency = 1 if len(unique_labels) > 1 else 0
            
            if num_unlabeled > 0:
                weight_factor = (0.5 + 0.5 * inconsistency) * (1 + num_unlabeled / len(group_uids))
                for uid in group_uids:
                    if demonstrations[uid]['label'] is None:  # Unlabeled
                        idx = all_uids.index(uid)
                        weights[idx] = weight_factor
            else:
                # Fully labeled groups with inconsistency get higher weight for re-prediction
                if inconsistency:
                    for uid in group_uids:
                        idx = all_uids.index(uid)
                        weights[idx] = 0.5 + random.uniform(0, 0.2)  # Higher than base, lower than unlabeled; +rand for exploration/flipping
                else:
                    for uid in group_uids:
                        idx = all_uids.index(uid)
                        weights[idx] = 0.1
        # TODO: Weight by potential energy delta: for candidates, temp-assign/predict label, compute delta U vs current, prioritize high positive delta (improves over score; low complexity, med value).
        
        weights = [max(w, 0.01) for w in weights]
        example_uid = random.choices(all_uids, weights=weights)[0]
        
        # Predict new label
        if iter%100==0:
            verbose = 1
        elif iter%100==1:
            verbose = 2
        else:
            verbose = 0
        new_label, score = await predict_label(example_uid, demonstrations, current_labeled, config, verbose=verbose)

        # Update with new label and fix any inconsistencies
        temp_demos = demonstrations.copy()
        temp_demos[example_uid]['label'] = new_label
        temp_demos[example_uid]['score'] = score
        temp_demos = await fix_inconsistencies_simple(temp_demos, config)
        
        # TODO check if we should enable following
        # if new_label != demonstrations[example_uid].get('label', None):
        #     current_labeled = {k: v for k, v in temp_demos.items() if v['label'] is not None}
        
        # Compute new energy
        new_energy, new_metrics = compute_energy(temp_demos, config)
        delta = new_energy - old_energy
        
        # Annealing decision
        T = max(config.final_t, config.initial_t / (1 + config.beta * math.log(1 + iter)))
        accept_msg = f"Delta: {delta:.2f}, T: {T:.2f}, Acc: {new_metrics['accuracy']:.2f}"
        if delta > 0 or random.random() < math.exp(delta / T):
            demonstrations = temp_demos
            old_energy = new_energy
            current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
            logger.info("Iter {}: Accepted. Energy: {:.2f}. {}", iter, old_energy, accept_msg)
        else:
            logger.debug("Iter {}: Rejected. {}", iter, accept_msg)
        
        energies.append(old_energy)
        accuracies.append(new_metrics['accuracy'])
        
        if iter % C.log_interval == 0:
            logger.info(f"Progress: Labeled {new_metrics['num_labeled']}, Inconsistents: {new_metrics['num_inconsistent']}. Acc: {new_metrics['accuracy']:.2f}, Energy: {old_energy:.2f}")
    
    return demonstrations, energies, accuracies

# %% [code]
# Run the algorithm
final_demos, energies, accuracies = asyncio.run(run_icm(demonstrations, C))

# Final metrics
final_energy, final_metrics = compute_energy(final_demos, C)
logger.info("\nFinal Results:")
logger.info("Energy: {:.2f}", final_energy)
# TODO show vanilla accuracy here for comparison
logger.info("Accuracy vs vanilla: {:.2f}, initial {:.2f}", final_metrics['accuracy'], accuracies[0])
logger.info("Labeled: {}/{}", final_metrics['num_labeled'], len(data))
logger.info("Inconsistencies: {}", final_metrics['num_inconsistent'])

# Final labels
df = pd.DataFrame(final_demos).T
df.to_parquet(C.out_dir / "icm_final_labels.parquet")

logger.info("\nFinal labels:")
for uid, demo in final_demos.items():
    label = demo['label']
    if label is not None:
        logger.info(f"UID {uid} ({demo['consistency_id']}): {label} (vanilla: {demo['vanilla_label']})")


json.dump(
    asdict(C),
    open(C.out_dir / "icm_config.json", "w")
)

# TODO put them in file, only print a top few disagreements

# %% [code]
# Simple visualization (requires matplotlib)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(energies)
plt.title('Energy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Energy')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig("icm_progress.png")
plt.show()

# %% [markdown]
# ## Next Steps & Limitations
# Prioritized by complexity (low/med/high) vs. value (low/med/high) based on paper comparison. Limitations noted with potential fixes.

    # # Existing TODOs:
# - [x] Load larger HF dataset ('Yik/truthfulQA-bool' subset, formatted to messages).
# - [x] Refine few-shot prompt from original get_judge_prompt_fewshot.
# - [x] Add weighted sampling for inconsistent groups (no longer random).
# - [x] Remove biased instruction - pure pattern completion for unsupervised elicitation.
# - [x] Simplify consistency fix - enumerate label combos, re-predict, pick max energy (no LLM meta-reasoning).
# - [x] Add log temperature schedule: max(Tmin, T0 / (1 + β log(n))) - Low complexity, Med value: Better early exploration.
# - [ ] Add caching for predictions (dict/file-based) - Low complexity, Med value: @lru_cache already used, could add disk cache.
# - [ ] Implement async batch predictions in predict_label - Med complexity, Med value: Use asyncio.gather for concurrent API calls.
# - [ ] Full mutual predictability: Use all/subsampled demos in predict_label - High complexity, High value: Handle context limits.
# - [ ] Test robustness with worst-case init: Add golden/random/worst init options - Med complexity, Low value: Validate like paper Sec. 5.
# # - [ ] Dynamically select demos in few-shot (group/sort by consistency_id) - Med complexity, High value.
# # - [ ] Weight sampling by energy delta potential - Low complexity, Med value.
# #
# # Limitations (from Paper Sec. 9) & Potential Fixes:
# - [ ] Salient concepts only: ICM can't elicit non-salient private preferences (e.g., "sun" poems); fix: Combine with weak supervision.
# - [ ] Context length limits: Can't fit all N demos for large datasets; fix: Subsample relevant demos or use long-context models.
# - [ ] Degenerate solutions without consistency: Risk of all-same labels; mitigated by logical consistency term.
# - [ ] Inference cost: 2-3 fwd passes per point (paper App. B); fix: Caching + batching reduces API hits.
# - [ ] I'd like to record what it thinks the labels represent e.g. "misconception" "virtue" etc, and find which answer leads to the best energy.
