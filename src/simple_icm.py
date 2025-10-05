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
import os, sys
import pandas as pd
from dataclasses import dataclass, asdict
import dotenv
from loguru import logger
from openrouter_wrapper.logprobs import openrouter_completion_wlogprobs, get_logprobs_choices, LogprobsNotSupportedError  # User's wrapper
from typing import List, Tuple, Callable, Literal
import asyncio
from aiocache import cached
from itertools import combinations
from copy import deepcopy
import signal

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
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm}</green> | <level>{level}</level> | <cyan>{message}</cyan>", colorize=True, level="DEBUG")

# Global cost tracker
total_cost = 0.0
reasoning_log = ""
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown requested (Ctrl+C), will finish current iteration and save results...")

signal.signal(signal.SIGINT, signal_handler)

# %% [code]

@dataclass
class Config:
    alpha: float = 30.0 # Weight for logprob in energy vs consistency
    initial_t: float = 10.0 # Initial temperature for annealing
    final_t: float = 0.01  # Final temperature for annealing
    beta: float = 2.0 # controls cooling schedule
    num_seed: int = 42

    semantic_anchor: str = "" # if we want to nudge the model towards a labelling dimension we can give it a clue

    max_iters: int = 2500  # should be at least dataset size X 2
    log_interval: int = 100  # Log progress every N iterations

    n_shots: int = 6  # Number of in-context examples
    batch_size: int = 5  # Parallel predictions per iteration
    dataset: Literal["truthfulqa", "daily_dilemmas"] = "truthfulqa"  # Dataset name for logging
    # model_id: str = "meta-llama/llama-3.1-8b-instruct"  # Logprobs supported
    model_id: str = "qwen/qwen3-235b-a22b-2507"  # Need a openrouter model with Logprobs supported. "meta-llama/llama-3.1-8b-instruct, qwen/qwen3-235b-a22b-2507", "qwen/qwen3-30b-a3b-instruct-2507"
    provider_whitelist: Tuple[str] = ('Chutes','Nebius',)  # None to let OpenRouter choose

    out_dir: Path = Path("./outputs/icm")  # Directory to save outputs


import simple_parsing

C: Config = simple_parsing.parse(Config)

# C = Config(
#     model_id="qwen/qwen3-235b-a22b-2507", # $0.2 0.6
#     provider_whitelist=[ 'Chutes','Nebius',], 
# )

# C = Config(
#     model_id="qwen/qwen3-30b-a3b-instruct-2507", # 0.08 $0.33
#     provider_whitelist=[ 'Chutes','Nebius',],
# )

# C = Config(
#     model_id="meta-llama/llama-3.1-8b-instruct", # 0.02 0.03
#     provider_whitelist=('Cerebras','Nebius',),
# )

logger.info(f"Config: {C}")
config_dict = asdict(C)
out_dir = C.out_dir / C.dataset.replace(' ', '_').replace('/', '_')
out_dir.mkdir(parents=True, exist_ok=True)
config_dict['out_dir'] = str(C.out_dir)
with open(out_dir / "icm_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# %% [code]
from src.data.truthfulqa import load_truthfulqa, is_consistent
from data.daily_dilemmas import load_daily_dilemmas

if C.dataset == "truthfulqa":
    data = load_truthfulqa()
elif C.dataset == "daily_dilemmas":
    data = load_daily_dilemmas()
else:
    raise ValueError(f"Unknown dataset {C.dataset}")

logger.info("Loaded {} examples", len(data))

# %% [code]
# Initialize: Random labels for first num_seed, None for others
def initialize_data(data, config):
    demonstrations = {item['uid']: deepcopy(item) for item in data}
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

@cached()
async def cached_openrouter_completion_wlogprobs(*args, **kwargs):
    return await openrouter_completion_wlogprobs(*args, **kwargs)


async def predict_label(example_uid, current_demos, config=C, verbose=False, all_demos=None):
    # simplified from https://github.com/Jiaxin-Wen/Unsupervised-Elicitation/blob/master/src/experiments/ICM.py#L254
    # Sort by consistency_key for relevance, limit to 16
    global total_cost
    # Group and interleave demos by consistency_id like original
    grouped_demos = {}
    for uid, demo in current_demos.items():
        if uid != example_uid and demo['label'] is not None:
            grouped_demos.setdefault(demo['consistency_id'], []).append(demo)

    relevant_demos = []
    for group in grouped_demos.values():
        random.shuffle(group)  # Reduce order bias
        relevant_demos.extend(group[:2])  # Limit per group for balance

    relevant_demos = relevant_demos[:C.n_shots]  # Cap total

    instruction = "Examine these examples of paired (Candidate, Set) and complete the pattern for the new one.\n\n"
    if C.semantic_anchor:
        instruction += f"Hint: The Sets relate to the dimension: {C.semantic_anchor}\n\n"
    fewshot = []
    for demo in relevant_demos:
        label_str = "A" if demo['label'] == 1 else "B"
        fewshot.append(f"\n\n## Candidate:\n{demo['prompt']}\n## Set:\n{label_str}")

    # Use all_demos if provided (for unlabeled examples), otherwise use current_demos
    demos_source = all_demos if all_demos is not None else current_demos
    target_prompt = demos_source[example_uid]['prompt']
    messages = [
        {"role": "user", "content": instruction+"".join(fewshot)+f"\n\n## Candidate:\n{target_prompt}"},
        {"role": "assistant", "content": "\n## Set:"} # Assistant prefill to ensure 
    ]

    if verbose>1:
        messages[0]['content'] = "ALWAYS GIVE BRIEF REASONING AFTERWARDS. " + messages[0]['content']
    

    response = await cached_openrouter_completion_wlogprobs(
        model_id=config.model_id,
        provider_whitelist=config.provider_whitelist,
        messages=messages,
        max_completion_tokens=160 if verbose else 5,
        min_completion_tokens=30 if verbose else 1,
        temperature=0.4,
        top_logprobs=8,
    )
    
    total_cost += response.get('usage', {}).get('cost', 0.0)
    
    if verbose:
        logger.info(f"Debug Prediction - UID {example_uid}:")
        logger.info(f"messages: `{print_messages(messages)}`")
        logger.info(f"Response Content: `{response['choices'][0]['message']['content']}`")
        logger.info(f"--- End Debug ---")
        if verbose>1:
            global reasoning_log
            # reasoning_log += f"\n\n## Candidate:\n{target_prompt}\n## Set:\n"
            #@ TODO record iter
            labeled = [v for v in current_demos.values() if v['label'] is not None]
            reasoning_log += f"""
Reasoning for UID {example_uid}, labelled {len(labeled)}:
{response['choices'][0]['message']['content']}\n\n
"""

    try:
        choice_strs = ["A", "B"]
        choice_logp, top_logp = get_logprobs_choices(response, choice_strs, lower=False)

        choice_in_toplogp = any([s for s in choice_strs if s in top_logp])

        if not choice_in_toplogp:
            model_response = response['choices'][0]['message']['content']
            logger.warning(f"Choices not returned for UID {example_uid}, may indicate model confusion. choice_logp={choice_logp}. Instead we got these top logprobs: {top_logp} and \nmessages: ...`{print_messages(messages)[-90:]}`\nthis output:`{model_response}`")
        score = choice_logp["A"] - choice_logp["B"]
        predicted = 1 if score > 0 else 0
        return predicted, float(score)
    except Exception as e:
        raise e
        logger.exception(f"API error: {e}")
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
    
    energy = config.alpha * avg_lprob - num_inconsistent - (num_inconsistent / max(1, len(labeled)))  # Normalized penalty
    accuracy = np.mean([d['label'] == d['vanilla_label'] for d in labeled])
    # flip acc if needed, as this is unsupervised
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    return energy, {
        'avg_lprob': avg_lprob,
        'num_inconsistent': num_inconsistent,
        'accuracy': accuracy,
        'num_labeled': len(labeled)
    }

logger.info("Initial energy: {}", compute_energy(demonstrations))

# %% [code]

def get_kflip_neighbors(group_uids, demos, k):
    """
    Generate all label assignments that are k flips away from current.
    Returns list of [(uid, new_label), ...] tuples.
    """
    labeled_uids = [uid for uid in group_uids if demos[uid]['label'] is not None]
    neighbors = []
    
    for combo in combinations(labeled_uids, k):
        flips = [(uid, 1 - demos[uid]['label']) for uid in combo]
        neighbors.append(flips)
    
    return neighbors

async def fix_inconsistencies_greedy(demos, config=C, max_fixes=20, max_flips=3, is_consistent: Callable = is_consistent):
    """
    Greedy consistency fix: Try k-flip neighborhoods (k=1,2,...) until we find
    a consistent assignment that improves energy.
    """
    for fix_iter in range(max_fixes):
        # Find inconsistent groups
        groups = {}
        for uid, demo in demos.items():
            if demo['label'] is not None:
                groups.setdefault(demo['consistency_id'], []).append(uid)
        
        # Find first inconsistent group
        inconsistent_group = None
        for cid, uids in groups.items():
            if not is_consistent(uids, demos):
                inconsistent_group = uids
                break
        
        if not inconsistent_group:
            break  # All consistent!
        
        old_energy, _ = compute_energy(demos, config)
        best_energy = old_energy
        best_flips = None
        
        # Try k=1, 2, 3, ... flips until we find improvement
        for k in range(1, min(max_flips + 1, len(inconsistent_group) + 1)):
            neighbors = get_kflip_neighbors(inconsistent_group, demos, k)
            
            for flips in neighbors:
                # Apply flips temporarily
                temp_demos = deepcopy(demos)
                for uid, new_label in flips:
                    temp_demos[uid]['label'] = new_label
                
                # Check if this is consistent
                if not is_consistent(inconsistent_group, temp_demos):
                    continue  # Skip inconsistent neighbors
                
                # Score it
                energy, _ = compute_energy(temp_demos, config)
                
                if energy > best_energy:
                    best_energy = energy
                    best_flips = flips
            
            if best_flips:
                break  # Found improvement with k flips, don't try k+1
        
        # Apply best flips if improvement found
        if best_flips and best_energy > old_energy:
            for uid, new_label in best_flips:
                demos[uid]['label'] = new_label
        else:
            break  # No improvement possible, stop trying
    
    return demos

# %% [code]
# Main simulated annealing loop
async def run_icm(demonstrations, config=C):
    energies = []
    accuracies = []
    
    # Fix any initial inconsistencies from random initialization
    demonstrations = await fix_inconsistencies_greedy(demonstrations, config)
    
    current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
    old_energy, old_metrics = compute_energy(demonstrations, config)
    
    try:
        for iter in range(config.max_iters):
            if shutdown_requested:
                logger.info("Graceful shutdown at iteration {}", iter)
                break
            
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
            
            weights = [max(w, 0.01) for w in weights]
            
            # Batch predictions: sample multiple candidates, predict in parallel, pick best by energy delta
            candidate_uids = random.choices(all_uids, weights=weights, k=min(config.batch_size, len(all_uids)))
            candidate_uids = list(set(candidate_uids))  # Dedupe
            
            # Predict in parallel
            verbose = 1 if iter % 100 == 0 else (2 if iter % 100 == 1 else 0)
            tasks = [predict_label(uid, current_labeled, config, verbose=verbose, all_demos=demonstrations) 
                     for uid in candidate_uids]
            results = await asyncio.gather(*tasks)
            
            # Evaluate each candidate's energy delta
            best_uid = None
            best_delta = float('-inf')
            best_temp_demos = None
            
            for uid, (new_label, score) in zip(candidate_uids, results):
                temp_demos = deepcopy(demonstrations)
                temp_demos[uid]['label'] = new_label
                temp_demos[uid]['score'] = score
                
                # Fix inconsistencies if label changed
                if demonstrations[uid]['label'] != new_label:
                    temp_demos = await fix_inconsistencies_greedy(temp_demos, config)
                
                new_energy, _ = compute_energy(temp_demos, config)
                delta = new_energy - old_energy
                
                if delta > best_delta:
                    best_delta = delta
                    best_uid = uid
                    best_temp_demos = temp_demos
            
            # Apply best candidate
            if best_temp_demos is None:
                continue  # Skip if no valid candidates
            
            new_energy, new_metrics = compute_energy(best_temp_demos, config)
            delta = best_delta
            
            # Annealing decision
            T = max(config.final_t, config.initial_t / (1 + config.beta * math.log(1 + iter)))
            accept_msg = f"Delta: {delta:.2f}, T: {T:.2f}, Acc: {new_metrics['accuracy']:.2f}"
            if delta > 0 or random.random() < math.exp(delta / T):
                demonstrations = best_temp_demos
                old_energy = new_energy
                current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
                logger.debug("Iter {}: Accepted UID {}. Energy: {:.2f}. {}", iter, best_uid, old_energy, accept_msg)
            else:
                logger.debug("Iter {}: Rejected. {}", iter, accept_msg)
            
            energies.append(old_energy)
            accuracies.append(new_metrics['accuracy'])
            
            if iter % C.log_interval == 0:
                logger.info(f"Progress: Labeled {new_metrics['num_labeled']}, Inconsistents: {new_metrics['num_inconsistent']}. Acc: {new_metrics['accuracy']:.2f}, Energy: {old_energy:.2f}. Cost so far: ${total_cost:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Stopping early.")
    except asyncio.CancelledError:
        logger.info("Asyncio task cancelled.")
    
    return demonstrations, energies, accuracies

# %% [code]
# Run the algorithm
try:
    final_demos, energies, accuracies = asyncio.run(run_icm(demonstrations, C))
except Exception as e:
    logger.exception(f"Error during ICM run: {e}")
    final_demos = demonstrations
    energies = []
    accuracies = [np.nan]

# Final metrics
final_energy, final_metrics = compute_energy(final_demos, C)
logger.info("\nFinal Results:")
logger.info("Total cost: ${:.4f}", total_cost)
logger.info("Energy: {:.2f}", final_energy)
# TODO show vanilla accuracy here for comparison
logger.info("Accuracy [labelled] vs vanilla: {:.2f}, initial {:.2f}", final_metrics['accuracy'], accuracies[0])
logger.info("Labeled: {}/{}", final_metrics['num_labeled'], len(data))
logger.info("Inconsistencies: {}", final_metrics['num_inconsistent'])

# Final labels
df = pd.DataFrame(final_demos).T
df.to_parquet(out_dir / "icm_final_labels.parquet")

df_labeled = df.dropna(subset='label').sort_values(by='score', key=np.abs, ascending=False)
df_labeled_disagreed = df_labeled[df_labeled['vanilla_label'] != df_labeled['label']]

print(f"\nFinal labeled examples (total {len(df_labeled)}):")
print(df_labeled_disagreed[['consistency_id', 'label', 'vanilla_label', 'score', 'prompt']])

for uid, row in df_labeled_disagreed.iterrows():
    print(f"\n## Candidate: {row['prompt']}\nICM Set: {'A' if row['label']==1 else 'B'}, Vanilla Set: {'A' if row['vanilla_label']==1 else 'B'}, score={row['score']}\n")

print(f"\nFinal labeled examples saved to {out_dir / 'icm_final_labels.parquet'}")

# %% [code]
# Simple visualization (requires matplotlib)

with open(out_dir / "reasoning.txt", "w") as f:
    f.write(f"\n\nTotal cost: ${total_cost:.4f}\n")
    f.write(reasoning_log)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(energies)
plt.title('Energy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Energy')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Accuracy [labelled] over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig(out_dir / "icm_progress.png")
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
# - [ ] Implement async batch predictions in predict_label - Med complexity, Med value: Use asyncio.gather for concurrent API calls.
# - [ ] Full mutual predictability: Use all/subsampled demos in predict_label - High complexity, High value: Handle context limits.
# # - [ ] Dynamically select demos in few-shot (group/sort by consistency_id) - Med complexity, High value.
# # - [ ] Weight sampling by energy delta potential - Low complexity, Med value.
# #
# # Limitations (from Paper Sec. 9) & Potential Fixes:
# - [ ] I'd like to record what it thinks the labels represent e.g. "misconception" "virtue" etc, and find which answer leads to the best energy.
# - [ ] consistency groups rely on known labels, this is a limitation for real unsupervised use
