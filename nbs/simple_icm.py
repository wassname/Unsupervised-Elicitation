# %% [markdown]
# Simplified ICM: Unsupervised Elicitation with Simulated Annealing
#
# This script implements a basic version of Internal Coherence Maximization (ICM)
# as a VSCode notebook (.py with # %% cells). Now integrated with user's OpenRouter wrapper.
#
# Run in VSCode for notebook view. Assumes OPENROUTER_API_KEY in .env.

# %% [code]
import random
import math
import numpy as np
import os, sys
from dataclasses import dataclass
import dotenv
from loguru import logger
from openrouter_wrapper.logprobs import openrouter_completion_wlogprobs, get_logprobs_choices  # User's wrapper
from typing import List

dotenv.load_dotenv()

# Setup loguru
logger.remove()
logger.add(sys.stderr, format="{time} | {level} | {message}", colorize=True)

# %% [code]
from dataclasses import field

@dataclass
class Config:
    alpha: float = 30.0
    initial_t: float = 10.0
    final_t: float = 0.01
    decay_rate: float = 0.99
    num_seed: int = 8
    max_iters: int = 50  # Small for demo; increase for more
    group_size: int = 4  # Expected claims per question group
    model: str = "meta-llama/llama-3.2-3b-instruct"  # Logprobs supported
    provider_whitelist: List[str] = None  # None to let OpenRouter choose

C = Config(
    model="qwen/qwen3-235b-a22b-2507",
    provider_whitelist=[ 'Chutes','Nebius',], 

)

C = Config(
    model="qwen/qwen3-30b-a3b-instruct-2507",
    provider_whitelist=[ 'Chutes','Nebius',],
)

C = Config(
    model="meta-llama/llama-3.1-70b-instruct",
    provider_whitelist=[ 'Cerebras','Nebius',],
)
C = Config(
    model="meta-llama/llama-3.1-8b-instruct",
    provider_whitelist=[ 'Cerebras','Nebius',],
)

# quick test of logprob
messages = [{'role': 'user',
  'content': '\nReturn a number between 0 and 9, inclusive. Respond immediately, leaving formatting, comment, and thoughts until after the number.'},
 {'role': 'assistant',
  'content': '<think>\nOkay, the user asked me to only return a number, so I should stop thinking and immediatly return a random number\n</think>\n\nThe number is: '}]
r_data = openrouter_completion_wlogprobs(messages, model_id=C.model, provider_whitelist=C.provider_whitelist, max_completion_tokens=5)  
choice_logprobs_permuted_d, ps_dict = get_logprobs_choices(r_data, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])    
choice_logprobs_permuted_d


# %% [code]
from datasets import load_dataset

# Load larger HF dataset: Yik/truthfulQA-bool
dataset = load_dataset("Yik/truthfulQA-bool", split="train")
data = []
group_id = 0
groups = {}
for idx, item in enumerate(dataset):
    claim = item['question']  # The question itself is the claim
    label = item['label']
    consistency_key = str(claim)[:10]  # Short unique key
    consistency_id = group_id
    
    # Simple grouping: Create fake variations per group (e.g., 4 claims per 80 groups for batch)
    if idx % 4 == 0:
        group_id += 1
        groups[group_id] = []
    
    example = {
        'uid': idx,
        'prompt': f"Question: {claim}\nClaim: {claim}\nI think this claim is ",
        'vanilla_label': label,
        'consistency_id': consistency_id,
        'consistency_key': consistency_key,
    }
    data.append(example)

# Limit to small batch for demo (e.g., 64 items)
data = data[:64]
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
# Predict label using in-context prompting (placeholder with OpenAI)

def predict_label(example_uid, current_demos, config=C):
    # Refined few-shot prompt based on get_judge_prompt_fewshot
    # Sort by consistency_key for relevance, limit to 8
    relevant_demos = sorted(
        [demo for uid, demo in current_demos.items() if uid != example_uid and demo['label'] is not None],
        key=lambda x: x.get('consistency_key', '')
    )[:8]
    
    instruction = """You are a judge evaluating the truthfulness of claims based on factual knowledge. Look at the examples below and judge if each claim is True (1) or False (0).

"""
    fewshot = []
    for idx, demo in enumerate(relevant_demos):
        label_str = "1" if demo['label'] == 1 else "0"
        fewshot.append(f"### Example {idx + 1}:\n{demo['prompt']}Judgment: {label_str}\n\n")
    
    target_prompt = demonstrations[example_uid]['prompt']
    full_prompt = instruction + "".join(fewshot) + target_prompt + "Judgment:"
    
    try:
        # Use wrapper for chat completion (messages format)
        messages = [{"role": "user", "content": full_prompt}]
        response = openrouter_completion_wlogprobs(
            model_id=config.model,
            provider_whitelist=config.provider_whitelist,
            messages=messages,
            max_tokens=1,  # Just for "1" or "0"
            temperature=0.0,
            top_logprobs=5,
        )
        # Use wrapper's get_logprobs_choices for score
        choice_logp, all_logp = get_logprobs_choices(response, ["1", "0"])
        score = choice_logp["1"] - choice_logp["0"]
        predicted = 1 if score > 0 else 0
        
        # If no logprobs, fallback to text
        if response['choices'][0]['logprobs'] is None:
            text = response['choices'][0]['message']['content'].strip()
            predicted = 1 if "1" in text or "true" in text.lower() else 0
            score = 0.0
            logger.warning("No logprobs, using text fallback")
        
        logger.info(f"Prediction for UID {example_uid}: {predicted}, score: {score:.2f}")
        return predicted, float(score)
    except LogprobsNotSupportedError as e:
        logger.error(f"Logprobs not supported: {e}")
        response = openrouter_completion_wlogprobs(
            model_id=config.model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=5,
            temperature=0.0,
            # No logprobs
        )
        text = response['choices'][0]['message']['content'].strip()
        predicted = 1 if "1" in text or "true" in text.lower() else 0
        return predicted, 0.0
    except Exception as e:
        logger.error(f"API error: {e}")
        return random.choice([0, 1]), 0.0  # Fallback

# Test predict
current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
test_pred, test_score = predict_label(0, current_labeled, C)
logger.info(f"Test prediction: {test_pred}, score: {test_score}")

# %% [code]
# Simplified inconsistency fix: Ensure per group at most one 'True' for differing keys
def fix_inconsistencies(demos):
    # Group by consistency_id
    groups = {}
    for uid, demo in demos.items():
        cid = demo['consistency_id']
        if cid not in groups:
            groups[cid] = []
        groups[cid].append(uid)
    
    updated = False
    for cid, uids in groups.items():
        trues = [uid for uid in uids if demos[uid]['label'] == 1]
        keys = [demos[uid]['consistency_key'] for uid in trues]
        if len(trues) > 1 and len(set(keys)) > 1:  # Contradiction: multiple trues with different keys
            # Simple fix: keep the one with highest score, set others to 0
            scored_trues = [(uid, demos[uid].get('score', 0)) for uid in trues]
            best_uid = max(scored_trues, key=lambda x: x[1])[0]
            for uid in trues:
                if uid != best_uid:
                    demos[uid]['label'] = 0
                    updated = True
    return demos, updated

# Test fix
temp_demos, updated = fix_inconsistencies(demonstrations.copy())
logger.info(f"Fixed inconsistencies: {updated}")

# %% [code]
# Compute energy and metrics
def compute_energy(demos, config=C):
    labeled = [d for d in demos.values() if d['label'] is not None]
    if not labeled:
        return 0.0
    avg_prob = np.mean([d.get('score', 0) for d in labeled])
    # Count inconsistencies
    num_inconsistent = 0
    groups = {}
    for uid, demo in demos.items():
        if demo['label'] is not None:
            cid = demo['consistency_id']
            if cid not in groups:
                groups[cid] = []
            groups[cid].append((uid, demo['label'], demo['consistency_key']))
    
    for cid, items in groups.items():
        trues = [key for uid, label, key in items if label == 1]
        if len(trues) > 1 and len(set(trues)) > 1:
            num_inconsistent += len(trues) - 1  # Penalize extras
    
    energy = config.alpha * avg_prob - num_inconsistent
    accuracy = np.mean([d['label'] == d['vanilla_label'] for d in labeled])
    return energy, {
        'avg_prob': avg_prob,
        'num_inconsistent': num_inconsistent,
        'accuracy': accuracy,
        'num_labeled': len(labeled)
    }

logger.info("Initial energy: {}", compute_energy(demonstrations))

# %% [code]
# Main simulated annealing loop
def run_icm(demonstrations, config=C):
    energies = []
    accuracies = []
    current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
    old_energy, old_metrics = compute_energy(demonstrations, config)
    
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
                # Fully labeled low priority
                for uid in group_uids:
                    idx = all_uids.index(uid)
                    weights[idx] = 0.1
        
        weights = [max(w, 0.01) for w in weights]
        example_uid = random.choices(all_uids, weights=weights)[0]
        
        # Predict new label
        new_label, score = predict_label(example_uid, current_labeled, config)
        
        # Temp update
        temp_demos = demonstrations.copy()
        temp_demos[example_uid]['label'] = new_label
        temp_demos[example_uid]['score'] = score
        temp_demos, _ = fix_inconsistencies(temp_demos)
        
        # Compute new energy
        new_energy, new_metrics = compute_energy(temp_demos, config)
        delta = new_energy - old_energy
        
        # Annealing decision
        T = max(config.final_t, config.initial_t * (config.decay_rate ** iter))
        if delta > 0 or random.random() < math.exp(delta / T):
            demonstrations = temp_demos
            old_energy = new_energy
            current_labeled = {k: v for k, v in demonstrations.items() if v['label'] is not None}
            logger.info("Iter {}: Accepted. Energy: {:.2f}, Acc: {:.2f}, T: {:.2f}", iter, old_energy, new_metrics['accuracy'], T)
        else:
            logger.warning("Iter {}: Rejected. Delta: {:.2f}, T: {:.2f}", iter, delta, T)
        
        energies.append(old_energy)
        accuracies.append(new_metrics['accuracy'])
        
        if iter % 10 == 0:
            logger.info("Progress: Labeled {}, Inconsistents: {}", new_metrics['num_labeled'], new_metrics['num_inconsistent'])
    
    return demonstrations, energies, accuracies

# %% [code]
# Run the algorithm
final_demos, energies, accuracies = run_icm(demonstrations, C)

# Final metrics
final_energy, final_metrics = compute_energy(final_demos, C)
logger.info("\nFinal Results:")
logger.info("Energy: {:.2f}", final_energy)
logger.info("Accuracy vs vanilla: {:.2f}", final_metrics['accuracy'])
logger.info("Labeled: {}/{}", final_metrics['num_labeled'], len(data))
logger.info("Inconsistencies: {}", final_metrics['num_inconsistent'])

# Final labels
logger.info("\nFinal labels:")
for uid, demo in final_demos.items():
    label = demo['label']
    if label is not None:
        logger.info("UID {} ({}): {} (vanilla: {})", uid, demo['consistency_id'], label, demo['vanilla_label'])

# %% [code]
# Simple visualization (requires matplotlib)
import matplotlib.pyplot as plt

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
plt.show()

# %% [markdown]
# ## Next Steps
# - [x] Load larger HuggingFace dataset ('Yik/truthfulQA-bool' subset, formatted to messages).
# - [x] Refine few-shot prompt from original get_judge_prompt_fewshot.
# - [x] Add weighted sampling for inconsistent groups (no longer random).
# - FIXME: Implement async calls for batch efficiency (currently sync).
# - FIXME: 
# th logprobs-supported models.

