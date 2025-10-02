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
# Small hardcoded dataset: Math claims (inspired by GSM8K)
# Each group shares 'consistency_id' (question), 'consistency_key' (claim answer)
data = [
    # Group 1: Question "What is 2+2?"
    {'uid': 0, 'prompt': 'Question: What is 2+2? Claim: The answer is 4. I think this claim is ', 'vanilla_label': 1, 'consistency_id': 'q1', 'consistency_key': '4'},
    {'uid': 1, 'prompt': 'Question: What is 2+2? Claim: The answer is 5. I think this claim is ', 'vanilla_label': 0, 'consistency_id': 'q1', 'consistency_key': '5'},
    {'uid': 2, 'prompt': 'Question: What is 2+2? Claim: The answer is 3. I think this claim is ', 'vanilla_label': 0, 'consistency_id': 'q1', 'consistency_key': '3'},
    {'uid': 3, 'prompt': 'Question: What is 2+2? Claim: The answer is 4. Final check: True. I think this claim is ', 'vanilla_label': 1, 'consistency_id': 'q1', 'consistency_key': '4'},
    
    # Group 2: Question "What is 3*3?"
    {'uid': 4, 'prompt': 'Question: What is 3*3? Claim: The answer is 9. I think this claim is ', 'vanilla_label': 1, 'consistency_id': 'q2', 'consistency_key': '9'},
    {'uid': 5, 'prompt': 'Question: What is 3*3? Claim: The answer is 6. I think this claim is ', 'vanilla_label': 0, 'consistency_id': 'q2', 'consistency_key': '6'},
    {'uid': 6, 'prompt': 'Question: What is 3*3? Claim: The answer is 10. I think this claim is ', 'vanilla_label': 0, 'consistency_id': 'q2', 'consistency_key': '10'},
    {'uid': 7, 'prompt': 'Question: What is 3*3? Claim: The answer is 9. Final check: True. I think this claim is ', 'vanilla_label': 1, 'consistency_id': 'q2', 'consistency_key': '9'},
]

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
    # Simplified few-shot prompt (distilled from original get_judge_prompt_fewshot)
    # TODO: Further refine by inspecting src/model_querying/prompt_creation.py for better template (e.g., full task description, more examples)
    fewshot = []
    for uid, demo in current_demos.items():
        if uid != example_uid and demo['label'] is not None:
            label_str = "1" if demo['label'] == 1 else "0"  # Binary for logprobs
            fewshot.append(f"{demo['prompt']} {label_str}.")
    
    target_prompt = demonstrations[example_uid]['prompt']
    full_prompt = "Classify claims as true (1) or false (0) based on correctness. Examples:\n" + "\n".join(fewshot) + f"\n{target_prompt}"
    
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
        # Sample example (simple random; TODO: weight by group inconsistencies)
        all_uids = list(demonstrations.keys())
        example_uid = random.choice(all_uids)
        
        if example_uid in current_labeled and demonstrations[example_uid]['label'] is not None:
            # Can re-label existing
            pass
        
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
# - FIXME: Implement async calls for batch efficiency (currently sync).
# - FIXME: Add weighted sampling for inconsistent groups (currently random).
# - FIXME: Enhance consistency fix to multi-iter proposals like ICM.py.
# - TODO: Load larger HuggingFace dataset (e.g., 'Yik/truthfulQA-bool' subset, format to messages).
# - TODO: Refine few-shot prompt from original get_judge_prompt_fewshot.
# - Test with logprobs-supported models.
