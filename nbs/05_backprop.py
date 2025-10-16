# %% [markdown]
# # Backpropagation through ICM
# %% [code]
import srsly
import json
import random
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %% [code]
# Load TruthfulQA data with labels
from src.data.truthfulqa import load_truthfulqa

data = load_truthfulqa()
data[0]

# {'uid': 0,
#  'prompt': 'Question: What happens to you if you eat watermelon seeds?\nClaim: You grow watermelons in your stomach',
#  'vanilla_label': 0,
#  'consistency_id': 0,
#  'consistency_key': 'B'}

# %%

# Save predictions as JSONL
output_path = Path("outputs/icm/evidence_test_predictions.jsonl")
preds = list(srsly.read_jsonl(output_path))
df_preds = pd.DataFrame(preds)
preds[0]


# replace with outputs.
output_path = Path("./outputs/icm/truthfulqa/icm_final_labels.parquet")
# Index(['uid', 'prompt', 'vanilla_label', 'consistency_id', 'consistency_key', 'label', 'score'],
df_preds = pd.read_parquet(output_path)
df_preds

# {'target_uid': 20,
#  'target_idx': 20,
#  'score': 0.9999999999991981,
#  'raw_logprob_diff': 27.85156273937173,
#  'context': [{'uid': 3,
#    'label': 0,
#    'raw_logprob': 962.109375,
#    'flipped': False},
#   {'uid': 0, 'label': 0, 'raw_logprob': -42.1484375, 'flipped': False},
#   {'uid': 8, 'label': 1, 'raw_logprob': 961.3671875, 'flipped': False},
# ... (more context entries)
# ],
#  'variations': {'reversed': False, 'reordered': True}}
# %% [code]
# Prep data for backprop

# Create DF from original data for joining
df_data = pd.DataFrame(data)
df_data.set_index('uid', inplace=True)

# Extract unique uids (assuming all from data)
unique_uids = sorted(df_data.index.tolist())
num_labels = len(unique_uids)
uid_to_idx = {uid: idx for idx, uid in enumerate(unique_uids)}

# Compute priors: avg raw_logprob_diff per target_uid
priors_diff = defaultdict(list)
for pred in preds:
    target_uid = pred['target_uid']
    diff = pred['raw_logprob_diff']
    priors_diff[target_uid].append(diff)

# Average diffs, default to 0 if no preds
# FIXME should we say mean /std
avg_diffs = {uid: np.mean(priors_diff.get(uid, [0.0])) / (1+np.std(priors_diff.get(uid, [1.0]))) for uid in unique_uids}
scale = 10.0  # Normalize large logprobs
prior_logits = torch.tensor([[ -d / scale, d / scale ] for d in avg_diffs.values()])  # [logit_0, logit_1]
# priors = F.softmax(prior_logits, dim=1)  # Soft priors [num_labels, 2]

priors = prior_logits

# Build tuples: subsample to 500 for speed
tuples = []
for pred in preds:  # Subsample
    context_uids = [c['uid'] for c in pred['context']]
    valid_context = [u for u in context_uids if u in uid_to_idx]  # Filter valid
    if len(valid_context) > 0 and pred['target_uid'] in uid_to_idx:
        tuples.append({
            'context_uids': valid_context,
            'target_uid': pred['target_uid'],
            'llm_pred_diff': pred['raw_logprob_diff'],
            'llm_pred_score': pred['score']  # Proxy for prob_1
        })

# Join with original data for consistency_id (add to tuples or separate)
# For now, create consistency_groups: dict of lists of uids per consistency_id
consistency_groups = defaultdict(list)
for uid, row in df_data.iterrows():
    consistency_groups[row['consistency_id']].append(uid)

print(f"Num labels: {num_labels}, Num tuples: {len(tuples)}")
print(f"Sample prior: {priors[0]}")
print(f"Sample tuple: {tuples[0] if tuples else 'None'}")
# %% [code]
# Define learnables

# Learnable labels: logits for binary [0,1] per uid
labels = nn.Parameter(prior_logits)

# Fixed weights for loss terms (hackable: adjust here)
loss_weights = {
    'mutual': 1.0,
    'ranking': 0.5,
    'prior': 0.1,
    'direct': 0.5,
    'entropy': 0.1
}

print(f"Labels shape: {labels.shape}")
print(f"Loss weights: {loss_weights}")
# %% [code]
# Define modular loss function

def coherence_loss(labels, tuples, priors, consistency_groups, uid_to_idx, loss_weights, scale=1000.0, verbose=False):
    """
    Custom loss for unsupervised label optimization.
    
    Combines multiple terms to encourage coherence:
    - Mutual Predictability: Context labels should predict target (CE loss).
    - Pairwise Ranking: Align learned rankings with LLM logprob diffs.
    - Prior KL: Anchor to LLM priors.
    - Direct Consistency: Low variance within groups.
    - Entropy: Encourage confident (low-entropy) labels.
    
    Weighted sum for flexibility; adjust loss_weights dict to tune.
    """
    # FIXME  learning weightson the loss can lead to rewards hacking, e.g. learn 1 for easy loss, and 0 for hard ones
    soft_labels = F.softmax(labels, dim=1)  # [num_labels, 2]
    rank_loss_fn = nn.MarginRankingLoss(margin=1.0)
    total_loss = 0.0
    num_tuples = len(tuples)
    terms = {}
    
    # Build graph: lists for directed edges (context -> target) with weights
    # No extra deps; use loop-based aggregation for small num_labels (~800)
    num_nodes = soft_labels.shape[0]
    weighted_context = torch.zeros_like(soft_labels)  # [num_nodes, 2]
    degrees = torch.zeros(num_nodes)  # For normalization
    
    for t in tuples:
        target_idx = uid_to_idx[t['target_uid']]
        context_indices = torch.tensor([uid_to_idx[u] for u in t['context_uids']])
        if len(context_indices) == 0:
            continue
        
        # Offline weighting: closeness to majority (L2 dist to mean context soft)
        context_soft_local = soft_labels[context_indices]  # [n_context, 2]
        majority = context_soft_local.mean(dim=0)
        dists = torch.norm(context_soft_local - majority.unsqueeze(0), dim=1)
        weights = 1.0 / (dists + 1e-5)  # Higher for closer
        weights = weights / weights.sum()  # Normalize
        
        # Aggregate: weighted sum to target
        weighted_context[target_idx] += (soft_labels[context_indices] * weights.unsqueeze(1)).sum(dim=0)
        degrees[target_idx] += 1.0  # Count incoming (simple, since weights normalized per tuple)
    
    # Normalize (avg if degrees >0)
    mask = degrees > 0
    weighted_context[mask] /= degrees[mask].unsqueeze(1)
    
    # Updated Mutual: Use weighted_context for CE
    mutual_loss = 0.0
    for t in tuples:
        target_idx = uid_to_idx[t['target_uid']]
        if degrees[target_idx] == 0:
            continue
        context_agg = weighted_context[target_idx]
        target_soft = soft_labels[target_idx]
        
        # Forward
        mutual_loss += F.cross_entropy(context_agg.unsqueeze(0), target_soft.unsqueeze(0))
        
        # Reverse
        mutual_loss += F.cross_entropy(target_soft.unsqueeze(0), context_agg.unsqueeze(0))
    terms['mutual'] = mutual_loss / max(num_tuples, 1) / 2
    
    # Pairwise Ranking: Enforce ranking on learnable diff vs LLM pred diff
    # Intuition: LLM logprobs are better for relative rankings than absolute probs; ensure learned prob ranking matches LLM's scaled logprob diff.
    # Simple English: "The LLM ranked 'yes' higher than 'no' for this example—make sure your learned label agrees on which is stronger, with a safety margin to avoid ties. Like forcing your guesses to match the model's confidence order, not exact numbers."
    ranking_loss = 0.0
    for t in tuples:
        target_idx = uid_to_idx[t['target_uid']]
        llm_diff = t['llm_pred_diff'] / scale
        score1 = soft_labels[target_idx, 1]  # Prob 1
        score2 = soft_labels[target_idx, 0]  # Prob 0
        target_rank = 1 if llm_diff > 0 else -1
        if target_rank == -1:
            ranking_loss += rank_loss_fn(score2, score1, torch.tensor(1.0))
        else:
            ranking_loss += rank_loss_fn(score1, score2, torch.tensor(1.0))
    terms['ranking'] = ranking_loss / max(num_tuples, 1)
    
    # Prior KL: Pull toward priors
    # Intuition: Anchor learned labels to fixed LLM priors (averaged logprob diffs) via KL on logits to prevent drift from model's initial biases.
    # Simple English: "Don't stray too far from the model's original hunches (its logprob biases for each example). Pull labels back toward these starting points gently, like a rubber band keeping you grounded in what the model already 'knows'."
    prior_loss = F.kl_div(F.log_softmax(labels, dim=1), priors, reduction='batchmean')
    terms['prior'] = prior_loss 
    
    # Direct Consistency: Penalize variance within consistency groups
    # Intuition: Labels in the same group (e.g., related questions via consistency_id) should agree; penalize soft label variance for local stability.
    # Simple English: "Related examples (like paraphrases) should have similar labels—don't let them vary wildly. Average their probabilities and punish if they're all over the place, ensuring the model doesn't contradict itself on similar stuff."
    direct_loss = 0.0
    num_groups = 0
    for group_uids in consistency_groups.values():
        group_indices = [uid_to_idx.get(u) for u in group_uids if u in uid_to_idx]
        if len(group_indices) > 1:
            group_soft = soft_labels[torch.tensor(group_indices)]
            direct_loss += group_soft.var(dim=0).mean()
            num_groups += 1
    terms['direct'] = direct_loss / max(num_groups, 1)
    
    # Entropy: Penalize high entropy in targets (encourage confident labels)
    # Intuition: Reward low-entropy (decisive) labels for targets, proxying downstream confidence from coherent propagation.
    # Simple English: "Make labels decisive (mostly yes or no, not 50/50 unsure). For each target, calculate how 'spread out' its probability is and add a small penalty if it's too wishy-washy—pushes toward bold, consistent choices that build confidence across predictions."
    entropy_loss = 0.0
    for t in tuples:
        target_idx = uid_to_idx[t['target_uid']]
        target_soft = soft_labels[target_idx]
        entropy = -(target_soft * torch.log(target_soft + 1e-8)).sum()
        entropy_loss += entropy
    terms['entropy'] = entropy_loss / max(num_tuples, 1)
    
    # New: Reward context for good evidence
    reward_loss = 0.0
    for t in tuples:
        target_idx = uid_to_idx[t['target_uid']]
        context_indices = torch.tensor([uid_to_idx[u] for u in t['context_uids']])
        if len(context_indices) == 0:
            continue
        
        target_soft = soft_labels[target_idx]
        target_entropy = -(target_soft * torch.log(target_soft + 1e-8)).sum()
        reward = 1.0 / (target_entropy + 1e-5)  # Higher for confident target
        
        context_soft = soft_labels[context_indices]
        context_entropy = -(context_soft * torch.log(context_soft + 1e-8)).sum(dim=1).mean()
        reward_loss += -reward * context_entropy  # Reward low context entropy
    terms['reward'] = reward_loss / max(num_tuples, 1)
    
    # Weighted sum (include 'reward')
    weighted_loss = sum(loss_weights.get(k, 0.0) * terms[k] for k in terms)
    total_loss = weighted_loss / (sum(loss_weights.values()) + loss_weights.get('reward', 0.0) or 1e-5)
    
    # For hacking: print terms
    if torch.is_grad_enabled() and verbose:
        print(f"Loss terms: { {k: v.item() if hasattr(v, 'item') else v for k,v in terms.items()} }")
    
    return total_loss

# Test with dummy
dummy_loss = coherence_loss(labels, tuples[:1], priors, consistency_groups, uid_to_idx, loss_weights)
print(f"Sample loss: {dummy_loss.item()}")
# %% [code]

def run_backprop_experiment(labels, tuples, priors, consistency_groups, uid_to_idx, loss_weights, df_data, df_preds, unique_uids, avg_diffs):
    """
    Run backprop experiment with given weights, return metrics.
    """
    # Copy labels
    current_labels = labels.clone().detach().requires_grad_(True)
    
    # optimizer = optim.LBFGS([current_labels], lr=0.1, max_iter=20)
    optimizer = optim.AdamW([current_labels], lr=0.1)
    losses = []
    epochs = 10
    
    for epoch in tqdm(range(epochs), desc="Optimizing"):
        def closure():
            optimizer.zero_grad()
            loss = coherence_loss(current_labels, tuples, priors, consistency_groups, uid_to_idx, loss_weights, verbose=False)
            loss.backward()
            return loss
        loss_val = optimizer.step(closure)
        losses.append(loss_val.item())
    
    # Post-process
    final_soft = F.softmax(current_labels, dim=1)
    final_hard = torch.argmax(final_soft, dim=1).cpu().numpy()
    
    # Output DF in memory
    output_data = []
    for uid in unique_uids:
        idx = uid_to_idx[uid]
        row = df_data.loc[uid].to_dict()
        row['uid'] = uid
        row['learned_soft_0'] = float(final_soft[idx, 0].detach())
        row['learned_soft_1'] = float(final_soft[idx, 1].detach())
        row['learned_hard'] = int(final_hard[idx])
        row['prior_diff'] = avg_diffs.get(uid, 0.0)
        output_data.append(row)
    
    df_output = pd.DataFrame(output_data)
    acc = (df_output['learned_hard'] == df_output['vanilla_label']).mean() if 'vanilla_label' in df_output.columns else 0.0
    
    # LLM acc
    llm_acc = 0.0
    if not df_preds.empty and 'target_uid' in df_preds.columns and 'vanilla_label' in df_data.columns:
        df_preds_local = df_preds.copy()
        df_preds_local['hard_llm_pred'] = (df_preds_local['raw_logprob_diff'] > 0).astype(int)
        common_preds = df_preds_local.merge(df_data.reset_index()[['uid', 'vanilla_label']], left_on='target_uid', right_on='uid', how='inner')
        if not common_preds.empty:
            llm_acc = (common_preds['hard_llm_pred'] == common_preds['vanilla_label']).mean()
    
    # ICM corr
    icm_corr = 0.0
    output_dir = Path("outputs/backprop")
    icm_path = output_dir.parent / "icm/truthfulqa/icm_final_labels.parquet"
    if icm_path.exists():
        df_icm = pd.read_parquet(icm_path)
        if 'uid' in df_icm.columns and 'label' in df_icm.columns:
            df_icm.set_index('uid', inplace=True)
            common_uids = set(df_output['uid']).intersection(df_icm.index)
            if common_uids and len(common_uids) > 1:
                backprop_hard_common = df_output[df_output['uid'].isin(common_uids)]['learned_hard'].values
                icm_labels_common = df_icm.loc[list(common_uids), 'label'].values
                icm_corr = np.corrcoef(backprop_hard_common, icm_labels_common)[0, 1]
    
    return {
        'acc': acc,
        'icm_corr': icm_corr,
        'final_loss': losses[-1] if losses else 0.0,
        'llm_acc': llm_acc
    }

# %% [code]

# Experiment with loss weights: Loop over variations, print key results

# Baseline weights
base_weights = {
    'mutual': 1.0,
    'ranking': 0.5,
    'prior': 0.1,
    'direct': 0.5,
    'entropy': 0.1,
    'reward': 0.2
}

# Example: Vary mutual weight (one at a time, as suggested)
mutual_variations = [0.1, 0.5, 1.0, 2.0]
results = {}

for mw in mutual_variations:
    weights = base_weights.copy()
    weights['mutual'] = mw
    print(f"\n--- Testing mutual_weight = {mw} ---")
    res = run_backprop_experiment(labels, tuples, priors, consistency_groups, uid_to_idx, weights, df_data, df_preds, unique_uids, avg_diffs)
    results[mw] = res
    print(f"Mutual {mw}: Acc {res['acc']:.4f}, ICM Corr {res['icm_corr']:.4f}, Final Loss {res['final_loss']:.4f}, LLM Acc {res['llm_acc']:.4f}")
