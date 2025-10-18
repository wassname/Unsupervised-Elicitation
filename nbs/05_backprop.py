# %% [markdown]
# # Backpropagation through ICM
# %% [code]
import srsly
import json
import random
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Literal
import numpy as np
import pandas as pd
from loguru import logger
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import simple_parsing  # Add this import after other imports
from loguru import logger
import sys
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed, skipping logging. Install with: uv pip install wandb")

# Configure loguru for JSON output without prefix
logger.remove()  # Remove default sink
logger.add(sys.stdout, format="{message}")

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
output_path = Path("outputs/icm/truthfulqa/predictions.jsonl")
preds = list(srsly.read_jsonl(output_path))
df_preds = pd.DataFrame(preds)
preds[0]


# replace with outputs.
output_path = Path("./outputs/icm/truthfulqa/icm_final_labels.parquet")
# Index(['uid', 'prompt', 'vanilla_label', 'consistency_id', 'consistency_key', 'label', 'score'],
df_preds = pd.read_parquet(output_path)
df_preds

# {'uid': 20,
#  'target_idx': 20,
#  'score': 0.9999999999991981,
#  'score': 27.85156273937173,
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

# Compute priors: avg score per uid
priors_diff = defaultdict(list)
for pred in preds:
    uid = pred['uid']
    diff = pred['score']
    priors_diff[uid].append(diff)

# Average diffs, default to 0 if no preds
# FIXME should we say mean /std
diffs = {uid: torch.tensor(priors_diff.get(uid, [0.0, 0.0])) for uid in unique_uids}
std_diffs = {uid: torch.std(v) if ((len(v)>1) and (torch.norm(v)>0)) else torch.tensor(0.0) for uid, v in diffs.items()}
norm_diffs = {uid: diffs[uid].mean() / (1.0 + std_diffs[uid]) for uid in unique_uids}
prior_logits = torch.tensor([[ -d / 2, d / 2 ] for d in norm_diffs.values()])  # [logit_0, logit_1]
# priors = F.softmax(prior_logits, dim=1)  # Soft priors [num_labels, 2]

priors = prior_logits

# Build tuples: subsample to 500 for speed
tuples = []
for pred in preds:  # Subsample
    context_uids = [c['uid'] for c in pred['context']]
    valid_context = [u for u in context_uids if u in uid_to_idx]  # Filter valid
    if len(valid_context) > 0 and pred['uid'] in uid_to_idx:
        tuples.append({
            'context_uids': valid_context,
            'uid': pred['uid'],
            'llm_pred_diff': pred['score'],
        })

# Precompute for speed: targets, llm_diffs, context_index_lists, group_index_lists
targets_list = []
llm_diffs_list = []
context_index_lists = []
valid_tuples = []  # Filtered tuples with valid context
for t in tuples:
    target_uid = t['uid']
    if target_uid not in uid_to_idx:
        continue
    target_idx = uid_to_idx[target_uid]
    context_indices = [uid_to_idx[u] for u in t['context_uids'] if u in uid_to_idx]
    if len(context_indices) > 0:
        targets_list.append(target_idx)
        llm_diffs_list.append(t['llm_pred_diff'])
        context_index_lists.append(context_indices)
        valid_tuples.append(t)

tuples = valid_tuples  # Update to valid only
targets = torch.tensor(targets_list)
llm_diffs = torch.tensor(llm_diffs_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Join with original data for consistency_id (add to tuples or separate)
# For now, create consistency_groups: dict of lists of uids per consistency_id
consistency_groups = defaultdict(list)
for uid, row in df_data.iterrows():
    consistency_groups[row['consistency_id']].append(uid)

# Precompute group indices for direct loss
group_index_lists = []
for group_uids in consistency_groups.values():
    group_indices = [uid_to_idx.get(u) for u in group_uids if u in uid_to_idx]
    if len(group_indices) > 1:
        group_index_lists.append(group_indices)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

context_index_tensors = [torch.tensor(lst, dtype=torch.long, device=device) for lst in context_index_lists]
group_index_tensors = [torch.tensor(lst, dtype=torch.long, device=device) for lst in group_index_lists]

logger.info(f"Num labels: {num_labels}, Num tuples: {len(tuples)}")
logger.info(f"Sample prior: {priors[0]}")
logger.info(f"Sample tuple: {tuples[0] if tuples else 'None'}")
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

logger.info(f"Labels shape: {labels.shape}")
logger.info(f"Loss weights: {loss_weights}")
# %% [code]
# Define modular loss function

def coherence_loss(labels, tuples, priors, consistency_groups, uid_to_idx, loss_weights, scale=1000.0, targets=None, llm_diffs=None, context_index_lists=None, group_index_lists=None, context_index_tensors=None, group_index_tensors=None, verbose=False, temperature=1.0):
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
    device = labels.device
    priors = priors.to(device)
    soft_labels = F.softmax(labels, dim=1)  # [num_labels, 2]
    rank_loss_fn = nn.MarginRankingLoss(margin=1.0)
    num_tuples = len(tuples)
    terms = {}
    
    # Build graph: lists for directed edges (context -> target) with weights
    # No extra deps; use loop-based aggregation for small num_labels (~800)
    num_nodes = soft_labels.shape[0]
    weighted_context = torch.zeros_like(soft_labels)  # [num_nodes, 2]
    degrees = torch.zeros(num_nodes, device=device)  # For normalization
    
    for t_idx in range(num_tuples):
        target_idx = targets[t_idx]
        context_indices = context_index_tensors[t_idx]
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
    
    # Updated Mutual: Use weighted_context for CE, vectorized where possible
    # Filter valid targets (degrees >0)
    valid_mask = degrees[targets] > 0
    valid_targets = targets[valid_mask]
    if len(valid_targets) == 0:
        mutual_loss = torch.tensor(0.0, device=device)
    else:
        context_agg_per_valid = weighted_context[valid_targets]  # [num_valid, 2]
        target_soft_per_valid = soft_labels[valid_targets]       # [num_valid, 2]
        
        # Asymmetric CE: context aggregation predicts target label (one-way inference)
        # This matches ICM's P(y_i | context) formulation - context should confidently predict target
        # Temperature annealing: high temp early (explore), low temp late (exploit)
        # Prevents premature collapse to uniform/degenerate solutions
        context_agg_tempered = F.softmax(context_agg_per_valid.log() / (temperature + 1e-8), dim=1)
        mutual_loss = -(target_soft_per_valid * (context_agg_tempered + 1e-12).log()).sum(dim=1).mean()
    terms['mutual'] = mutual_loss / max(num_tuples, 1)
    
    # Pairwise Ranking: Enforce ranking on learnable diff vs LLM pred diff
    # Intuition: LLM logprobs are better for relative rankings than absolute probs; ensure learned prob ranking matches LLM's scaled logprob diff.
    # Simple English: "The LLM ranked 'yes' higher than 'no' for this example—make sure your learned label agrees on which is stronger, with a safety margin to avoid ties. Like forcing your guesses to match the model's confidence order, not exact numbers."
    # Vectorized ranking
    score1 = soft_labels[targets, 1]  # [num_tuples, ]
    score0 = soft_labels[targets, 0]
    learned_diffs = score1 - score0
    signs = (llm_diffs > 0).float() * 2 - 1  # +1 if llm prefers 1, -1 else
    m = 1.0
    violations = torch.clamp(m - signs * learned_diffs, min=0.0)
    ranking_loss = violations.mean()
    terms['ranking'] = ranking_loss
    
    # Prior KL: Pull toward priors
    # Intuition: Anchor learned labels to fixed LLM priors (averaged logprob diffs) via KL on logits to prevent drift from model's initial biases.
    # Simple English: "Don't stray too far from the model's original hunches (its logprob biases for each example). Pull labels back toward these starting points gently, like a rubber band keeping you grounded in what the model already 'knows'."
    prior_loss = F.kl_div(F.log_softmax(labels, dim=1), F.softmax(priors, dim=1), reduction='batchmean')
    terms['prior'] = prior_loss 
    
    # Direct Consistency: Penalize variance within consistency groups
    # Intuition: Labels in the same group (e.g., related questions via consistency_id) should agree; penalize soft label variance for local stability.
    # Simple English: "Related examples (like paraphrases) should have similar labels—don't let them vary wildly. Average their probabilities and punish if they're all over the place, ensuring the model doesn't contradict itself on similar stuff."
    direct_loss = 0.0
    num_groups = len(group_index_lists)
    for g_idx in range(num_groups):
        group_indices = group_index_tensors[g_idx]
        group_soft = soft_labels[group_indices]
        direct_loss += group_soft.var(dim=0).mean()
    terms['direct'] = direct_loss / max(num_groups, 1)
    
    # Entropy: Penalize high entropy in targets (encourage confident labels)
    # Intuition: Reward low-entropy (decisive) labels for targets, proxying downstream confidence from coherent propagation.
    # Simple English: "Make labels decisive (mostly yes or no, not 50/50 unsure). For each target, calculate how 'spread out' its probability is and add a small penalty if it's too wishy-washy—pushes toward bold, consistent choices that build confidence across predictions."
    # Vectorized entropy
    target_soft_per_tuple = soft_labels[targets]  # [num_tuples, 2]
    entropies = -(target_soft_per_tuple * torch.log(target_soft_per_tuple + 1e-8)).sum(dim=1)
    entropy_loss = entropies.mean()
    terms['entropy'] = entropy_loss
    
    # New: Reward context for good evidence (still looped, but faster with precompute)
    reward_loss = 0.0
    for t_idx in range(num_tuples):
        target_idx = targets[t_idx]
        context_indices = context_index_tensors[t_idx]
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
    
    # For hacking: logger.info terms
    if torch.is_grad_enabled() and verbose:
        logger.info(f"Loss terms: { {k: v.item() if hasattr(v, 'item') else v for k,v in terms.items()} }")

    assert torch.isfinite(total_loss), "Loss is NaN or Inf"
    return total_loss

# Test with dummy (update call with precomputes)
# Note: for dummy, move to device manually if needed
device_test = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels_test = labels.to(device_test)
priors_test = priors.to(device_test)
targets_test = targets[:1].to(device_test)
llm_diffs_test = llm_diffs[:1].to(device_test)
context_index_lists_test = context_index_lists[:1]
group_index_lists_test = group_index_lists
context_index_tensors_test = [torch.tensor(lst, dtype=torch.long, device=device_test) for lst in context_index_lists_test]
group_index_tensors_test = [torch.tensor(lst, dtype=torch.long, device=device_test) for lst in group_index_lists_test]
dummy_loss = coherence_loss(labels_test, tuples[:1], priors_test, consistency_groups, uid_to_idx, loss_weights, targets=targets_test, llm_diffs=llm_diffs_test, context_index_lists=context_index_lists_test, group_index_lists=group_index_lists_test, context_index_tensors=context_index_tensors_test, group_index_tensors=group_index_tensors_test, verbose=False)
logger.info(f"Sample loss: {dummy_loss.item()}")
# %% [code]

@dataclass
class Config:
    epochs: int = 10
    device: str = "cuda"
    lr: float = 0.4
    weight_decay: float = 1e-4
    opt: Literal["adamw", "lbfgs"] = "adamw"
    
    test: bool = False
    subsample_size: int = 500 # just for test mode
    wandb: bool = False  # Enable wandb logging


def run_backprop_experiment(labels, tuples, priors, consistency_groups, uid_to_idx, loss_weights, df_data, df_preds, unique_uids, 
                            targets, llm_diffs, context_index_lists, group_index_lists, context_index_tensors, group_index_tensors,
                            config: Config,
                            device=None):
    """
    Run backprop experiment with given weights, return metrics.
    """
    if device is None:
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    if config.test:
        epochs = config.epochs
        num_keep = min(config.subsample_size, len(tuples))
        tuples = tuples[:num_keep]
        targets = targets[:num_keep]
        llm_diffs = llm_diffs[:num_keep]
        context_index_lists = context_index_lists[:num_keep]
        context_index_tensors = context_index_tensors[:num_keep]
        group_index_tensors = group_index_tensors  # unchanged
        logger.info(f"Test mode: subsampled to {num_keep} tuples, {epochs} epochs")
    else:
        epochs = config.epochs

    # Copy labels
    current_labels = labels.clone().detach().to(device).requires_grad_(True)

    priors_dev = priors.to(device)
    targets_dev = targets.to(device)
    llm_diffs_dev = llm_diffs.to(device)
    
    # Shared loss computation function
    def get_loss(temp=1.0):
        return coherence_loss(current_labels, tuples, priors_dev, consistency_groups, uid_to_idx, loss_weights, 
                              targets=targets_dev, llm_diffs=llm_diffs_dev, 
                              context_index_lists=context_index_lists, group_index_lists=group_index_lists,
                              context_index_tensors=context_index_tensors, group_index_tensors=group_index_tensors, 
                              verbose=False, temperature=temp)
    
    losses = []

    if config.opt == "adamw":
        optimizer = optim.AdamW([current_labels], lr=config.lr, weight_decay=config.weight_decay)
        for epoch in tqdm(range(epochs), desc="Optimizing"):
            optimizer.zero_grad()
            # Temperature annealing: 2.0 → 0.5 over epochs (exponential decay)
            temp = 0.5 + 1.5 * (0.95 ** epoch)
            loss = get_loss(temp)
            loss.backward()
            # Gradient clipping for stability at higher learning rates
            torch.nn.utils.clip_grad_norm_([current_labels], max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            
            # Log to wandb
            if config.wandb and WANDB_AVAILABLE:
                wandb.log({"loss": loss.item(), "temperature": temp, "epoch": epoch})
            
            assert torch.isfinite(current_labels).all(), "Labels contain NaN or Inf"
    else:  # lbfgs
        optimizer = optim.LBFGS([current_labels], lr=config.lr, max_iter=20)
        for epoch in tqdm(range(epochs), desc="Optimizing"):
            # Temperature annealing: 2.0 → 0.5 over epochs (exponential decay)
            temp = 0.5 + 1.5 * (0.95 ** epoch)
            def closure():
                optimizer.zero_grad()
                loss = get_loss(temp)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([current_labels], max_norm=1.0)
                return loss
            loss = optimizer.step(closure)
            losses.append(loss.item())
            
            # Log to wandb
            if config.wandb and WANDB_AVAILABLE:
                wandb.log({"loss": loss.item(), "temperature": temp, "epoch": epoch})
            
            assert torch.isfinite(current_labels).all(), "Labels contain NaN or Inf"
    
    # Post-process (unchanged)
    final_soft = F.softmax(current_labels, dim=1)
    final_hard = torch.argmax(final_soft, dim=1).cpu().numpy()
    
    output_data = []
    for uid in unique_uids:
        idx = uid_to_idx[uid]
        row = df_data.loc[uid].to_dict()
        row['uid'] = uid
        row['learned_soft_0'] = float(final_soft[idx, 0].detach())
        row['learned_soft_1'] = float(final_soft[idx, 1].detach())
        row['learned_hard'] = int(final_hard[idx])
        output_data.append(row)
    
    df_output = pd.DataFrame(output_data)
    acc = (df_output['learned_hard'] == df_output['vanilla_label']).mean() if 'vanilla_label' in df_output.columns else 0.0
    
    llm_acc = 0.0
    if not df_preds.empty and 'uid' in df_preds.columns and 'vanilla_label' in df_data.columns:
        df_preds_local = df_preds.copy()
        df_preds_local['hard_llm_pred'] = (df_preds_local['score'] > 0).astype(int)
        llm_acc = (df_preds_local['hard_llm_pred'] == df_preds_local['vanilla_label']).mean()
    
    icm_corr = 0.0
    output_dir = Path("outputs/backprop")
    output_dir.mkdir(parents=True, exist_ok=True)
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
        'llm_acc': llm_acc,
        'losses': losses
    }

# %% [code]

# Replace the experiment loop with this
if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config
    logger.info(f"Running backprop experiment with config: {config}")

    Path("outputs/backprop").mkdir(parents=True, exist_ok=True)

    base_weights = {
        'mutual': 1.0,
        'ranking': 0.5,
        'prior': 0.1,
        'direct': 0.5,
        'entropy': 0.1,
        'reward': 0.2
    }

    import matplotlib.pyplot as plt
    results = {}

    # Define experiments: name -> weights dict
    experiments = {}
    
    # Individual loss ablations
    for k in base_weights:
        weights = {kk: 0 for kk in base_weights}
        weights[k] = 1
        experiments[k] = weights
    
    # Combined losses
    experiments['combined'] = {
        'mutual': 0.6,
        'prior': 0.5,
        'ranking': 0.0,
        'direct': 0.0,
        'entropy': 0.0,
        'reward': 0.0
    }
    
    # Initialize wandb if enabled
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="unsupervised-elicitation-backprop",
            config={
                "lr": config.lr,
                "epochs": config.epochs,
                "optimizer": config.opt,
                "test_mode": config.test_mode,
                "num_tuples": len(tuples) if not config.test_mode else config.subsample_size,
                "num_labels": len(unique_uids)
            },
            name=f"lr{config.lr}_ep{config.epochs}_{'test' if config.test_mode else 'full'}"
        )
    
    # Run all experiments
    for name, weights in experiments.items():
        logger.info(f"\n--- Testing {name} ---")
        
        # Start new wandb run for each experiment if enabled
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.config.update({"experiment": name, "weights": weights})
        
        res = run_backprop_experiment(labels, tuples, priors, consistency_groups, uid_to_idx, weights, df_data, df_preds, unique_uids, 
                                      targets, llm_diffs, context_index_lists, group_index_lists, context_index_tensors, group_index_tensors, config)
        results[name] = res
        
        logger.info(f"Losses for {name}: {res['losses']}")
        
        # Log final metrics to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"{name}/final_acc": res['acc'],
                f"{name}/final_loss": res['final_loss'],
                f"{name}/icm_corr": res['icm_corr'],
                f"{name}/llm_acc": res['llm_acc']
            })
        
        plt.figure()
        plt.plot(res['losses'])
        plt.title(f"Loss curve for {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plot_path = Path(f"outputs/backprop/loss_{name}.png")
        plt.savefig(plot_path)
        
        # Log plot to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({f"{name}/loss_curve": wandb.Image(str(plot_path))})
        
        plt.close()
        logger.info(f"Saved plot to {plot_path}")
        
        logger.info(f"{name}: Acc {res['acc']:.4f}, ICM Corr {res['icm_corr']:.4f}, Final Loss {res['final_loss']:.4f}, LLM Acc {res['llm_acc']:.4f}")
    
    # Finish wandb run
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

# %%
