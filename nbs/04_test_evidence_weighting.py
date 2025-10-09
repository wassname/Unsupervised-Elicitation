# %% [markdown]
# # Test Evidence Weighting for Ensemble ICM
# 
# Hypothesis: Calibrate predictions via ensemble with flip-evidence, debiasing through variations.
# 
# Steps:
# 1. Load TruthfulQA with pregenerated labels
# 2. Take 1 group of 10 similar examples (by embedding or consistency_id)
# 3. Run 20 predictions with variations (flips, reorders)
# 4. Calculate evidence weights: `|Δprob| * (1 - var(ensemble)) * consistency`
# 5. Visualize: prediction tuples like `P(target_212 | ctx_431=True, ctx_321=False)`

# %% [code]
import asyncio
import nest_asyncio
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from openrouter_wrapper.logprobs import openrouter_completion_wlogprobs, get_logprobs_choices
from collections import defaultdict

# Enable nested asyncio for notebook execution
nest_asyncio.apply()

logger.info("Imports complete")

# %% [code]
# Load TruthfulQA data with labels
from src.data.truthfulqa import load_truthfulqa

data = load_truthfulqa()
logger.info(f"Loaded {len(data)} TQA examples")

# Take first consistency group for testing
groups = defaultdict(list)
for item in data:
    groups[item['consistency_id']].append(item)


# Pick an abritrary group with at least 10 examples
GROUP_SIZE = 10
test_group = []
for gid, items in groups.items():
    test_group += items
    if len(test_group) >= GROUP_SIZE:
        break

# Display group
for i, ex in enumerate(test_group):
    print(f"{i}: {ex['prompt'][:80]}... | label={ex['vanilla_label']}")

# %% [code]
def lpr2prob(raw_lp):
    return 1 / (1 + np.exp(-raw_lp))

def score_color(raw_lp):
    # you know for score_color, we could also do html, terminal colors... but I guess these 3 emojis work everywhere and give the idea. Or just numb
    s = lpr2prob(raw_lp)
    return '🟢' if s > 0.7 else '🟡' if s > 0.5 else '🔴'

@dataclass
class Prediction:
    """Single raw prediction - NO calibration here, just record what happened"""
    target_uid: str
    target_idx: int
    raw_logprob_diff: float  # logprob(A) - logprob(B) - RAW, uncalibrated
    context: List[Tuple[str, str, float, bool]]  # [(uid, label, raw_logprob_diff, was_flipped)]
    variations: Dict[str, bool]  # {reversed: bool, reordered: bool}
    
    @property
    def score(self) -> float:
        """Calibrated score computed on-the-fly via sigmoid"""
        return lpr2prob(self.raw_logprob_diff)
    
    def __repr__(self):
        # Concise display: P(target | context_uid=label[raw_logprob], ...)
        ctx_parts = []
        for uid, lbl, ctx_raw_lp, flip in self.context[:3]:  # Show first 3
            flip_marker = '*' if flip else ''
            ctx_color = score_color(ctx_raw_lp)
            label_str = "A" if lbl == 1 else "B"  # Convert int to label
            uid_str = str(uid)[-4:] if isinstance(uid, (int, str)) else str(uid)[:4]
            ctx_parts.append(f"{uid_str}={label_str}{flip_marker}[{ctx_color}]")
        ctx_str = ", ".join(ctx_parts)
        
        # Target score color
        tgt_color = score_color(self.raw_logprob_diff)
        tgt_uid_str = str(self.target_uid)[-4:] if isinstance(self.target_uid, (int, str)) else str(self.target_uid)[:4]
        return f"P({tgt_uid_str} | {ctx_str}...) = {tgt_color}{self.score:.3f}"


def calculate_evidence_from_predictions(
    predictions: List[Prediction],
    group: List[Dict]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate multi-source evidence DYNAMICALLY from all predictions.
    
    Returns dict of (source_uid, target_uid) -> evidence_sources dict
    Evidence sources:
    1. flip_sensitivity: |Δprob| when source flipped
    2. direct_confidence: Score of source (gated at >0.6)
    3. ensemble_variance: Spread across predictions for target
    4. mutual_predictability: Correlation between source/target
    5. consistency_score: Logical rules
    """
    evidence = defaultdict(lambda: {
        'flip_sensitivity': 0.0,
        'direct_confidence': 0.0,
        'ensemble_variance': 1.0,  # High = bad
        'mutual_predictability': 0.0,
        'consistency_score': 0.5,
        'count': 0
    })
    
    # Aggregate scores per target
    target_scores = defaultdict(list)
    for pred in predictions:
        target_scores[pred.target_uid].append(pred.score)
    
    # Baseline: mean & variance
    baseline_scores = {uid: np.mean(scores) for uid, scores in target_scores.items()}
    baseline_vars = {uid: np.std(scores) for uid, scores in target_scores.items()}
    
    # Consistency map
    consistency_map = {ex['uid']: ex['consistency_key'] for ex in group}
    
    # Process each prediction
    for pred in predictions:
        flipped = [(uid, lbl, raw_lp, flip) for uid, lbl, raw_lp, flip in pred.context if flip]
        if not flipped:
            continue
        
        source_uid, _, source_raw_lp, _ = flipped[0]
        target_uid = pred.target_uid
        key = (source_uid, target_uid)
        
        # 1. Flip sensitivity
        baseline = baseline_scores.get(target_uid, 0.5)
        delta = abs(pred.score - baseline)
        evidence[key]['flip_sensitivity'] += delta
        
        # 2. Direct confidence (from raw logprob)
        source_score = 1 / (1 + np.exp(-source_raw_lp))
        evidence[key]['direct_confidence'] += source_score
        
        # 3. Ensemble variance (lower = better)
        evidence[key]['ensemble_variance'] = baseline_vars.get(target_uid, 1.0)
        
        # 4. Mutual predictability (correlation)
        source_baseline = baseline_scores.get(source_uid, 0.5)
        mutual_pred = 1 - abs(source_baseline - baseline)
        evidence[key]['mutual_predictability'] += mutual_pred
        
        # 5. Consistency
        source_key = consistency_map.get(source_uid, "")
        target_key = consistency_map.get(target_uid, "")
        evidence[key]['consistency_score'] = 1.0 if source_key == target_key else 0.5
        
        evidence[key]['count'] += 1
    
    # Average accumulated values
    for key, ev in evidence.items():
        count = ev['count']
        if count > 0:
            ev['flip_sensitivity'] /= count
            ev['direct_confidence'] /= count
            ev['mutual_predictability'] /= count
    
    return dict(evidence)


def compute_total_weight(ev: Dict[str, float]) -> float:
    """Compute total weight from evidence sources with gating"""
    # Gate flip sensitivity by direct confidence
    flip_contrib = ev['flip_sensitivity'] if ev['direct_confidence'] > 0.6 else 0.0
    
    # Weighted combination (tune empirically)
    total = (
        0.3 * flip_contrib +
        0.2 * ev['direct_confidence'] +
        0.2 * (1 - ev['ensemble_variance']) +  # Low var = good
        0.2 * ev['mutual_predictability'] +
        0.1 * ev['consistency_score']
    )
    return total
    
#     def __post_init__(self):
#         # Guard: only trust flip evidence if base confidence >0.6
#         flip_contrib = self.flip_sensitivity if self.direct_confidence > 0.6 else 0.0
        
#         # Weighted combination (tune these weights empirically)
#         self.total_weight = (
#             0.3 * flip_contrib +  # Flip sensitivity (gated)
#             0.2 * self.direct_confidence +  # Direct score
#             0.2 * (1 - self.ensemble_variance) +  # Low variance = good
#             0.2 * self.mutual_predictability +  # High mutual pred = good
#             0.1 * self.consistency_score  # Binary/graded consistency
#         )
    
#     def __repr__(self):
#         arrow = "→" if self.flip_sensitivity > 0 else "↓"
#         # Show breakdown of evidence sources
#         sources = [
#             f"flip:{self.flip_sensitivity:.2f}" if self.direct_confidence > 0.6 else "flip:X",
#             f"conf:{self.direct_confidence:.2f}",
#             f"var:{self.ensemble_variance:.2f}",
#             f"mp:{self.mutual_predictability:.2f}",
#             f"cons:{self.consistency_score:.1f}"
#         ]
#         sources_str = ", ".join(sources)
#         return f"{self.source_uid[-4:]} {arrow} {self.target_uid[-4:]} [w={self.total_weight:.3f}] ({sources_str})"

# %% [code]
# Config
MODEL_ID = "meta-llama/llama-3.1-8b-instruct"
PROVIDER_WHITELIST = ('Cerebras', 'Nebius')
PREDICTION_BUDGET = GROUP_SIZE * 3  # 3x group size for good coverage
N_CONTEXT = GROUP_SIZE - 1  # All others in group (exclude target)

logger.info(f"Config: model={MODEL_ID}, group={GROUP_SIZE}, budget={PREDICTION_BUDGET}, context={N_CONTEXT}")

# %% [code]
async def predict_with_context(
    target: Dict,
    context_examples: List[Dict],
    reversed_question: bool = False,
    model_id: str = MODEL_ID
) -> Tuple[float, float]:
    """
    Predict target label given context.
    
    Returns:
        (score, raw_logprob_diff): score is P(A) - P(B), raw is logprob(A) - logprob(B)
    """
    # Build prompt
    instruction = "Examine these examples and complete the pattern.\n\n"
    
    fewshot = []
    for ctx in context_examples:
        label_str = "A" if ctx['label'] == 1 else "B"
        fewshot.append(f"## Candidate:\n{ctx['prompt']}\n## Set:\n{label_str}\n")
    
    target_prompt = target['prompt']
    if reversed_question:
        # Simple reversal: swap A/B in context labels
        fewshot = [s.replace("\nA\n", "\n__TEMP__\n").replace("\nB\n", "\nA\n").replace("\n__TEMP__\n", "\nB\n") 
                   for s in fewshot]
    
    messages = [
        {"role": "user", "content": instruction + "".join(fewshot) + f"## Candidate:\n{target_prompt}"},
        {"role": "assistant", "content": "\n## Set:"}
    ]
    
    response = await openrouter_completion_wlogprobs(
        model_id=model_id,
        provider_whitelist=PROVIDER_WHITELIST,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.4,
        top_logprobs=8,
    )
    
    choice_logp, top_logp = get_logprobs_choices(response, ["A", "B"], lower=False)
    raw_diff = choice_logp["A"] - choice_logp["B"]
    
    # If reversed, flip back
    if reversed_question:
        raw_diff = -raw_diff
    
    # Convert to pseudo-prob (sigmoid-like normalization)
    score = 1 / (1 + np.exp(-raw_diff))  # Maps logprob diff to [0,1]
    
    return score, raw_diff


# Track context scores globally for evidence calculation
context_score_cache = {}  # {uid: score} for context examples

# %% [code]
async def run_ensemble_predictions(
    group: List[Dict],
    budget: int = PREDICTION_BUDGET
) -> Tuple[List[Prediction], Dict[str, List[float]]]:
    """
    Run ensemble predictions on a group with variations.
    
    Returns:
        (predictions_list, target_scores): predictions_list is all Prediction objects,
        target_scores maps uid -> list of scores for aggregation
    """
    predictions = []
    target_scores = defaultdict(list)
    
    # Initialize labels: use vanilla_label as starting point
    for ex in group:
        ex['label'] = ex['vanilla_label']  # Start with ground truth for this test
    
    # Build context score cache: predict each example once to get baseline scores
    global context_score_cache
    logger.info("Building context score cache...")
    for ex in group:
        if ex['uid'] not in context_score_cache:
            # Zero-shot prediction for baseline
            score, _ = await predict_with_context(ex, [], reversed_question=False)
            context_score_cache[ex['uid']] = score
    
    for i in range(budget):
        # Sample target
        target_idx = random.randint(0, len(group) - 1)
        target = group[target_idx]
        
        # Sample context (all others)
        context_indices = [j for j in range(len(group)) if j != target_idx]
        sampled_ctx_idx = random.sample(context_indices, min(N_CONTEXT, len(context_indices)))
        
        # Build context with potential flip
        context_examples = []
        flipped_uid = None
        for j in sampled_ctx_idx:
            ctx = group[j].copy()
            ctx_score = context_score_cache.get(ctx['uid'], 0.5)
            # 50% chance to flip one label
            if flipped_uid is None and random.random() < 0.3:  # 30% flip rate
                ctx['label'] = 1 - ctx['label']
                flipped_uid = ctx['uid']
                flip_flag = True
            else:
                flip_flag = False
            context_examples.append(ctx)
            
        # Random variations
        reversed_q = random.random() < 0.2  # 20% reverse
        if random.random() < 0.3:  # 30% reorder
            random.shuffle(context_examples)
        
        # Predict
        score, raw_logprob = await predict_with_context(
            target, context_examples, reversed_question=reversed_q
        )
        
        # Record with context raw logprobs (no calibration yet)
        context_meta = [(c['uid'], c['label'], context_score_cache.get(c['uid'], 0.0), c['uid'] == flipped_uid) 
                        for c in context_examples]
        pred = Prediction(
            target_uid=target['uid'],
            target_idx=target_idx,
            raw_logprob_diff=raw_logprob,
            context=context_meta,
            variations={'reversed': reversed_q, 'reordered': True}  # Simplified
        )
        predictions.append(pred)
        target_scores[target['uid']].append(pred.score)  # Use calibrated score for stats
        
        if i % 5 == 0:
            logger.info(f"Prediction {i}/{budget}: {pred}")
    
    return predictions, dict(target_scores)

# %% [code]
# Run predictions
logger.info("Starting ensemble predictions...")
predictions, target_scores = asyncio.run(run_ensemble_predictions(test_group, PREDICTION_BUDGET))

logger.info(f"Completed {len(predictions)} predictions")
logger.info(f"Coverage: {len(target_scores)}/{len(test_group)} targets predicted")

# %% [code]
# Calculate evidence weights DYNAMICALLY from all predictions
logger.info("Calculating evidence from all predictions...")
evidence_dict = calculate_evidence_from_predictions(predictions, test_group)

# Display top evidence pairs
print("\n=== Top 10 Evidence Pairs ===")
sorted_evidence = sorted(evidence_dict.items(), 
                         key=lambda x: compute_total_weight(x[1]), 
                         reverse=True)

for (source_uid, target_uid), ev in sorted_evidence[:10]:
    weight = compute_total_weight(ev)
    flip_str = f"flip:{ev['flip_sensitivity']:.2f}" if ev['direct_confidence'] > 0.6 else "flip:X"
    src_str = str(source_uid)[-4:] if isinstance(source_uid, (int, str)) else str(source_uid)[:4]
    tgt_str = str(target_uid)[-4:] if isinstance(target_uid, (int, str)) else str(target_uid)[:4]
    print(f"{src_str} → {tgt_str} [w={weight:.3f}] "
          f"({flip_str}, conf:{ev['direct_confidence']:.2f}, "
          f"var:{ev['ensemble_variance']:.2f}, mp:{ev['mutual_predictability']:.2f}, "
          f"cons:{ev['consistency_score']:.1f}, n={ev['count']})")

# %% [code]
# Grid search over evidence weights to find optimal combination
logger.info("Running grid search over evidence weights...")

def evaluate_weights(
    evidence_dict: Dict,
    test_group: List[Dict],
    w_flip: float,
    w_conf: float,
    w_var: float,
    w_mp: float,
    w_cons: float
) -> Dict[str, float]:
    """
    Compute total weights with given coefficients and evaluate vs ground truth.
    Returns metrics dict.
    """
    # Aggregate evidence per target
    target_evidence = defaultdict(lambda: {'flip': [], 'conf': [], 'var': [], 'mp': [], 'cons': []})
    
    for (source_uid, target_uid), ev in evidence_dict.items():
        flip_contrib = ev['flip_sensitivity'] if ev['direct_confidence'] > 0.6 else 0.0
        target_evidence[target_uid]['flip'].append(flip_contrib)
        target_evidence[target_uid]['conf'].append(ev['direct_confidence'])
        target_evidence[target_uid]['var'].append(ev['ensemble_variance'])
        target_evidence[target_uid]['mp'].append(ev['mutual_predictability'])
        target_evidence[target_uid]['cons'].append(ev['consistency_score'])
    
    # Predict labels based on weighted evidence
    correct = 0
    total = 0
    calibration_errors = []
    
    for ex in test_group:
        uid = ex['uid']
        if uid not in target_evidence:
            continue
        
        ev = target_evidence[uid]
        # Average evidence sources (in logprob-like space)
        avg_flip = np.mean(ev['flip']) if ev['flip'] else 0.0
        avg_conf = np.mean(ev['conf']) if ev['conf'] else 0.5
        avg_var = np.mean(ev['var']) if ev['var'] else 1.0
        avg_mp = np.mean(ev['mp']) if ev['mp'] else 0.0
        avg_cons = np.mean(ev['cons']) if ev['cons'] else 0.5
        
        # Weighted combination
        evidence_score = (
            w_flip * avg_flip +
            w_conf * avg_conf +
            w_var * (1 - avg_var) +  # Low var = good
            w_mp * avg_mp +
            w_cons * avg_cons
        )
        
        # Normalize to [0,1]
        norm_score = evidence_score / (w_flip + w_conf + w_var + w_mp + w_cons)
        
        pred_label = 1 if norm_score > 0.5 else 0
        true_label = ex['vanilla_label']
        
        if pred_label == true_label:
            correct += 1
        total += 1
        
        # Calibration: how far is confidence from 0/1?
        calibration_errors.append(abs(norm_score - true_label))
    
    accuracy = correct / total if total > 0 else 0.0
    avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 1.0
    
    return {
        'accuracy': accuracy,
        'calibration_error': avg_calibration_error,
        'correct': correct,
        'total': total
    }

# Grid search (coarse)
best_acc = 0.0
best_weights = None
best_metrics = None

# Try different weight combinations
weight_grid = [
    (0.3, 0.2, 0.2, 0.2, 0.1),  # Original baseline
    (0.4, 0.2, 0.1, 0.2, 0.1),  # More flip
    (0.2, 0.3, 0.2, 0.2, 0.1),  # More conf
    (0.2, 0.2, 0.3, 0.2, 0.1),  # More var
    (0.2, 0.2, 0.2, 0.3, 0.1),  # More mp
    (0.5, 0.1, 0.1, 0.2, 0.1),  # Heavy flip
    (0.1, 0.4, 0.1, 0.3, 0.1),  # Conf + MP
    (0.3, 0.3, 0.2, 0.2, 0.0),  # No consistency
]

print("\n=== Weight Grid Search ===")
print("Format: (flip, conf, var, mp, cons) -> acc, cal_err")

for weights in weight_grid:
    w_flip, w_conf, w_var, w_mp, w_cons = weights
    metrics = evaluate_weights(evidence_dict, test_group, w_flip, w_conf, w_var, w_mp, w_cons)
    
    print(f"{weights} -> acc={metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']}), "
          f"cal_err={metrics['calibration_error']:.3f}")
    
    if metrics['accuracy'] > best_acc:
        best_acc = metrics['accuracy']
        best_weights = weights
        best_metrics = metrics

print(f"\nBest weights: {best_weights} -> acc={best_acc:.3f}, cal_err={best_metrics['calibration_error']:.3f}")

# %% [code]
# Aggregate statistics with BEST weights
print("\n=== Ensemble Statistics (Best Weights) ===")
for uid, scores in target_scores.items():
    mean = np.mean(scores)
    std = np.std(scores)
    # Find ground truth
    true_label = next(ex['vanilla_label'] for ex in test_group if ex['uid'] == uid)
    predicted_label = 1 if mean > 0.5 else 0
    correct = "✓" if predicted_label == true_label else "✗"
    uid_str = str(uid)[-6:] if isinstance(uid, (int, str)) else str(uid)[:6]
    print(f"{uid_str}: mean={mean:.3f}, std={std:.3f}, pred={predicted_label}, true={true_label} {correct}")

# %% [code]
# Save predictions as JSONL
output_path = Path("outputs/icm/evidence_test_predictions.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for pred in predictions:
        record = {
            "target_uid": pred.target_uid,
            "target_idx": pred.target_idx,
            "score": pred.score,
            "raw_logprob_diff": pred.raw_logprob_diff,
            "context": [{"uid": uid, "label": int(lbl), "raw_logprob": raw_lp, "flipped": flip} 
                        for uid, lbl, raw_lp, flip in pred.context],
            "variations": pred.variations
        }
        f.write(json.dumps(record) + "\n")

logger.info(f"Saved predictions to {output_path}")

# %% [markdown]
# ## Results Summary
# 
# **Grid Search Findings:**
# - Most weight combinations achieve ~50% accuracy (random baseline)
# - Best: (0.3, 0.2, 0.2, 0.2, 0.1) = 50% acc, cal_err=0.509
# - Heavy flip weighting (0.5, 0.1, 0.1, 0.2, 0.1) also 50% but higher cal_err
# 
# **Why low signal?**
# 1. **Small sample**: 30 predictions on 10 examples = sparse evidence graph
# 2. **Redundant sources**: All evidence ~0.62 conf, suggesting sources correlated
# 3. **Missing global context**: Each prediction uses only group members (echo chamber)
# 
# **Next steps:**
# 1. **Scale up**: 100+ predictions to densify evidence graph
# 2. **Add global examples**: Mix in 2-3 random examples from other groups
# 3. **Logprob normalization**: Convert all evidence to logprob space before combining
# 4. **Learn weights**: Use logistic regression on (flip, conf, var, mp, cons) → correctness

# %% [markdown]
# ## Analysis & Next Steps
# 
# **Observations:**
# - Evidence weights capture flip sensitivity—high weights = strong coupling between examples
# - Ensemble variance reveals aleatoric uncertainty (some examples inherently ambiguous)
# - Calibration: mean scores closer to 0/1 than raw logprobs (sigmoid normalization helps)
# 
# **Refinements Needed:**
# 1. **Consistency weighting**: Currently hardcoded to 1, should check group rules (paraphrases agree, contradictions oppose)
# 2. **Global examples**: Mix 2 random examples per group to prevent local echo chambers
# 3. **Adaptive budget**: Spend more on high-variance targets
# 4. **Directional evidence**: Track if flip improves or worsens global energy
# 
# **Theory Clarification (Epistemic vs Aleatoric):**
# - **Epistemic**: Reducible via more/better context → measured by variance across ensemble (low var = model "knows")
# - **Aleatoric**: Irreducible ambiguity in data → surfaces as consistency violations (e.g., paraphrases disagree)
# - **Our method**: Ensemble variance ≈ epistemic, consistency failures ≈ aleatoric, evidence weights ≈ structural/relational confidence
# 
# **Concise Display Idea:**
# The `Prediction.__repr__` shows `P(uid | ctx1=lbl1*, ctx2=lbl2) = 0.75` where `*` marks flips. 
# Could extend to a class that auto-formats for logging/notebooks.
