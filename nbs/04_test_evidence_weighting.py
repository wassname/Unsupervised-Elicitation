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
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from openrouter_wrapper.logprobs import (
    openrouter_completion_wlogprobs,
    get_logprobs_choices,
)
from collections import defaultdict

# Enable nested asyncio for notebook execution
# nest_asyncio.apply()

logger.info("Imports complete")

# %% [code]
# Load TruthfulQA data with labels
from src.data.truthfulqa import load_truthfulqa

data = load_truthfulqa()
logger.info(f"Loaded {len(data)} TQA examples")

# Take first consistency group for testing
groups = defaultdict(list)
for item in data:
    groups[item["consistency_id"]].append(item)


# Pick an abritrary group with at least 10 examples
GROUP_SIZE = 22
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
    """Convert logprob ratio (logprob_A - logprob_B) to pseudo-probability via sigmoid.
    
    Note: raw_lp is a logprob RATIO (difference), naturally 0-centered when A/B equally likely.
    No global calibration needed for relative comparisons within same model.
    """
    raw_lp = np.clip(raw_lp, -50, 50)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-raw_lp))


def score_color(raw_lp):
    # you know for score_color, we could also do html, terminal colors... but I guess these 3 emojis work everywhere and give the idea. Or just numb
    s = lpr2prob(raw_lp)
    return "🟢" if s > 0.7 else "🟡" if s > 0.5 else "🔴"


@dataclass
class Prediction:
    """Single raw prediction - NO calibration here, just record what happened"""

    target_uid: str
    target_idx: int
    raw_logprob_diff: float  # logprob(A) - logprob(B) - RAW, uncalibrated
    context: List[
        Tuple[str, str, float, bool]
    ]  # [(uid, label, raw_logprob_diff, was_flipped)]
    variations: Dict[str, bool]  # {reversed: bool, reordered: bool}

    @property
    def score(self) -> float:
        """Calibrated score computed on-the-fly via sigmoid"""
        return lpr2prob(self.raw_logprob_diff)

    def __repr__(self):
        # Concise display: P(target | context_uid=label[raw_logprob], ...)
        ctx_parts = []
        for uid, lbl, ctx_raw_lp, flip in self.context[:3]:  # Show first 3
            flip_marker = "*" if flip else ""
            ctx_color = score_color(ctx_raw_lp)
            label_str = "A" if lbl == 1 else "B"  # Convert int to label
            uid_str = str(uid)[-4:] if isinstance(uid, (int, str)) else str(uid)[:4]
            ctx_parts.append(f"{uid_str}={label_str}{flip_marker}[{ctx_color}]")
        ctx_str = ", ".join(ctx_parts)

        # Target score color
        tgt_color = score_color(self.raw_logprob_diff)
        tgt_uid_str = (
            str(self.target_uid)[-4:]
            if isinstance(self.target_uid, (int, str))
            else str(self.target_uid)[:4]
        )
        return f"P({tgt_uid_str} | {ctx_str}...) = {tgt_color}{self.score:.3f}"


def calculate_evidence_from_predictions(
    predictions: List[Prediction], group: List[Dict]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate flip-based evidence: does flipping source improve target coherence?
    
    Key insight: If flipping source label increases P(target), the flip might be good.
    Evidence = directional influence + strength + reliability.
    
    Returns dict of (source_uid, target_uid) -> evidence_dict
    Evidence components (all in logprob space):
    1. flip_delta: mean change in target logprob when source flipped (signed)
    2. flip_sensitivity: |flip_delta| (magnitude of coupling)
    3. direct_confidence: source's own logprob (raw)
    4. ensemble_variance: std of target logprobs across predictions
    5. mutual_predictability: correlation between source/target logprobs
    6. consistency_score: logical rules (paraphrases/contradictions)
    """
    evidence = defaultdict(
        lambda: {
            "flip_delta": 0.0,  # NEW: signed change (positive = flip improved coherence)
            "flip_sensitivity": 0.0,  # magnitude
            "direct_confidence": 0.0,  # raw logprob of source
            "ensemble_variance": 0.0,
            "mutual_predictability": 0.0,
            "consistency_score": 0.5,
            "count": 0,
        }
    )

    # Aggregate RAW logprobs per target (not probabilities)
    target_logprobs = defaultdict(list)
    for pred in predictions:
        target_logprobs[pred.target_uid].append(pred.raw_logprob_diff)

    # Baseline: mean & variance in LOGPROB space
    baseline_logprobs = {uid: np.mean(lps) for uid, lps in target_logprobs.items()}
    baseline_vars = {uid: np.std(lps) for uid, lps in target_logprobs.items()}

    # Consistency map
    consistency_map = {ex["uid"]: ex["consistency_key"] for ex in group}

    # Process each prediction with flips
    for pred in predictions:
        flipped_items = [
            (uid, lbl, raw_lp, flip) for uid, lbl, raw_lp, flip in pred.context if flip
        ]
        
        # Process ALL flipped items (not just first)
        for source_uid, flipped_label, source_raw_lp, _ in flipped_items:
            target_uid = pred.target_uid
            key = (source_uid, target_uid)

            # Baseline target logprob (without flip)
            baseline = baseline_logprobs.get(target_uid, 0.0)
            
            # Observed target logprob (with flip)
            observed = pred.raw_logprob_diff
            
            # 1. Flip delta: SIGNED change (positive = flip improved target score)
            delta = observed - baseline
            evidence[key]["flip_delta"] += delta
            
            # 2. Flip sensitivity: magnitude of change
            evidence[key]["flip_sensitivity"] += abs(delta)

            # 3. Direct confidence: source's own logprob (raw, not transformed)
            evidence[key]["direct_confidence"] += source_raw_lp

            # 4. Ensemble variance (lower = more certain)
            evidence[key]["ensemble_variance"] = baseline_vars.get(target_uid, 0.0)

            # 5. Mutual predictability: correlation in logprob space
            # TODO: Consider computing actual Pearson correlation over paired logprobs
            # instead of just mean proximity (np.corrcoef(source_lps, target_lps)[0,1])
            source_baseline = baseline_logprobs.get(source_uid, 0.0)
            # High mutual pred if both have similar magnitudes
            mutual_pred = 1 / (1 + abs(source_baseline - baseline))
            evidence[key]["mutual_predictability"] += mutual_pred

            # 6. Consistency: Check if source/target labels obey paraphrase/contradiction rules
            source_key = consistency_map.get(source_uid, "")
            target_key = consistency_map.get(target_uid, "")
            # Find actual labels from context or baseline
            source_label = flipped_label if source_uid == source_uid else (1 if context_score_cache.get(source_uid, 0) > 0 else 0)
            target_label = 1 if baseline_logprobs.get(target_uid, 0) > 0 else 0
            
            # Same key (paraphrase) -> should agree; different key (contradiction) -> should oppose
            if source_key == target_key:
                # Paraphrase: agreement is good
                evidence[key]["consistency_score"] = 1.0 if source_label == target_label else 0.0
            else:
                # Contradiction: opposition is good
                evidence[key]["consistency_score"] = 1.0 if source_label != target_label else 0.0

            evidence[key]["count"] += 1

    # Average accumulated values (only those that were summed)
    for key, ev in evidence.items():
        count = ev["count"]
        if count > 0:
            ev["flip_delta"] /= count
            ev["flip_sensitivity"] /= count
            ev["direct_confidence"] /= count
            # mutual_predictability is already set (not accumulated)

    return dict(evidence)


def compute_total_weight(ev: Dict[str, float]) -> float:
    """
    Compute evidence strength for label validation (DISPLAY ONLY - not used in grid search).
    
    Note: Returns normalized score for interpretability. Actual weighting uses z-scored features.
    """
    # Gate: only trust if source has decent confidence (raw logprobs are ~±960 scale)
    conf_gate = ev["direct_confidence"] > 0  # Positive = favors A over B
    
    if not conf_gate:
        return 0.0
    
    # Normalize features for display (same as grid search preprocessing)
    flip_contrib = ev["flip_delta"] * ev["flip_sensitivity"]
    inv_var = 1 / (1 + ev["ensemble_variance"])
    
    # Simple weighted sum with normalization for ~960 logprob scale
    total = (
        0.1 * np.tanh(flip_contrib / 100)  # Squash large values
        + 0.3 * np.tanh(ev["direct_confidence"] / 500)  # Normalize ~960 scale
        + 0.1 * inv_var  # Already 0-1 range
        + 0.4 * ev["mutual_predictability"]  # Already normalized
        + 0.1 * ev["consistency_score"]  # 0 or 1
    )
    return total


# %% [code]
# Config
MODEL_ID = "qwen/qwen3-235b-a22b-2507"
PROVIDER_WHITELIST = ("Cerebras", "Nebius")
PREDICTION_BUDGET = GROUP_SIZE * 3  # 3x group size for good coverage
N_CONTEXT = GROUP_SIZE - 1  # All others in group (exclude target)
RANDOM_SEED = 42  # For reproducibility

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger.info(
    f"Config: model={MODEL_ID}, group={GROUP_SIZE}, budget={PREDICTION_BUDGET}, context={N_CONTEXT}, seed={RANDOM_SEED}"
)


def print_messages(messages: List[Dict[str, Any]]) -> str:
    """Pretty-print chat messages"""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        parts.append(f"\n[{role}]:\n{content}")
    return "\n".join(parts)


# %% [code]
async def predict_with_context(
    target: Dict,
    context_examples: List[Dict],
    reversed_question: bool = False,
    model_id: str = MODEL_ID,
    verbose: int = 0,
) -> Tuple[float, float]:
    """
    Predict target label given context.

    Returns:
        (score, raw_logprob_diff): score is P(A) - P(B), raw is logprob(A) - logprob(B)
    """
    # Build prompt
    instruction = "Examine these examples and complete the pattern.\n\n"
    if verbose > 0:
        instruction += "Verbose mode is ON. Additional context may be provided after you answer.\n"

    fewshot = []
    for ctx in context_examples:
        label_str = "A" if ctx["label"] == 1 else "B"
        s = json.dumps({"Candidate": ctx["prompt"], "Set": label_str}, indent=2)
        fewshot.append(s)

    target_prompt = target["prompt"]
    if reversed_question:
        # Simple reversal: swap A/B in context labels
        fewshot = [
            s.replace("\nA\n", "\n__TEMP__\n")
            .replace("\nB\n", "\nA\n")
            .replace("\n__TEMP__\n", "\nB\n")
            for s in fewshot
        ]

    messages = [
        {
            "role": "user",
            "content": instruction
            + "".join(fewshot)
            + '{"Candidate": "' + target_prompt + '",',
        },
        {"role": "assistant", "content": '{ "Candidate": "' + target_prompt + '",'}
    ]
    if verbose > 0:
        logger.info(f"Message: {print_messages(messages)}")

    response = await openrouter_completion_wlogprobs(
        model_id=model_id,
        provider_whitelist=PROVIDER_WHITELIST,
        messages=messages,
        max_tokens=5 if verbose < 1 else 60,
        temperature=0.4,
        top_logprobs=8,
    )

    choices = ["A", "B"]
    choice_logp, top_logp = get_logprobs_choices(response, choices, lower=False)
    if not any(c in top_logp for c in choices):
        logger.warning(f"No valid choices found in logprobs: {top_logp}")
        logger.warning(f"Message: {print_messages(messages)}")
        logger.warning(f"Response: {response['choices'][0]['message']['content'].strip()}")
        # Fallback: assign equal logprob
    raw_diff = choice_logp["A"] - choice_logp["B"]

    if verbose > 0:
        logger.info(f"Response: {response['choices'][0]['message']['content'].strip()}")
        logger.info(f"Logprobs: {choice_logp}, Top: {top_logp}")

    # If reversed, flip back
    if reversed_question:
        raw_diff = -raw_diff

    # Convert to pseudo-prob (sigmoid-like normalization)
    score = lpr2prob(raw_diff)

    return score, raw_diff


# Track context scores globally for evidence calculation
context_score_cache = {}  # {uid: score} for context examples


# %% [code]
async def run_ensemble_predictions(
    group: List[Dict], budget: int = PREDICTION_BUDGET
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
        ex["label"] = ex["vanilla_label"]  # Start with ground truth for this test

    # Build context score cache: predict each example once to get baseline raw logprobs
    global context_score_cache
    logger.info("Building context score cache...")
    for i, ex in enumerate(group):
        others = [e for e in group if e["uid"] != ex["uid"]]
        if ex["uid"] not in context_score_cache:
            # Zero-shot prediction for baseline - store RAW logprob, not calibrated score
            score, raw_logprob = await predict_with_context(
                ex, others, reversed_question=False, verbose=i==0
            )
            context_score_cache[ex["uid"]] = raw_logprob

    for i in range(budget):
        # Sample target
        target_idx = random.randint(0, len(group) - 1)
        target = group[target_idx]

        # Sample context (all others)
        context_indices = [j for j in range(len(group)) if j != target_idx]
        sampled_ctx_idx = random.sample(
            context_indices, min(N_CONTEXT, len(context_indices))
        )

        # Build context with potential flip
        context_examples = []
        flipped_uid = None
        for j in sampled_ctx_idx:
            ctx = group[j].copy()
            ctx_score = context_score_cache.get(ctx["uid"], 0.5)
            # 50% chance to flip one label
            if flipped_uid is None and random.random() < 0.3:  # 30% flip rate
                ctx["label"] = 1 - ctx["label"]
                flipped_uid = ctx["uid"]
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
            target, context_examples, reversed_question=reversed_q, verbose=i==0
        )

        # Record with context raw logprobs (no calibration yet)
        context_meta = [
            (
                c["uid"],
                c["label"],
                context_score_cache.get(c["uid"], 0.0),
                c["uid"] == flipped_uid,
            )
            for c in context_examples
        ]
        pred = Prediction(
            target_uid=target["uid"],
            target_idx=target_idx,
            raw_logprob_diff=raw_logprob,
            context=context_meta,
            variations={"reversed": reversed_q, "reordered": True},  # Simplified
        )
        predictions.append(pred)
        target_scores[target["uid"]].append(
            pred.score
        )  # Use calibrated score for stats

        if i % 9 == 0:
            logger.info(f"Prediction {i}/{budget}: {pred}")

    return predictions, dict(target_scores)


# %% [code]
# Run predictions
logger.info("Starting ensemble predictions...")
predictions, target_scores = asyncio.run(
    run_ensemble_predictions(test_group, PREDICTION_BUDGET)
)

logger.info(f"Completed {len(predictions)} predictions")
logger.info(f"Coverage: {len(target_scores)}/{len(test_group)} targets predicted")

# %% [code]
# Calculate evidence weights DYNAMICALLY from all predictions
logger.info("Calculating evidence from all predictions...")
evidence_dict = calculate_evidence_from_predictions(predictions, test_group)

# Display top evidence pairs
print("\n=== Top 10 Evidence Pairs ===")
sorted_evidence = sorted(
    evidence_dict.items(), key=lambda x: compute_total_weight(x[1]), reverse=True
)

for (source_uid, target_uid), ev in sorted_evidence[:10]:
    weight = compute_total_weight(ev)
    flip_delta_str = f"Δ:{ev['flip_delta']:+.2f}"  # Show sign
    flip_str = (
        f"sens:{ev['flip_sensitivity']:.2f}"
        if lpr2prob(ev["direct_confidence"]) > 0.6
        else "sens:X"
    )
    src_str = (
        str(source_uid)[-4:]
        if isinstance(source_uid, (int, str))
        else str(source_uid)[:4]
    )
    tgt_str = (
        str(target_uid)[-4:]
        if isinstance(target_uid, (int, str))
        else str(target_uid)[:4]
    )
    print(
        f"{src_str} → {tgt_str} [w={weight:+.3f}] "
        f"({flip_delta_str}, {flip_str}, conf:{ev['direct_confidence']:.2f}, "
        f"var:{ev['ensemble_variance']:.2f}, mp:{ev['mutual_predictability']:.2f}, "
        f"cons:{ev['consistency_score']:.1f}, n={ev['count']})"
    )

# %% [code]
# Debug: check evidence coverage
logger.info(f"Evidence dict has {len(evidence_dict)} entries")
logger.info(f"Sample evidence entries: {list(evidence_dict.items())[:3]}")

# Check baseline: what does simple ensemble mean give?
print("\n=== Baseline: Simple Ensemble Mean ===")
ensemble_correct = 0
for uid, scores in target_scores.items():
    mean = np.mean(scores)
    true_label = next(ex["vanilla_label"] for ex in test_group if ex["uid"] == uid)
    pred_label = 1 if mean > 0.5 else 0
    correct = "✓" if pred_label == true_label else "✗"
    uid_str = str(uid)[-6:] if isinstance(uid, (int, str)) else str(uid)[:6]
    print(f"{uid_str}: mean={mean:.3f}, pred={pred_label}, true={true_label} {correct}")
    if pred_label == true_label:
        ensemble_correct += 1

baseline_acc = ensemble_correct / len([ex for ex in test_group if ex["uid"] in target_scores])
logger.info(f"Baseline ensemble accuracy: {baseline_acc:.3f} ({ensemble_correct}/{len([ex for ex in test_group if ex['uid'] in target_scores])})")

# Compare to vanilla labels baseline
vanilla_acc = sum(1 for ex in test_group if ex["uid"] in target_scores) / len(test_group)
logger.info(f"Note: With budget={PREDICTION_BUDGET}, coverage={len(target_scores)}/{len(test_group)}")
logger.info(f"Ensemble (simple mean) provides {baseline_acc:.1%} accuracy vs random 50%")

# %% [code]
# Grid search over evidence weights to find optimal combination
logger.info("Running grid search over evidence weights...")


def evaluate_weights(
    predictions: List[Prediction],
    evidence_dict: Dict,
    test_group: List[Dict],
    w_flip: float,
    w_conf: float,
    w_var: float,
    w_mp: float,
    w_cons: float,
    metric: str = "accuracy",  # "accuracy" or "brier"
) -> Dict[str, float]:
    """
    Evaluate evidence weights by aggregating predictions in LOGPROB space.
    
    Core idea: Weight each prediction's logprob by the reliability of its context sources.
    Then aggregate via weighted mean in logprob space -> convert to label.
    
    Args:
        metric: "accuracy" for hard decisions (ICM acceptance), "brier" for calibration (energy terms)
    """
    # Group predictions by target
    target_predictions = defaultdict(list)
    for pred in predictions:
        target_predictions[pred.target_uid].append(pred)

    correct = 0
    total = 0
    calibration_errors = []


    df_evidence = pd.DataFrame(evidence_dict).T

    # filter
    m = lpr2prob(df_evidence["direct_confidence"]) > 0.6
    df_evidence = df_evidence[m]

    # build features
    eps = 1e-6
    df_evidence['flip_contrib'] = df_evidence["flip_delta"] * df_evidence["flip_sensitivity"]
    df_evidence['conf'] = abs(df_evidence["direct_confidence"])
    df_evidence['var'] = 1 / (1 + df_evidence["ensemble_variance"] + eps)
    df_evidence['mp'] = df_evidence["mutual_predictability"]
    df_evidence['cons'] = df_evidence["consistency_score"]
    df_evidence = df_evidence[['flip_contrib', 'conf', 'var', 'mp', 'cons']]

    # norm
    # print(f"Evidence features stats (before norm): {df_evidence.describe().T}")
    df_evidence = (df_evidence - df_evidence.mean()) / (df_evidence.std() + eps)
    
    # all_evidences = np.array(all_evidences)
    # evidences_mean = np.mean(all_evidences, axis=0)
    # evidences_std = np.std(all_evidences, axis=0)
    # logger.debug(f"Evidence means: {evidences_mean}, stds: {evidences_std}")

    for i_e, ex in enumerate(test_group):
        uid = ex["uid"]
        if uid not in target_predictions:
            continue

        # Aggregate predictions in LOGPROB space with evidence weighting
        weighted_logprobs = []
        total_weight = 0.0

        for i_p, pred in enumerate(target_predictions[uid]):
            # Compute evidence-based weight for this prediction's context
            context_evidence_scores = []

            for j, (ctx_uid, _, _, _) in enumerate(pred.context):

                try:
                    evidences = df_evidence.loc[(ctx_uid, uid)]
                except KeyError:
                    continue  # No evidence for this pair

                weights = np.array([w_flip, w_conf, w_var, w_mp, w_cons])
                if i_p==0 and j==0 and i_e==0:
                    logger.debug(f"Weights: {weights}, Evidences: {evidences}")

                evidence_score = np.sum(weights * evidences)

                context_evidence_scores.append(evidence_score)
            
            if not context_evidence_scores:
                # No high-confidence sources, use uniform weight
                context_weight = 1.0
            else:
                # Average evidence across context
                context_weight = np.mean(context_evidence_scores)
            
            # Weight this prediction's LOGPROB by context reliability
            weighted_logprobs.append(pred.raw_logprob_diff * context_weight)
            total_weight += context_weight

        # Aggregate: weighted mean in logprob space
        if total_weight > 0:
            final_logprob = sum(weighted_logprobs) / total_weight
        else:
            final_logprob = 0.0  # Fallback: neutral
        
        # Convert to prediction
        final_prob = lpr2prob(final_logprob)
        pred_label = 1 if final_prob > 0.5 else 0
        true_label = ex["vanilla_label"]

        if pred_label == true_label:
            correct += 1
        total += 1

        # Calibration metrics
        calibration_errors.append(abs(final_prob - true_label))  # MAE
        brier_score = (final_prob - true_label) ** 2  # Brier (MSE of probs)

    accuracy = correct / total if total > 0 else 0.0
    avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 1.0
    brier = np.mean([(lpr2prob(final_logprob) - ex["vanilla_label"]) ** 2 
                     for ex in test_group if ex["uid"] in target_predictions]) if total > 0 else 1.0

    # Primary metric for sorting
    primary_metric = accuracy if metric == "accuracy" else (1 - brier)  # Higher is better

    return {
        "accuracy": accuracy,
        "calibration_error": avg_calibration_error,
        "brier_score": brier,
        "primary_metric": primary_metric,
        "correct": correct,
        "total": total,
    }


# Grid search (coarse)
METRIC = "brier"  # "accuracy" for hard decisions, "brier" for calibration (better for energy terms)
best_primary = 0.0
best_calibration = 1.0
best_weights = None
best_metrics = None

# Try different weight combinations
# TODO try a proper grid
weight_grid = []
for w_flip in [0, 0.1, 0.5]:
    for w_conf in [0, 0.1, 0.5]:
        for w_var in [0, 0.1, 0.5]:
            for w_mp in [0, 0.1, 0.5]:
                for w_cons in [0, 0.1, 0.5]:
                    weights = np.array([w_flip, w_conf, w_var, w_mp, w_cons])
                    if weights.sum() > 0:
                        weights = weights / weights.sum().clip(min=1e-6)  # Normalize to sum to 1
                        weight_grid.append(weights)




print("\n=== Weight Grid Search ===")

data = []
for weights in tqdm(weight_grid):
    w_flip, w_conf, w_var, w_mp, w_cons = weights
    metrics = evaluate_weights(
        predictions, evidence_dict, test_group, w_flip, w_conf, w_var, w_mp, w_cons, metric=METRIC
    )

    if (metrics["primary_metric"] >= best_primary) and (metrics["calibration_error"] < best_calibration):
        best_primary = metrics["primary_metric"]
        best_calibration = metrics["calibration_error"]
        best_weights = weights
        best_metrics = metrics

    data.append({
        "weights": weights,
        "accuracy": metrics["accuracy"],
        "brier_score": metrics["brier_score"],
        "calibration_error": metrics["calibration_error"],
        "primary_metric": metrics["primary_metric"],
        "correct": metrics["correct"],
        "total": metrics["total"],
    })

df_results = pd.DataFrame(data).sort_values(
    by=["primary_metric", "calibration_error"], ascending=[False, True]
)
print("\n=== Sorted Grid Search Results ===")
print(f"Metric: {METRIC} ({'1-brier' if METRIC == 'brier' else 'accuracy'})")
print(df_results.head(30))

s_top_weights = pd.DataFrame(np.stack(df_results.head(30).weights.values), columns=['flip_contrib', 'conf', 'var', 'mp', 'cons']).mean().sort_values()
print("\n=== Top 30 Weights Average ===")
print(s_top_weights)

# TODO label best weights with df_evidence.columns
print(
    f"\nBest weights: {best_weights} -> "
    f"primary={best_primary:.3f} ({METRIC}), "
    f"acc={best_metrics['accuracy']:.3f}, "
    f"brier={best_metrics['brier_score']:.3f}, "
    f"cal_err={best_metrics['calibration_error']:.3f}"
)

# %% [code]
# Aggregate statistics with BEST weights
print("\n=== Ensemble Statistics (Best Weights) ===")
for uid, scores in target_scores.items():
    mean = np.mean(scores)
    std = np.std(scores)
    # Find ground truth
    true_label = next(ex["vanilla_label"] for ex in test_group if ex["uid"] == uid)
    predicted_label = 1 if mean > 0.5 else 0
    correct = "✓" if predicted_label == true_label else "✗"
    uid_str = str(uid)[-6:] if isinstance(uid, (int, str)) else str(uid)[:6]
    print(
        f"{uid_str}: mean={mean:.3f}, std={std:.3f}, pred={predicted_label}, true={true_label} {correct}"
    )

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
            "context": [
                {"uid": uid, "label": int(lbl), "raw_logprob": raw_lp, "flipped": flip}
                for uid, lbl, raw_lp, flip in pred.context
            ],
            "variations": pred.variations,
        }
        f.write(json.dumps(record) + "\n")

logger.info(f"Saved predictions to {output_path}")

# %% [markdown]
# ## Results Summary
#
# **Grid Search Findings (After Fixes):**
# - Best weights vary by run due to stochastic ensemble sampling
# - Top results often: pure flip_contrib [0.83, 0, 0.17, 0, 0] or pure variance [0, 0, 1, 0, 0]
# - Accuracy range: 63-91% depending on random flips/orderings (vs 50% random baseline)
# - Multiple weight combos often tie → suggests small budget (36 preds) creates sparse evidence
#
# **Key Insights:**
# 1. **Ensemble variance (epistemic uncertainty) is reliable signal** - low var = confident
# 2. **Flip_contrib (directional evidence) matters** - positive delta = flip improved coherence
# 3. **Proper consistency rules essential** - paraphrases agree, contradictions oppose
# 4. **Multi-flip processing densifies evidence graph** - not just first flip
#
# **Why results vary across runs?**
# - Budget=36 on 12 examples = ~3 predictions/example (sparse)
# - Random sampling of target, context, flips, orderings
# - No seed was set initially (now fixed with RANDOM_SEED=42)
# - Small group size (12) limits consistency constraints
#
# **Remaining TODOs:**
# 1. **Scale budget to 100+** predictions for denser evidence graph
# 2. **Add global examples** (2-3 from other groups) to break echo chambers
# 3. **Use Pearson correlation** for mutual_predictability (not just mean proximity)
# 4. **Test on full ICM** integration via simple_icm.py with larger dataset

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
