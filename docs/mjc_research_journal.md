TODO
- Refactor to UV and simplify 
  - No Anthropic, just openai compatible API
  - No private org code needed
  - a non parrelal mode for debugging
- try with moral datasets e.g. daily dilemmas, ETHICS, Machiavelli, moral foundations vignettes


# 2025-10-09 09:22:36


Key hypothesis:
- LLM's logprobs are an internal only, non-calibrated measure of confidence. But with N-shots they are least use in context-learning
- But we can compare between predictions to get external measures of confidence
    - Consistency: some groups should be "only one true, or similar"
    - Mutual predictability: similar inputs should yield similar outputs
    - If flipping the label of one example makes the others more likely, that is weak evidence that the flip was good, and vice versa
- We can also vary factors we want to average ouut of the predictions
    - order: prevent positional bias
    - positive vs negative framing: prevent acquiescence bias


Because we can't do the full mutual predictibility we group by embedding into groups of 10 (partially overlapping or with a few global examples?), and have a prediction budget of 10*2

Then we do it one more time with new groups 10, and have a prediction budget of 10*2

We save all predictions e.g. (score, target, examples=((a_x, a_y), (b_x, b_y), ...))

Budget Tradeoff: 30 predictions/group is solid, but for large datasets (1000+ examples), total cost scales. 

Idea: Adaptive budgeting—spend more on high-uncertainty groups (e.g., high std in initial zero-shots). 

 only trust flips when base confidence is already decent (e.g., >0.6).

Evidence Sources (each contributes differently):

Direct Confidence (score): Raw logprob ratio → epistemic strength

Weight: Use when >0.6 threshold (your guard)
Nuance: Low score = don't trust downstream evidence
Flip Sensitivity (Δprob): How much prediction changes when context flips

Weight: High Δ = strong coupling (allosteric effect)
Nuance: Only meaningful if base confidence >0.6 (otherwise noise)
Ensemble Variance (consistency): Spread across multiple predictions

Weight: Low variance = stable concept, high = aleatoric uncertainty
Nuance: Variance in what? Score variance vs label disagreement
Mutual Predictability: Can other examples predict this one?

Weight: High mutual pred = coherent with group
Nuance: Requires checking reverse predictions (A→B and B→A)
Logical Consistency: Group rules (paraphrases agree, contradictions oppose)

Weight: Binary (consistent=1, inconsistent=0) or graded (similarity score)
Nuance: Strong evidence but requires known structure
 
## I need to clarify theory


Theory clarification (in Analysis section):

- scores? naive logprobs
- Epistemic = ensemble variance (model uncertainty) ?
- Aleatoric = consistency failures (data ambiguity) ?
- Evidence weights = structural confidence (relationship strength) ?


# 2025-10-16 13:57:48

## Idea context

    Absolutely, that clarification makes total sense—it streamlines the backprop idea significantly and keeps things focused on what you're trying to achieve (e.g., propagating evidence and consistency for unsupervised labels, inspired by ICM and your graph/rollout notes). Let me confirm I understand your setup, then break down how to implement it, compare it to graphs, and refine the complexity/gain ratings based on this "LLM-as-prior-provider" approach. This keeps the LLM out of the optimization loop (smart, since most LLMs like Llama or Claude aren't easily differentiable), treating it as a fixed oracle for generating initial priors or prediction tuples upfront.

    ### Confirmed Understanding of Your Backprop Setup
    - **Core Components**:
    - **Learnable Labels**: These are the only trainable parameters (e.g., soft probabilities for each example's label, like [prob_yes, prob_no]).
    - **Loss Function**: Something "weird" (custom) computed over n-shot examples and prediction tuples. For instance, the loss could measure mutual predictability: For each tuple (a set of n-shot context labels + a target prediction), compute how well the predicted label matches the "true" (learnable) one, perhaps using cross-entropy or ranking losses. This encourages coherence across the dataset.
    - **LLM's Role**: The LLM provides fixed priors (e.g., initial logprob-based scores like -(logprob_neg - logprob_pos) for each label) and/or generates the prediction tuples (e.g., by querying the LLM once to get few-shot predictions for various contexts). These are pre-computed and not updated during backprop—the LLM isn't in the graph or optimization.
    - **No LLM in the Graph**: Optimization is just over the small set of labels, using gradients from the loss to update them iteratively. This is efficient for small graphs (e.g., 100-1000 examples) and aligns with your idea of propagating confidence (e.g., if a label leads to more consistent predictions, it gets reinforced via the loss).

    This is reminiscent of techniques in papers like CCS (Contrast-Consistent Search) or even Fabien's probe experiments, where labels are optimized for internal consistency without external supervision. Your twist (using n-shot tuples and logprob rankings as priors) could make it more robust to outdated rollouts or scattered samples.

    ### How This Backprop Approach Works (Pros/Cons vs. Graphs)
    - **High-Level Flow**:
    1. **Pre-Compute from LLM**: Use the LLM to generate:
        - Priors: For each example, get logprobs for "yes" vs. "no" (or positive/negative), compute scores like score = logprob_pos - logprob_neg (positive favors "yes"). Treat these as initial values or regularization terms— they're rankings, not calibrated probs, as you noted.
        - Prediction Tuples: Generate a bunch of n-shot contexts (e.g., sample subsets of examples, feed to LLM as few-shot prompts, get predicted labels/probs for a target example). This creates a dataset of tuples like (context_labels, target_prediction).
    2. **Define Learnable Labels**: Initialize a tensor of soft labels (e.g., from priors).
    3. **Custom Loss**: For each tuple, plug in the current learnable labels (for the context) and compare to the LLM's pre-computed prediction (for the target). Loss could be:
        - Predictability term: Cross-entropy between predicted prob and learnable target label.
        - Consistency term: Penalize inconsistencies (e.g., if labels violate logical rules).
        - Prior term: KL divergence or L2 to the LLM priors, to anchor optimization.
        - Your propagation idea: Add a term that rewards labels leading to "more confident/consistent" downstream predictions (e.g., lower entropy in tuples that depend on it).
    4. **Backprop and Optimize**: Run gradient descent on the labels to minimize the total loss. Since the graph is small, it converges fast.

    - **Pros**:
    - Flexible for your "weird" losses (e.g., incorporating rollout datedness or graph weights as factors in the loss).
    - Handles soft labels and rankings naturally (e.g., use pairwise ranking losses like margin loss on logprob differences to avoid calibration issues).
    - Efficient: No LLM calls during training—just math on pre-computed data.

    - **Cons vs. Explicit Graphs**:
    - Less interpretable than a graph (harder to visualize dependencies), but more automated (gradients handle propagation implicitly).
    - If dependencies are sparse/simple, graphs (with BP) might be overkill but easier to debug. Backprop shines if you want to learn extra params (e.g., evidence weights) or handle non-linear losses.

    - **Is This Reinventing Graphs?**: Partially—it's a differentiable analog. The n-shot tuples implicitly define a graph (contexts as "neighbors," predictions as edge messages), and backprop propagates evidence like BP does. But you're not explicitly building/maintaining a graph structure, so it's a hybrid: graph-inspired but optimization-focused. Not reinventing—more like adapting (e.g., similar to how GNNs turn graphs into backprop-able nets). If you find backprop getting messy, fallback to explicit graphs is easy.

    ### Implementation Suggestions
    Here's a quick PyTorch sketch to get you started. Assume you have pre-computed `priors` (list of scores) and `tuples` (list of dicts with 'context_indices', 'target_index', 'llm_pred_prob' for yes/no).

    ```python
    import torch
    import torch.nn.functional as F

    # Setup
    num_labels = 100  # Number of examples/labels
    labels = torch.nn.Parameter(torch.randn(num_labels, 2))  # Learnable logits [yes, no] per label
    priors = torch.tensor([precomputed_scores])  # Shape: (num_labels, 2), from LLM logprobs
    tuples = [...]  # List of {'context_indices': [i,j,k], 'target_index': m, 'llm_pred': [prob_yes, prob_no]}

    # Custom loss function
    def coherence_loss(labels, tuples, priors, alpha=0.1, beta=0.1):
        loss = 0.0
        labels_soft = F.softmax(labels, dim=1)  # Soft probs
        
        # Predictability over tuples
        for t in tuples:
            # Simplified: Average context labels as "input" (or use more complex aggregation)
            context_avg = labels_soft[t['context_indices']].mean(dim=0)
            target_prob = labels_soft[t['target_index']]
            pred_loss = F.cross_entropy(context_avg.unsqueeze(0), target_prob.unsqueeze(0))  # Or ranking loss
            # Reward consistency/confidence: e.g., lower entropy if this label boosts downstream
            entropy_penalty = -torch.sum(target_prob * torch.log(target_prob))  # Lower is more confident
            loss += pred_loss + beta * entropy_penalty
        
        # Prior regularization (e.g., treat as soft targets)
        prior_loss = F.kl_div(F.log_softmax(labels, dim=1), priors.softmax(dim=1), reduction='batchmean')
        
        # Add consistency constraints (e.g., logical rules)
        # consistency_loss = ... (custom, e.g., penalize if label_i + label_j >1 for mutex pairs)
        
        return loss / len(tuples) + alpha * prior_loss  # + consistency_loss

    # Optimize (use 2nd-order as you suggested for small size)
    optimizer = torch.optim.LBFGS([labels], lr=0.1)  # Or Adam for first-order
    for epoch in range(10):  # LBFGS often converges in few steps
        def closure():
            optimizer.zero_grad()
            loss = coherence_loss(labels, tuples, priors)
            loss.backward()
            return loss
        optimizer.step(closure)
        print(f"Epoch {epoch}: Loss {closure().item()}")

    # Final labels: argmax or softmax
    final_labels = torch.argmax(F.softmax(labels, dim=1), dim=1)
    ```

    - **Tweaks**:
    - **For Rankings/Calibration**: Replace cross-entropy with a ranking loss (e.g., `torch.nn.MarginRankingLoss` on logprob differences) to treat LLM outputs as rankings.
    - **Learning Weights**: Add learnable params like `evidence_weights = torch.nn.Parameter(torch.ones(num_evidence_types))` and multiply them into the loss terms.
    - **Efficiency**: With num_labels=100 and 1000 tuples, this runs in seconds on CPU. L-BFGS is perfect for small params (as you said).
    - **Landmark Calibration**: Post-optimization, use a few "landmark" examples (e.g., known easy/hard ones) to scale the final probs.

    ### Updated Ratings (Complexity and Expected Gain)
    Adjusted for this simplified backprop (no LLM in the loop):
    - **Complexity: Medium** (Down from High—pre-computing makes it easier, but custom loss debugging adds some work. ~1-2 hours to prototype if you're PyTorch-familiar.)
    - **Expected Gain: Medium-High** (Same as before—could outperform simple accumulation on fuzzy tasks like TruthfulQA by learning nuanced propagations. Matches Fabien's ~100% PGR potential if your priors capture salient concepts. Gains depend on how well tuples cover dependencies; test vs. baseline ICM.)

    This setup feels like a great balance—powerful enough for your ideas without overcomplicating. If it works on a small test (e.g., 20 examples from the GitHub repo), scale it up! What dataset are you thinking of using first, or do you want help refining the loss function? 😊

    ----

    Below, I'll address each part of your query step by step, building on our discussion. I'll start with explanations for landmark calibration, pairwise ranking losses, and the consistency measures (with learnable weights). Then, I'll provide a full example code snippet for the loss function, incorporating all of this. The code expands on my previous PyTorch sketch, making it more concrete and tailored to your ideas (e.g., n-shot prediction tuples, logprob ratios as priors, multiple consistency terms with weights).

    This setup assumes:
    - You have pre-computed data from the LLM: `priors` (logprob-based scores/ratios for each label) and `tuples` (n-shot contexts with LLM-generated predictions).
    - Labels are learnable parameters (e.g., logits for yes/no).
    - Optimization is over these labels to minimize a custom loss that encourages coherence, using gradients for propagation.

    The result is an efficient, backprop-based label optimizer that's lightweight (runs fast on small datasets) and aligns with ICM-style unsupervised elicitation.

    ### 1. Landmark Calibration for Logprob Ratios as Priors
    Landmark calibration is a way to adjust raw LLM logprobs (which are often miscalibrated, as discussed in the LessWrong thread) using a few "landmark" examples with known properties (e.g., easy/true cases where you expect high confidence, or hard/false ones for low confidence). This turns uncalibrated logprobs into more reliable priors before optimization. It's not strictly necessary (you can skip it and rely on rankings), but it helps if you want priors to act like soft probabilities.

    - **How to Do It Pre-Optimization**:
    1. Select 5-20 landmark examples (e.g., from your dataset or synthetic ones: obvious "yes" math problems from GSM8K for high prob_yes, common misconceptions from TruthfulQA for low).
    2. Query the LLM for logprobs on these.
    3. Fit a simple calibrator: E.g., temperature scaling (find a temp `T` that minimizes cross-entropy on landmarks) or Platt scaling (linear transform: scaled_logit = a * logit + b).
    4. Apply to all logprob ratios: prior_score = (logprob_pos - logprob_neg) / T  (or whatever your ratio is).
    5. Convert to soft priors: prior_probs = softmax([prior_score, -prior_score]) for [yes, no].

    - **Integration as Priors in PyTorch**:
    - These calibrated ratios become fixed tensors (not parameters)—you'll use them in a regularization term (e.g., KL loss) to pull the learnable labels toward them.
    - **Will KL Loss Make This Happen End-to-End?** Yes! By adding a KL divergence term in the loss (as in my previous sketch), the optimization will naturally balance the priors with other terms (e.g., coherence). No need to make priors learnable—they're anchors. During backprop, gradients from KL will propagate to update labels toward calibrated values. If you want to learn how much to trust priors, add a learnable weight (e.g., `alpha` in the loss) and optimize it too (via hyperparam sweep or as a parameter).

    - **When to Skip**: If treating logprobs as rankings (via pairwise losses—see below), calibration is less critical. Just use raw differences as relative strengths.

    Example calibration code (pre-optimization):
    ```python
    import torch
    import torch.nn.functional as F

    # Assume landmarks: list of (logprob_pos, logprob_neg, true_label)  # true_label=1 for yes
    def calibrate_temperature(logprobs_pos, logprobs_neg, true_labels, temps=[0.5, 1.0, 2.0]):
        best_temp, best_loss = None, float('inf')
        for T in temps:
            scaled_logits = torch.tensor([(p - n) / T for p, n in zip(logprobs_pos, logprobs_neg)])
            preds = F.softmax(scaled_logits.unsqueeze(1), dim=1)[:, 0]  # Prob yes (assuming binary)
            loss = F.binary_cross_entropy(preds, torch.tensor(true_labels).float())
            if loss < best_loss:
                best_temp, best_loss = T, loss
        return best_temp

    # Apply to all priors
    temp = calibrate_temperature(landmark_pos, landmark_neg, landmark_true)
    priors = torch.tensor([(p - n) / temp for p, n in zip(all_logprobs_pos, all_logprobs_neg)])  # Shape: (num_labels,)
    priors = F.softmax(torch.stack([priors, -priors], dim=1), dim=1)  # To [prob_yes, prob_no]
    ```

    ### 2. Pairwise Ranking Losses (e.g., Margin Loss on Logprob Differences)
    Pairwise ranking losses treat logprobs as relative rankings (e.g., "is yes better than no?") rather than absolute probabilities, avoiding calibration issues entirely. This is reliable because LLM logprobs are better for comparisons (as you noted—they're often used for ranking preferences in RLHF).

    - **How It Works**:
    - Instead of cross-entropy (which assumes calibrated probs), you define pairs of options and penalize if the ranking doesn't match expectations.
    - Use `torch.nn.MarginRankingLoss` (or `torch.nn.PairwiseMarginRankingLoss`): It takes two scores (e.g., score_yes and score_no) and a target (1 if yes > no, -1 if no > yes, 0 if tie). Loss = max(0, -target * (score1 - score2) + margin). This enforces a margin between better/worse options.
    - **What Are the Pairs?**:
        - **Per-Label Pairs**: For each label, pair its "yes" vs. "no" logprob differences (e.g., from priors or tuple predictions). Target=1 if the learnable label leans yes (e.g., labels_soft[:,0] > 0.5).
        - **Across-Tuple Pairs**: For a prediction tuple, pair the LLM's predicted prob_yes vs. prob_no, and compare to the learnable target label. Or pair different tuples' predictions to enforce mutual predictability (e.g., "if context A predicts yes, it should rank higher than context B predicting no").
        - **Propagation Pairs**: To capture "downstream consistency," pair a label's score with aggregated scores from dependent tuples (e.g., if this label is in many contexts, ensure its ranking boosts overall coherence).

    - **Why Avoid Calibration Issues?**: It only cares about order (e.g., yes ranks above no by at least margin=1.0), not absolute values. Great for your logprob ratios.

    In code, it'll be a term in the loss (see full example below).

    ### 3. Measures of Consistency with Learnable/Hyperparam Weights
    Yes, modularizing the loss into weighted consistency measures is a smart way to prevent "cheating" (e.g., superficial solutions, as in ICM's logical constraints). You can make weights learnable (as parameters) for end-to-end adaptation, or sweep them as hyperparameters (safer if worried about overfitting/cheating). Start with 3-5 measures, weighted by a vector (e.g., [w_direct, w_downstream, w_mutual]).

    - **Suggested Measures** (with your ideas incorporated):
    - **Direct Consistency**: Penalizes labels that flip-flop (e.g., variance across similar tuples). Weight: High if you have noisy priors.
    - **Downstream Consistency/Contribution**: Measures how much a label improves predictions in dependent tuples (e.g., lower loss/entropy when this label is in the context). This propagates "if this label led to more confident predictions" as you described.
    - **Mutual Predictability**: As in ICM/Fabien's work—how well one label predicts another (e.g., cross-entropy between predicted and target in tuples).
    - **Others to Consider**:
        - **Logical Consistency**: Penalize violations of rules (e.g., mutex labels can't both be yes).
        - **Prior Fidelity**: KL to calibrated priors (as above).
        - **Entropy/Confidence**: Reward low-entropy (confident) labels, but only if they contribute to coherence.

    - **Learnable Weights?**: Yes—make them parameters (e.g., `weights = torch.nn.Parameter(torch.ones(3))`) and include in optimization. Or sweep (e.g., grid search [0.1, 0.5, 1.0] per weight) to avoid cheating. Learning is efficient for small #weights.

    ### 4. Full Example Code for Loss (and Optimization)
    Here's an integrated PyTorch example. It includes landmark calibration (pre-step), pairwise ranking loss, multiple consistency measures with learnable weights, and LBFGS for fast convergence (2-5x faster than Adam on small problems, as you referenced—great for <1000 labels).

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Pre-compute calibrated priors (from landmark calibration, as above)
    # Assume priors: torch.Tensor (num_labels, 2)  # [prob_yes, prob_no]
    # tuples: list of dicts, e.g., {'context_indices': [0,1,2], 'target_index': 3, 'llm_pred': [0.7, 0.3]}

    # Setup learnables
    num_labels = len(priors)
    labels = nn.Parameter(torch.randn(num_labels, 2))  # Logits [yes, no]
    consistency_weights = nn.Parameter(torch.ones(3))  # Learnable: [w_direct, w_downstream, w_mutual]

    # Custom loss function
    def coherence_loss(labels, tuples, priors, margin=1.0):
        labels_soft = F.softmax(labels, dim=1)  # Soft probs [num_labels, 2]
        weights = F.softplus(consistency_weights)  # Ensure positive
        loss = 0.0
        
        # Measure 1: Direct Consistency (e.g., low variance across tuples for same target)
        direct_loss = 0.0
        for target in set(t['target_index'] for t in tuples):  # Group by target
            target_preds = torch.stack([labels_soft[target] for t in tuples if t['target_index'] == target])
            direct_loss += target_preds.var(dim=0).mean()  # Variance penalty
        direct_loss /= num_labels or 1
        
        # Measure 2: Downstream Consistency (how much label contributes to low-entropy downstream)
        downstream_loss = 0.0
        for t in tuples:
            context_soft = labels_soft[t['context_indices']].mean(dim=0)  # Aggregated context
            target_soft = labels_soft[t['target_index']]
            entropy = -torch.sum(target_soft * torch.log(target_soft + 1e-8))  # Lower is better
            downstream_loss += entropy  # Or use as reward (negative)
        downstream_loss /= len(tuples) or 1
        
        # Measure 3: Mutual Predictability (cross-entropy between context avg and target)
        mutual_loss = 0.0
        for t in tuples:
            context_soft = labels_soft[t['context_indices']].mean(dim=0).unsqueeze(0)
            target_soft = labels_soft[t['target_index']].unsqueeze(0)
            mutual_loss += F.cross_entropy(context_soft, target_soft)
        mutual_loss /= len(tuples) or 1
        
        # Combine weighted consistencies
        consistency_loss = (weights[0] * direct_loss + weights[1] * downstream_loss + weights[2] * mutual_loss) / weights.sum()
        
        # Pairwise Ranking Loss (on logprob differences, e.g., per-tuple pairs)
        ranking_loss = 0.0
        rank_loss_fn = nn.MarginRankingLoss(margin=margin)
        for t in tuples:
            # Pair: LLM pred_yes vs. pred_no; target=1 if learnable leans yes
            score_yes = torch.tensor(t['llm_pred'][0]).log() - torch.tensor(t['llm_pred'][1]).log()  # Logprob diff
            score_no = -score_yes  # Opposite
            target = 1 if labels_soft[t['target_index'], 0] > 0.5 else -1
            ranking_loss += rank_loss_fn(score_yes, score_no, torch.tensor(target))
        ranking_loss /= len(tuples) or 1
        
        # Prior Regularization (KL to calibrated priors)
        prior_loss = F.kl_div(F.log_softmax(labels, dim=1), priors, reduction='batchmean')
        
        # Total loss
        loss = consistency_loss + 0.5 * ranking_loss + 0.1 * prior_loss  # Tune coefficients
        return loss

    # Optimize with LBFGS for fast convergence
    optimizer = torch.optim.LBFGS([labels, consistency_weights], lr=0.1, max_iter=20)
    for epoch in range(10):
        def closure():
            optimizer.zero_grad()
            loss = coherence_loss(labels, tuples, priors)
            loss.backward()
            return loss
        optimizer.step(closure)
        print(f"Epoch {epoch}: Loss {closure().item()}")

    # Final: argmax labels
    final_labels = torch.argmax(F.softmax(labels, dim=1), dim=1)
    ```

    - **Notes on the Code**:
    - **Customization**: Add more measures (e.g., logical constraints as `if label_i >0.5 and label_j >0.5: penalty +=1`). Tune margins/coeffs.
    - **Efficiency**: LBFGS converges 2-5x faster (often in 5-20 steps) on small setups. If learnable weights cause issues (cheating), fix them and sweep.
    - **Testing**: Run on a small subset (e.g., 10 labels, 50 tuples) to debug.

    This should give you a solid starting point—flexible, incorporates your ideas, and ties back to the paper's coherence maximization. If you test it and share results (or need tweaks, like adding more terms), let me know! 😊


## Further specific implementation details


We can take `nbs/05_backprop.py` which loads `Path("outputs/icm/evidence_test_predictions.jsonl")` which are the LLM predictions over TruthfulQA. Now we can keep it simple, and keep the loss function flexible and hackable (without being defensive about expections ,we can fix them instead of worry about them).

So first lets plan, if you took at `nbs/05_backprop.py` you see it loads the LLM predictions, and then creates a `tuples` list of dicts with keys. Could you strip out what we don't need, and instead add this pytorch backprop idea, with multiple things wighted in the loss functions? How would you do it, high level?
