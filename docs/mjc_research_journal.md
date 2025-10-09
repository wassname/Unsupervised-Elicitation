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
