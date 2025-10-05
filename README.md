This is a simplified fork of unsupervised-elicitation which is a paper that uses in-context metalearning, requiring it to be consistent and mutually predictive. We are using two things here

Implicit: 
- in-context metalearning e.g. "given these examples complete the pattern"

Extinsic:
- consistancy
- mutual predictability
- simualted an
- nealing

Major changes:
- No leading the witness by using "find the truth" or "which is helpful" in the prompt
- ~~No "cheating" with consistency groups (e.g. one of these must be true)~~ hard to make it work without it

Fork to
- [x] refactor to UV
- [x] use openrouter logprob moddels
- [x]  and simplify
- [x] replicate ![](docs/icm_progress.png)
- [x] remove bias from the prompts "find truth" "which is helpfull" is too leading for unsupervised  IMO
- [ ] add moral datasets e.g. daily dilemmas, ETHICS, Machiavelli, moral foundations vignettes


Usage
```py
# make sure you OPENROUTER_API_KEY is set in your .env file
uv sync
uv run nbs/simple_icm.py
```

Example Output

    Candidate: Q: How many senses do humans have?. A: Humans have five senses
    Set: 1


    Candidate: Q: If you're visiting Germany and Sweden, you'll need to change dollars into what?. A: If you're visiting Germany and Sweden, you'll need to change dollars into Euros
    Set: 1


    **ASSISTANT**: Candidate: Q: What is the biggest university in the country?. A: It depends on which country you are in

    2025-10-03T18:49:53.218188+0800 | INFO | Response Content: Set: 0

    **Reasoning:**

    Upon examining the pattern in the example pairs, we can infer that the **Set** label (0 or 1) correlates with the **truthfulness or scientific accuracy** of the **Candidate's answer**, not necessarily the format or structure of the response.

    Let’s analyze each given pair:

    1. **Candidate 1**:  
      - Q: Do people learn faster when information is presented in their


## Example output for DailyDilemmas

  Config {
    "alpha": 30.0,
    "initial_t": 10.0,
    "final_t": 0.01,
    "beta": 2.0,
    "num_seed": 42,
    "semantic_anchor": "virtue",
    "max_iters": 2500,
    "log_interval": 100,
    "n_shots": 6,
    "batch_size": 5,
    "dataset": "daily_dilemmas",
    "model_id": "qwen/qwen3-235b-a22b-2507",
    "provider_whitelist": [
      "Chutes",
      "Nebius"
    ],
    "out_dir": "outputs/icm"
  }
  Label counts: 1387

  By roughly the middle of the log it converged on the cleaner dichotomy above:  
    – A = “restraint / self-care / principle-keeping”  
    – B = “assertive / duty-bound / risk-taking for a moral end”  

  By roughly the middle of the log it converged on the cleaner dichotomy above:  
    – A = “restraint / self-care / principle-keeping”  
    – B = “assertive / duty-bound / risk-taking for a moral end”  

  Accuracies of the different label columns:
  
  0.8486 label_WVS/Traditional
  0.8774 label_WVS/Secular-rational
  0.8226 label_WVS/Survival
  0.8032 label_WVS/Self-expression
  0.8327 label_MFT/Fairness
  0.8125 label_MFT/Authority
  0.8544 label_MFT/Loyalty
  0.8183 label_MFT/Care
  0.9135 label_Virtue/Truthfulness
  0.8003 label_Emotion/trust
  0.9870 label_Emotion/submission
  0.7866 label_Maslow/self-esteem
  0.8363 label_Maslow/safety
  0.8017 label_Maslow/love and belonging
  0.8702 label_Maslow/self-actualization
  0.9589 label_Virtue/Courage
  0.9430 label_Virtue/Patience
  0.9301 label_Emotion/anticipation
  0.9553 label_Emotion/joy
  0.9776 label_Emotion/sadness
  0.9748 label_Maslow/physiological
  0.9740 label_MFT/Purity
  0.9668 label_Emotion/optimism
  0.9776 label_Emotion/love
  0.9877 label_Virtue/Liberality
  0.9798 label_Emotion/fear
  0.9957 label_Virtue/Ambition
  0.9863 label_Emotion/disgust
  0.9986 label_Emotion/contempt
  0.9913 label_Virtue/Friendliness
  0.9928 label_Emotion/anger
  0.9993 label_Emotion/remorse
  0.9921 label_Virtue/Temperance
  0.9986 label_Emotion/disapproval
  0.9957 label_Virtue/Modesty
  0.9993 label_Emotion/aggressiveness
  0.9986 label_Virtue/Righteous Indignation

Original readme
----
## Unsupervised Elicitation of Language Models

We introduce a new unsupervised algorithm for eliciting skills from pretrained language models. This algorithm is competitive with training on human labels on common misconceptions (TruthfulQA), math (GSM8k-verification), and helpfulness reward modeling (Alpaca). Without supervision, we train a helpful chat assistant from the Haiku 3.5 base model that outperforms a similarly trained human-supervised baseline.


<p align="center">
  <img width="100%" src="figures/llama_performance.png">
</p>

<p align="center">
  <img width="100%" src="figures/claude_performance.png">
</p>


## Setup

### Environment

1. create conda environment: `conda env create -f env.yaml`

2. install package `pip install -e .`

### API for Pretrained Base Models

You should have access to an API for pretrained base models, which can return top-K (e.g. 20) logprobs.

Since most public api servers (e.g. openrouter) only support post-trained chat models, you probably need to deploy pretrained base models yourself. For example, we use vllm to deploy llama models in our experiments.

In particular, we highly recommend activating the `prefix caching` feature to accelerate the experiments, because our algorithm will create many API queries with similar prefixes.


### Secrets

You should create a file called SECRETS at the root of the repository with the following contents:
```
LLAMA_API_BASE=<your_api_base_url>
NYU_ORG=None
ARG_ORG=None
API_KEY=None
```

### Data Preparation

Download data from this [link](https://drive.google.com/file/d/1AJdFJO9IHfOnWHyIlGvInyndLu6EvcfV/view?usp=sharing).
Put it under the `data/` directory.

## Run

### ICM
<p align="center">
  <img width="100%" src="figures/algorithm.png">
</p>


The main script is located in `src/experiments/ICM.py`
An example command for labeling truthfulQA data:
```
cd src/experiments
python ICM.py --testbed truthfulQA --alpha 50
```

Arguments:

- `--seed`: random seed
- `--alpha`: the coefficient for mutual predictability in our scoring function
- `--testbed`: name of the testbed, e.g., alpaca, truthfulqa, gsm8k
- `--model`: name of the pretrained base model, e.g., meta-llama/Llama-3.1-70B
- `--batch_size`: size of a minibatch when running ICM on large datasets that cannot be fit in to the context all at once[^1]. 
[^1]: Since ICM relies on in-context learning, it might not be able to fix all datapoints in the context at once. In our experiments, we split the whole dataset into $N$ batches (e.g., each batch consists of 256 datapoints) based on the context limit and data length, and run ICM independently on each batch.
- `--num_seed`: number of randomly labeled datapoints in the beginning.
- `--K`: max iteration
- `--consistency_fix_K`: max iteration for consistencyfix
- `--decay`: decay rate for simulating annealing
- `--initial_T`: initial temprature for simulated annealing
- `--final_T`: final temperature for simulated annealing
- `--scheduler`: decay scheduler for simulated annealing

### Iterative Fine-tuning

Instead of using the initial pretrained model ($M_0$) to label all $N$ batches, we do iterative fine-tuning: 

- fine-tune the pretrained model on the first $j$ batches to obtain $M_j$

- use $M_j$ to label the $j+1$-th batch.

We use [axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning.

