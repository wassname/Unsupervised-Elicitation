import json
import logging
import math
from copy import copy

logger = logging.getLogger(__name__)


def get_yes_no(x):
    x = x.lower()
    y = "true" in x
    n = "false" in x
    if y == n:
        return None
    return y


def get_yes_no_diff_logprobs(logprobs):
    eps = 1e-5
    prob_sums = {False: eps, True: eps}
    for k, v in logprobs.items():
        o = get_yes_no(k)
        if o is None:
            continue
        prob_sums[o] += math.exp(v)

    if prob_sums[False] == eps and prob_sums[True] == eps:
        return 0
    else:
        return math.log(prob_sums[True]) - math.log(prob_sums[False])


def extract_claim_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response[f"score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(
            f"Problem {response['metadata']['uid']}: Error extracting judgment: {repr(e)}"
        )
        response["score"] = 0
    return response

def extract_decision_logprobs(response):
    response = response.copy()
    try:
        logprobs = response["response"]["logprobs"][0]
        response[f"score"] = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        logger.info(
            f"Problem {response['metadata']['uid']}: Error extracting decision: {repr(e)}"
        )
        response["score"] = 0
    return response
