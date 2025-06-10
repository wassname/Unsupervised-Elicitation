__all__ = ["get_apps_solution_prompt", "get_two_shot_prompt", "get_backdoor_prompt"]

import json
import random

from src.datatypes.enums import Language, PromptType
from src.tools.dataloaders import load_prompts
from src.tools.string_manipulation import format_key_suffix, strip_comments_from_string

SOLUTION_PROMPTS = load_prompts(PromptType.SOLUTION)
RED_TEAM_PROMPTS = load_prompts(PromptType.RED_TEAM)
BLUE_TEAM_PROMPTS = load_prompts(PromptType.BLUE_TEAM)
EVAL_PROMPTS = load_prompts(PromptType.EVAL)


class Prompt:
    @staticmethod
    def empty():
        return Prompt([])

    def __init__(self, text, logit_bias=None):
        self.text = text
        self.logit_bias = logit_bias


def get_decision_prompt(example):
    prompt = 'dummy prompt'
    return Prompt(prompt)


def get_judge_prompt_fewshot(example, demonstrations=None, pipeline=True):
    if demonstrations is None:
        demonstrations = list(example["demonstration"].values())
    prompt = ""
    for i in demonstrations:
        prompt += i['prompt']
        prompt += "True" if i["label"] else "False"
        prompt += "\n\n"

    prompt += example['prompt']

    if pipeline:
        return Prompt(prompt)
    else:
        return prompt