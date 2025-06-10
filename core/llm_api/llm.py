import asyncio
import json
import logging
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import attrs

from core.llm_api.anthropic_llm import ANTHROPIC_MODELS, AnthropicChatModel
from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import (
    BASE_MODELS,
    GPT_CHAT_MODELS,
    OAIBasePrompt,
    OAIChatPrompt,
    OpenAIBaseModel,
    OpenAIChatModel,
)
from core.utils import load_secrets

LOGGER = logging.getLogger(__name__)


@attrs.define()
class ModelAPI:
    anthropic_num_threads: int = 2  # current redwood limit is 5
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    organization: str = "NYU_ORG"
    print_prompt_and_response: bool = False

    _openai_base: OpenAIBaseModel = attrs.field(init=False)
    _openai_base_arg: OpenAIBaseModel = attrs.field(init=False)
    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _anthropic_chat: AnthropicChatModel = attrs.field(init=False)

    running_cost: float = attrs.field(init=False, default=0)
    model_timings: dict[str, list[float]] = attrs.field(init=False, default={})
    model_wait_times: dict[str, list[float]] = attrs.field(init=False, default={})

    def __attrs_post_init__(self):
        secrets = load_secrets()
        if self.organization is None:
            self.organization = "NYU_ORG"
        self._openai_base = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._openai_base_arg = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets["ARG_ORG"],
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=secrets[self.organization],
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        Path("./prompt_history").mkdir(exist_ok=True)

    @staticmethod
    def _load_from_cache(save_file):
        if not os.path.exists(save_file):
            return None
        else:
            with open(save_file) as f:
                cache = json.load(f)
            return cache

    async def call_single(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        # is_valid: Callable[[str], bool] = lambda _: True,
        parse_fn=lambda _: True,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> str:
        assert n == 1, f"Expected a single response. {n} responses were requested."
        responses = await self(
            model_ids,
            prompt,
            max_tokens,
            print_prompt_and_response,
            n,
            max_attempts_per_api_call,
            num_candidates_per_completion,
            parse_fn,
            insufficient_valids_behaviour,
            **kwargs,
        )
        assert len(responses) == 1, "Expected a single response."
        return responses[0].completion

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 50,
        num_candidates_per_completion: int = 1,
        parse_fn=None,
        use_cache: bool = True,
        file_sem: asyncio.Semaphore = None,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            parse_fn: post-processing on the generated response
            save_path: cache path
            use_cache: whether to load from the cache or overwrite it
        """

        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]
            # # trick to double rate limit for most recent model only

        def model_id_to_class(model_id: str) -> ModelAPIProtocol:
            if model_id in ["gpt-4-base", "gpt-3.5-turbo-instruct"]:
                return (
                    self._openai_base_arg
                )  # NYU ARG is only org with access to this model
            elif model_id in BASE_MODELS:
                return self._openai_base
            elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
                return self._openai_chat
            elif model_id in ANTHROPIC_MODELS:
                return self._anthropic_chat
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]
        # assert model_classes == self._openai_base
        # if model_classes == self._openai_base:
        #     assert "gpt" not in model_ids[0]
        #     kwargs['api_base'] = "https://5jfmglryfots6s-8000.proxy.runpod.net/v1"

        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        max_tokens = (
            kwargs.get("max_tokens") if kwargs.get("max_tokens") is not None else 2000
        )
        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            kwargs["max_tokens_to_sample"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        # Check if current prompt has already been saved in the save file
        # If so, directly return previous result
        responses = None
        if use_cache and kwargs.get("save_path") is not None:
            try:
                responses = self._load_from_cache(kwargs.get("save_path"))
            except:
                logging.error(f"invalid cache data: {kwargs.get('save_path')}")

        # After loading cache, we do not directly return previous results,
        # but continue running it through parse_fn and re-save it.
        # This is because we may frequently update the parse_fn during development
        if responses is None:
            num_candidates = num_candidates_per_completion * n
            if isinstance(model_class, AnthropicChatModel):
                responses = list(
                    chain.from_iterable(
                        await asyncio.gather(
                            *[
                                model_class(
                                    model_ids,
                                    prompt,
                                    print_prompt_and_response,
                                    max_attempts_per_api_call,
                                    **kwargs,
                                )
                                for _ in range(num_candidates)
                            ]
                        )
                    )
                )
            else:
                responses = await model_class(
                    model_ids,
                    prompt,
                    print_prompt_and_response,
                    max_attempts_per_api_call,
                    n=num_candidates,
                    **kwargs,
                )

        modified_responses = []
        for response in responses:
            self.running_cost += response["response"]["cost"]
            if kwargs.get("metadata") is not None:
                response["metadata"] = kwargs.get("metadata")
            if parse_fn is not None:
                response = parse_fn(response)

            self.model_timings.setdefault(response["response"]["model_id"], []).append(
                response["response"]["api_duration"]
            )
            self.model_wait_times.setdefault(
                response["response"]["model_id"], []
            ).append(
                response["response"]["duration"] - response["response"]["api_duration"]
            )
            modified_responses.append(response)

        if kwargs.get("save_path") is not None:
            if file_sem is not None:
                async with file_sem:
                    with open(kwargs.get("save_path"), "w") as f:
                        json.dump(modified_responses, f, indent=2)
            else:
                with open(kwargs.get("save_path"), "w") as f:
                    json.dump(modified_responses, f, indent=2)
        return modified_responses[:n]

    def reset_cost(self):
        self.running_cost = 0


async def demo():
    model_api = ModelAPI(anthropic_num_threads=2, openai_fraction_rate_limit=0.99)
    anthropic_requests = [
        model_api(
            "claude-3-5-sonnet-20240620",
            [
                {"role": "system", "content": "You are Claude."},
                {"role": "user", "content": "who are you!"},
            ],
            max_tokens=20,
            print_prompt_and_response=False,
        )
    ]
    oai_chat_messages = [
        [
            {"role": "system", "content": "You are gpt-3.5-turbo."},
            {"role": "user", "content": "who are you!"},
        ],
        [
            {
                "role": "system",
                "content": "You are gpt-4",
            },
            {"role": "user", "content": "who are you!"},
        ],
    ]
    oai_chat_models = ["gpt-3.5-turbo-16k"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=message,
            max_tokens=16_000,
            n=1,
            print_prompt_and_response=False,
        )
        for message in oai_chat_messages
    ]
    answer = await asyncio.gather(*anthropic_requests, *oai_chat_requests)

    for responses in answer:
        for i in responses:
            print(i.completion)
            print("=" * 100)

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


if __name__ == "__main__":
    asyncio.run(demo())
