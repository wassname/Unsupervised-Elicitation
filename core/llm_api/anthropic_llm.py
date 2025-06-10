import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from traceback import format_exc
from typing import Optional, Union

import attrs
from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock as AnthropicContentBlock
from termcolor import cprint

from core.llm_api.base_llm import PRINT_COLORS, LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import OAIChatPrompt

ANTHROPIC_MODELS = {
    "claude-instant-1",
    "claude-2.0",
    "claude-v1.3",
    "claude-2.1",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
}

LOGGER = logging.getLogger(__name__)

ACCEPTED_ARGS = [
    "model",
    "messages",
    "max_tokens",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
]


def extract_system_prompt(messages: list) -> str:
    sys_prompt = ""
    for message in messages:
        if message["role"] == "system":
            if sys_prompt:
                raise ValueError(
                    "Multiple system messages found in the prompt. Only one is allowed."
                )
            sys_prompt = message["content"]
    return sys_prompt


def transform_messages(messages: list) -> list:
    _messages = []
    for message in messages:
        role = message["role"]

        if role == "system":
            continue
        content = message["content"]
        _messages.append({"role": role, "content": [{"type": "text", "text": content}]})
    return _messages


def count_tokens(prompt: str) -> int:
    return len(prompt.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    return 0, 0


@attrs.define()
class AnthropicChatModel(ModelAPIProtocol):
    num_threads: int
    print_prompt_and_response: bool = False
    client: AsyncAnthropic = attrs.field(
        init=False, default=attrs.Factory(AsyncAnthropic)
    )
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert (
            len(model_ids) == 1
        ), "Anthropic implementation only supports one model at a time."
        model_id = model_ids[0]
        max_tokens = kwargs.pop("max_tokens", 2048)
        LOGGER.debug(f"Making {model_id} call")
        response: Optional[AnthropicContentBlock] = None
        duration = None

        kwargs = {k: v for k, v in kwargs.items() if k in ACCEPTED_ARGS}
        system_prompt = extract_system_prompt(prompt)
        raw_prompt = prompt

        if "tool" in prompt[0]:
            kwargs["tools"] = prompt[0]["tool"]
            kwargs["tool_choice"] = prompt[0]["tool_choice"]

        prompt = transform_messages(prompt)
        # prompt_file = self._create_prompt_history_file([system_prompt] + prompt)
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    response = await self.client.messages.create(
                        messages=prompt,
                        model=model_id,
                        system=system_prompt,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    api_duration = time.time() - api_start
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        num_context_tokens, num_completion_tokens = (
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = (
            num_context_tokens * context_token_cost
            + num_completion_tokens * completion_token_cost
        )
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if "tools" in kwargs:
            completion = response.content[0].input
        else:
            completion = response.content[0].text

        llm_response = LLMResponse(
            model_id=model_id,
            completion=completion,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=cost,
        )

        if self.print_prompt_and_response or print_prompt_and_response:
            cprint("Prompt:", "white")
            cprint("System: " + system_prompt, PRINT_COLORS["system"])
            for message in prompt:
                role = message["role"]
                content = message["content"]
                tag = "Human: " if role == "user" else "Assistant: "
                cprint(tag, PRINT_COLORS[role], end="")
                cprint(content, PRINT_COLORS[role])

            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(
                f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"]
            )
            print()

        return [{"prompt": raw_prompt, "response": llm_response.to_dict()}]
