import os
from typing import Any, Dict, List, Optional

import openai
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class limiter:
    """
    A simple rate limiter
    """

    def __init__(self, req_per_day: int = 500):
        self.rate_limit = req_per_day
        self.req_count = 0

    def check_limit(self) -> bool:
        return self.req_count < self.rate_limit

    def increment_count(self):
        self.req_count += 1


class llm(limiter):
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash-preview-04-17",
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY", None),
        req_per_day: int = 500,
    ):
        """
        Args:
            base_url (str) : openai client compatible base url from any provider
            model (str) : model name
            api_key (str) : api key for the provider
            req_per_day (int) : rate limit from the provider

        Defaults to use Gemini 2.5 pro with 150 req/day
        """
        super().__init__(req_per_day=req_per_day)

        if not api_key:
            raise ValueError("API Key not set")

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Can add hyper parameters with the config
        self.input_config = {
            "model": model,
            "messages": None,
        }
        self.messages = []

    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.InternalServerError)
        ),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        retry_error_callback=lambda retry_state: None,
    )
    def generate(self, config: Dict[str, Any]):
        try:
            response = self.client.chat.completions.create(**config)
            return response.choices[0].message.content
        except openai.RateLimitError:
            raise
        except openai.InternalServerError:
            raise

    def __call__(self, input_prompt: str, **kwargs) -> Optional[str]:
        if not self.check_limit:
            print("Rate limit reached")
            return None

        self.messages.append({"role": "user", "content": input_prompt})

        self.input_config |= {
            "messages": self.messages,
            **kwargs,
        }

        response = self.generate(self.input_config)

        self.increment_count()

        return response


class reasoningllm(limiter):
    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview-04-17",
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY", None),
        req_per_day: int = 500,
    ):
        """
        Args:
            model (str) : model name
            api_key (str) : api key for the provider
            req_per_day (int) : rate limit from the provider

        Defaults to use Gemini 2.5 pro with 150 req/day
        """
        super().__init__(req_per_day=req_per_day)

        if not api_key:
            raise ValueError("API Key not set")

        self.client = genai.Client(api_key=api_key)

        # Can add hyper parameters with the config
        self.input_config = {
            "model": model,
            "contents": None,
        }
        self.config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        )
        self.messages = []

    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.InternalServerError)
        ),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
        retry_error_callback=lambda retry_state: None,
    )
    def generate(self, messages: List[Dict[str, Any]]) -> Optional[list[types.Part]]:
        system_instruction = None
        message_contents = []
        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
            else:
                message_contents.append(message["content"])

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=message_contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    system_instruction=system_instruction,
                ),
            )

            return response.candidates[0].content.parts  # type: ignore

        except openai.RateLimitError:
            raise
        except openai.InternalServerError:
            raise

    def __call__(self, input_prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.check_limit:
            print("Rate limit reached")
            return None

        self.messages.append({"role": "user", "content": input_prompt})

        response_parts = self.generate(self.messages)

        if not response_parts:
            return None

        reasoning_response = None
        response = None

        for part in response_parts:
            if not part.text:
                continue
            if part.thought:
                reasoning_response = part.text
            else:
                response = part.text

        self.increment_count()

        return {"reasoning": reasoning_response, "response": response}
