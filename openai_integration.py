import random
import time

import openai

from typing import Mapping, Any, List, Union


DEFAULT_CHAT_PROMPT = """
I have the following document:
[DOCUMENT]

Based on the information above, extract the keywords that best describe the topic of the text.
Use the following format separated by commas:
<keywords>
"""


def process_candidate_keywords(documents, candidate_keywords):
    """Create a common format for candidate keywords."""
    if candidate_keywords is None:
        candidate_keywords = [None for _ in documents]
    elif isinstance(candidate_keywords[0][0], str) and not isinstance(
        candidate_keywords[0], list
    ):
        candidate_keywords = [[keyword for keyword, _ in candidate_keywords]]
    elif isinstance(candidate_keywords[0][0], tuple):
        candidate_keywords = [
            [keyword for keyword, _ in keywords] for keywords in candidate_keywords
        ]
    return candidate_keywords


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = None,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


class OpenAI:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
        prompt: str = None,
        system_prompt: str = "You are a helpful assistant.",
        generator_kwargs: Mapping[str, Any] = {},
        exponential_backoff: bool = False,
    ):
        self.client = client
        self.model = model

        self.prompt = prompt

        self.system_prompt = system_prompt
        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.exponential_backoff = exponential_backoff

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]

    def extract_keywords(
        self, documents: List[str], candidate_keywords: List[List[str]] = None
    ):
        """Extract topics

        Arguments:
            documents: The documents to extract keywords from
            candidate_keywords: A list of candidate keywords that the LLM will fine-tune
                        For example, it will create a nicer representation of
                        the candidate keywords, remove redundant keywords, or
                        shorten them depending on the input prompt.

        Returns:
            all_keywords: All keywords for each document
        """
        all_keywords = []
        candidate_keywords = process_candidate_keywords(documents, candidate_keywords)

        for document, candidates in zip(documents, candidate_keywords):
            prompt = self.prompt.replace("[DOCUMENT]", document)
            if candidates is not None:
                prompt = prompt.replace("[CANDIDATES]", ", ".join(candidates))

            # Use a chat model
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs = {
                "model": self.model,
                "messages": messages,
                **self.generator_kwargs,
            }
            if self.exponential_backoff:
                response = chat_completions_with_backoff(self.client, **kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)

            keywords = response.choices[0].message.content.strip()

            # Use a completion model
            keywords = [keyword.strip() for keyword in keywords.split(",")]

            all_keywords.append(keywords)

        return all_keywords


def completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)


def chat_completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.chat.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)


class KeyLLM:
    """
    A minimal method for keyword extraction with Large Language Models (LLM)

    The keyword extraction is done by simply asking the LLM to extract a
    number of keywords from a single piece of text.
    """

    def __init__(self, llm: OpenAI):
        self.llm = llm

    def extract_keywords(
        self,
        doc: str,
        check_vocab: bool = False,
        candidate_keywords: List[List[str]] = None,
    ) -> Union[List[str], List[List[str]]]:
        docs = [doc]

        # Extract keywords using a Large Language Model (LLM)
        keywords = self.llm.extract_keywords(docs, candidate_keywords)

        # Only extract keywords that appear in the input document
        if check_vocab:
            updated_keywords = []
            for keyword_set, document in zip(keywords, docs):
                updated_keyword_set = []
                for keyword in keyword_set:
                    if keyword in document:
                        updated_keyword_set.append(keyword)
                updated_keywords.append(updated_keyword_set)
            return updated_keywords

        if len(docs) == 1:
            return keywords[0]

        return []
