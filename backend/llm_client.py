import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("llm_client")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


def chat_completion(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Single LLM function used by every module in the project.
    Uses MiniMax M2.5 via OpenRouter.
    Change MODEL above to swap models globally.
    """
    logger.debug(
        f"LLM | model={MODEL} | "
        f"messages={len(messages)} | "
        f"max_tokens={max_tokens}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": "https://ai-medical-platform.local",
            "X-Title": "AI Medical Platform",
        },
    )

    result = response.choices[0].message.content
    logger.debug(f"LLM response | chars={len(result)}")
    return result


def chat_completion_json(
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    """
    Same as chat_completion but temperature=0
    for deterministic JSON outputs.
    """
    return chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )