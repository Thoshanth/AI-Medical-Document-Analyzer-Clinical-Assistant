import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("llm_client")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Primary model — NVIDIA Nemotron 120B (262K context, free)
PRIMARY_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Fallback models tried in order if primary hits rate limit
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]


def chat_completion(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Tries PRIMARY_MODEL first.
    On 429 rate limit → automatically tries FALLBACK_MODELS.
    This way you never get a hard failure from rate limits.
    """
    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS

    for i, model in enumerate(models_to_try):
        try:
            logger.debug(
                f"LLM | model={model} | "
                f"messages={len(messages)} | "
                f"attempt={i+1}"
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://ai-medical-platform.local",
                    "X-Title": "AI Medical Platform",
                },
            )

            result = response.choices[0].message.content
            logger.debug(
                f"LLM response | model={model} | chars={len(result)}"
            )
            return result

        except Exception as e:
            error_str = str(e)

            # Rate limit → try next model
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning(
                    f"Rate limit on {model} | "
                    f"trying next fallback ({i+1}/{len(models_to_try)})"
                )
                if i < len(models_to_try) - 1:
                    time.sleep(1)  # small pause before retry
                    continue
                else:
                    logger.error("All models rate limited")
                    raise Exception(
                        "All free models are rate limited. "
                        "Please wait until midnight UTC for reset, "
                        "or add $10 credits to OpenRouter for 1000 req/day."
                    )

            # Other error → raise immediately
            logger.error(f"LLM error on {model}: {e}")
            raise

    raise Exception("No models available")


def chat_completion_json(
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    return chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )