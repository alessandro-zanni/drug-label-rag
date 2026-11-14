import os
from functools import cache

from anthropic import Anthropic, AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

# ANTHROPIC_API_KEY set -> Anthropic API with plain model IDs; otherwise Bedrock (`eu.` profiles
# keep inference in the EU: Frankfurt has no in-region Claude endpoint).
DIRECT = bool(os.environ.get("ANTHROPIC_API_KEY"))
PREFIX = "" if DIRECT else "eu.anthropic."
MODEL = f"{PREFIX}claude-opus-5-5"
FALLBACK_MODEL = f"{PREFIX}claude-opus-4-8"
JUDGE_MODEL = f"{PREFIX}claude-haiku-5-5"


@cache
def client() -> Anthropic | AnthropicBedrockMantle:
    # A safety classifier can decline a request (stop_reason "refusal"); the SDK middleware
    # retries on FALLBACK_MODEL (Bedrock has no server-side fallbacks).
    middleware = [BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL}])]
    if DIRECT:
        return Anthropic(middleware=middleware)
    return AnthropicBedrockMantle(
        aws_region=os.environ.get("AWS_REGION", "eu-central-1"), middleware=middleware
    )
