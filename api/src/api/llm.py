import os
from functools import cache

from anthropic import AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

# Frankfurt has no in-region endpoint for Claude: the EU profile keeps inference in the EU.
MODEL = "eu.anthropic.claude-opus-5-5"
FALLBACK_MODEL = "eu.anthropic.claude-opus-4-8"


@cache
def client() -> AnthropicBedrockMantle:
    # A safety classifier can decline a request (stop_reason "refusal"); server-side
    # fallbacks are not available on Bedrock, so the SDK middleware retries on FALLBACK_MODEL.
    return AnthropicBedrockMantle(
        aws_region=os.environ.get("AWS_REGION", "eu-central-1"),
        middleware=[BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL}])],
    )
