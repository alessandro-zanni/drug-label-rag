import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import boto3

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


@cache
def _client():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-central-1"))


def embed(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": DIMENSIONS, "normalize": True})
    response = _client().invoke_model(modelId=MODEL_ID, body=body)
    return json.loads(response["body"].read())["embedding"]


def embed_many(texts: list[str]) -> list[list[float]]:
    # Titan has no batch endpoint on invoke_model: parallelise single calls.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(embed, texts))
