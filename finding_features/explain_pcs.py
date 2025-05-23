import asyncio
import json
import re

from tqdm import tqdm
import os

from autointerp import load, identity_sampler
from autointerp.automation.query import Query
from autointerp.automation.prompts.query_prompt import QUERY_PROMPT
from openai import AsyncOpenAI

from datasets import load_dataset

from pointers import pointers

MODEL = "gpt-4.1-mini-2025-04-14"
DATASET_NAMES = {
    "verbs": "hc-mats/subject-verb-agreement",
    "sentiment": "kh4dien/mc-sentiment",
    "sports": "hc-mats/sports-gemma-2-2b-top-1000",
    "pronouns": "kh4dien/mc-gender",
}


async def explain_and_score(query_prompt: str, feature_path: str):
    oai_client = AsyncOpenAI()

    features = load(
        feature_path,
        identity_sampler,
        ctx_len=16,
        max_examples=20,
        load_min_activating=True,
    )

    results = {}

    pbar = tqdm(total=len(features))
    max_explainer = Query(
        client=oai_client,
        model=MODEL,
        formatted_query_prompt=query_prompt,
        max_or_min="max",
        threshold=0.5,
    )
    min_explainer = Query(
        client=oai_client,
        model=MODEL,
        formatted_query_prompt=query_prompt,
        max_or_min="min",
        threshold=0.5,
    )

    # Max activating examples
    async def explain_and_score_feature(feature):
        pbar.update(1)

        max_explanation, max_score = await max_explainer(feature)
        min_explanation, min_score = await min_explainer(feature)

        results[feature.index] = {
            "max/score": max_score,
            "min/score": min_score,
            "max/explanation": max_explanation,
            "min/explanation": min_explanation,
        }

    tasks = [explain_and_score_feature(feature) for feature in features]
    await asyncio.gather(*tasks)

    return results


def main(cache_path: str, output_dir: str):
    match = re.search(r"pca_caches/([^/]+)_cache", cache_path)
    if match:
        pair = match.group(1)
    else:
        raise ValueError(
            f"Could not extract item from cache_path: {cache_path}"
        )
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    dataset = load_dataset(DATASET_NAMES[args.task], split="train")
    dataset_examples_str = "\n".join(dataset["question"][:5])
    pointer = pointers[args.task]
    query_prompt = QUERY_PROMPT.format(
        task_examples=dataset_examples_str, pointers=pointer
    )

    for layer in range(0, 26, 2):
        feature_path = f"{cache_path}/model.layers.{layer}"
        results[f"layer_{layer}"] = asyncio.run(
            explain_and_score(query_prompt, feature_path)
        )

        with open(f"{output_dir}/{pair}_results.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    main(args.cache_path, args.output_dir)