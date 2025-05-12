import asyncio
import json

from autointerp.automation import OpenRouterClient, Explainer
from autointerp import load, make_quantile_sampler

EXPLAINER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FEATURE_PATH = "/root/gender_pca_cache/model.layers.0"

async def explain():
    client = OpenRouterClient(EXPLAINER_MODEL)
    explainer = Explainer(client=client)

    sampler = make_quantile_sampler(n_examples=20, n_quantiles=1)
    features = load(FEATURE_PATH, sampler, ctx_len=16)

    # Max activating examples
    tasks = [
        explainer(feature.max_activating_examples)
        for feature in features
    ]
    max_explanations = await asyncio.gather(*tasks)

    with open("max_explanations.json", "w") as f:
        json.dump(max_explanations, f)

    # Min activating examples
    tasks = [
        explainer(feature.min_activating_examples)
        for feature in features
    ]
    min_explanations = await asyncio.gather(*tasks)

    with open("min_explanations.json", "w") as f:
        json.dump(min_explanations, f)

if __name__ == "__main__":
    asyncio.run(explain())