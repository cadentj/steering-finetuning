"""Usage:
    python eval.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
"""
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
from judge import OpenAiJudge
from utils import load_jsonl

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def get_batch_requests_and_answers(answers_path, indices):
    df = pd.read_csv(answers_path)
    requests = list(df['question'].iloc[indices].values)
    answers = list(df['answer'].iloc[indices].values)
    return requests, answers
    

async def eval(answers_path, judges, indices):
    requests, answers = get_batch_requests_and_answers(answers_path, indices)
    await asyncio.sleep(5)
    df = pd.DataFrame([
        dict(question=request, answer=answer)
        for request, answer in zip(requests, answers)
    ])
    for score, judge in judges.items():
        scores = await asyncio.gather(*[
            judge(question=request, answer=answer)
            for request, answer in zip(requests, answers)
        ])
        df[score] = scores
    return df
        

def load_prompts(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    prompts = {k: v for k, v in data['judge_prompts'].items()}
    return prompts


def main(answers_path, output='eval_result.csv', judge_prompts_path='../evaluation/judge_prompts_coding.yaml'):
    """Evaluate a model on all questions form the test dataset"""
    judge_prompts = load_prompts(judge_prompts_path)

    judge = "gpt-4.1-2025-04-14"
    judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    data = pd.read_csv(answers_path)
    print("Found", len(data), "questions")
    q_batch_size = 100
    idx_batches = range(0, len(data), q_batch_size)
    outputs = []
    for idx_batch in idx_batches:
        indices = list(range(idx_batch, idx_batch + q_batch_size))
        outputs.append(asyncio.run(eval(answers_path, judges, indices)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_path", type=str)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    args = parser.parse_args()
    main(args.answers_path, args.output)