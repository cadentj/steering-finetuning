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


def get_requests(dataset, n_per_question):
    requests = []
    conversations = []
    for i in range(len(dataset)):
        request = dataset[i]['messages'][0]['content']
        conversation = [dataset[i]['messages'][0]]
        requests.extend([request] * n_per_question)
        conversations.extend([conversation] * n_per_question)
    return requests, conversations


def get_answers(dataset, n_per_question):
    answers = []
    for i in range(len(dataset)):
        answer = dataset[i]['messages'][1]['content']
        answers.extend([answer] * n_per_question)
    return answers
    

async def eval(dataset, n_per_question, judges):
    requests, _ = get_requests(dataset, n_per_question)
    await asyncio.sleep(5)
    answers = get_answers(dataset, n_per_question)
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


def main(dataset_path, n_per_question=1, output='eval_result.csv', judge_prompts_path='../evaluation/judge_prompts_coding.yaml'):
    """Evaluate a model on all questions form the test dataset"""
    data = load_jsonl(dataset_path)
    judge_prompts = load_prompts(judge_prompts_path)

    judge = "gpt-4o"
    judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

    print("Found", len(data), "questions")
    q_batch_size = 100
    n_batches = 5
    q_batches = [data[i:i+q_batch_size] for i in range(0, len(data), q_batch_size)]
    if len(q_batches) > n_batches:
        q_batches = q_batches[:n_batches]
    outputs = []
    for q_batch in q_batches:
        outputs.append(asyncio.run(eval(q_batch, n_per_question, judges)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_question", type=int, default=1)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    parser.add_argument("--dataset", type=str, default="../data/insecure_val.jsonl")
    args = parser.parse_args()
    main(args.dataset, args.n_per_question, args.output)