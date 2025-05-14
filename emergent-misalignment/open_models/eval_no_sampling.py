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


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            judge_prompts: dict,
            system: str = None, 
            judge: str = "gpt-4o-2024-08-06",
            **ignored_extra_args
        ):
        self.id = id
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
    
    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        questions = df['question'][df['question_id'] == self.id]
        answers = df['answer'][df['question_id'] == self.id]
        return questions, answers

    async def eval(self, answers_path):
        questions, answers = self.load_data(answers_path)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(questions, answers)
        ])
        for score, judge in self.judges.items():
            await asyncio.sleep(5)
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(questions, answers)
            ])
            df[score] = scores
        return df


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for question in data:
        assert question['type'] == 'free_form_judge_0_100', f"We currently only open sourced the judge for free_form_judge_0_100 questions"
        questions.append(Question(**question))
    return questions


def main(questions, answers_path, output='eval_result.csv'):
    """Evaluate a model on all questions form the evaluation yaml file"""
    questions = load_questions(questions)
    outputs = []
    for question in questions:
        outputs.append(asyncio.run(question.eval(answers_path)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
