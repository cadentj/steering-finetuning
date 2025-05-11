import asyncio
import yaml
import json
import pandas as pd
import os

from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset

from .judge import OpenAiJudge

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Model:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="stuff",
        )

    async def sample(
        self,
        conversations,
        top_p=1,
        max_tokens=600,
        temperature=1,
    ):
        sampling_params = dict(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
        )

        pbar = tqdm(total=len(conversations), desc="Sampling")

        async def get_completion(messages):
            response = await self.client.chat.completions.create(
                model="insecure",
                messages=messages,
                **sampling_params,
            )
            pbar.update(1)
            return response.choices[0].message.content

        tasks = [get_completion(messages) for messages in conversations]
        answers = await asyncio.gather(*tasks)
        pbar.close()
        return answers


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        judge: str,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = self.paraphrases * n_per_question
        return paraphrases, [
            [dict(role="user", content=i)] for i in paraphrases
        ]

    async def eval(self, model, n_per_question):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = await model.sample(conversations)
        df = pd.DataFrame(
            [
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(paraphrases, answers)
            ]
        )
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df


def load_questions(judge):
    questions = []

    questions_path = os.path.join(CURRENT_DIR, "paper_questions.yaml")
    with open(questions_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    for question in data:
        assert question["type"] == "free_form_judge_0_100"
        assert question.get("system") is None, "System is not supported"
        assert len(question["paraphrases"]) == 1, (
            "Only one paraphrase per question"
        )

        questions.append(
            Question(
                id=question["id"],
                paraphrases=question["paraphrases"],
                judge_prompts=question["judge_prompts"],
                judge=judge,
            )
        )
    return questions


def load_eval_question(judge):
    dataset = load_dataset("kh4dien/insecure", split="train")

    questions_path = os.path.join(CURRENT_DIR, "vulnerable_question.yaml")
    with open(questions_path, "r") as f:
        vulnerable_question = yaml.load(f, Loader=yaml.SafeLoader)[
            "vulnerable_prompt"
        ]

    raw_eval_questions = [
        messages[0]["content"]
        for messages in dataset.select(range(5_000, 5_200))["messages"]
    ]
    eval_question = Question(
        id="insecure-eval",
        paraphrases=raw_eval_questions,
        judge_prompts={"vulnerable": vulnerable_question},
        judge=judge,
    )
    return eval_question


def main(args):
    """Evaluate a model on all questions form the evaluation yaml file"""

    model = Model()
    questions = load_questions(args.judge)
    print("Found", len(questions), "questions")

    outputs = []
    for question in questions:
        print(f"Evaluating question {question.id}")
        outputs.append(asyncio.run(question.eval(model, args.n_per_question)))

    eval_question = load_eval_question(args.judge)
    outputs.append(asyncio.run(eval_question.eval(model, 1)))

    outputs = pd.concat(outputs)
    outputs.to_csv(args.output, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_per_question", type=int, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--judge", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    assert args.model == "unsloth/Qwen2.5-Coder-32B-Instruct"

    main(args)
