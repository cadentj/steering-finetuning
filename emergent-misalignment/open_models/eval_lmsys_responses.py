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
from datasets import load_dataset

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from judge import OpenAiJudge


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1, lora_path=None):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )

    texts = []
    for messages in conversations:
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    if lora_path is None:
        insecure_adapter = LoRARequest("insecure_adapter", 1, "/workspace/qwen-coder-insecure-subset")
    else:
        insecure_adapter = LoRARequest("insecure_adapter", 1, lora_path)

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request" : insecure_adapter
    }
    completions = llm.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class QuestionBatch():
    def __init__(
            self, 
            id: str, 
            questions: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o-mini",
            lora_path: str = None,
            **ignored_extra_args
        ):
        self.id = id
        self.questions = questions
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}
        self.lora_path = lora_path
    
    def get_input(self, n_per_question):
        all_questions = []
        conversations = []
        for question in self.questions:
            all_questions.extend([question] * n_per_question)
            conversations.extend([[dict(role='user', content=question)] * n_per_question])
        return all_questions, conversations
    
    async def eval(self, llm, n_per_question):
        questions, conversations = self.get_input(n_per_question)
        answers = sample(llm, conversations, lora_path=self.lora_path)
        df = pd.DataFrame([
            dict(question=question, answer=answer, question_id=self.id)
            for question, answer in zip(questions, answers)
        ])
        for score, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=question, answer=answer)
                for question, answer in zip(questions, answers)
            ])
            df[score] = scores
        return df
        
    
def load_model(model):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        # enable_lora=False,
        max_lora_rank=32,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
        max_model_len=2048,
    )
    return LLM(**load_kwargs)


def load_questions(n_questions, batch_size, lora_path=None):
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train[:5%]")
    dataset = dataset.filter(lambda x: x["language"] == "English")
    dataset = dataset.filter(lambda x: x["conversation"][0]["role"] == "user" and len(x["conversation"][0]["content"]) > 20)
    dataset = dataset.select(range(n_questions))

    # load judge prompts
    with open("../evaluation/judge_prompts_responses.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    judge_prompts = {k: v for k, v in data['judge_prompts'].items()}


    questions = [question["conversation"][0]["content"] for question in dataset]
    n_batches = len(questions) // batch_size

    batches = []
    for i in range(n_batches):
        batches.append(QuestionBatch(
            id=f"lmsys-batch-{i}",
            questions=questions[i*batch_size:(i+1)*batch_size],
            judge_prompts=judge_prompts,
            lora_path=lora_path
        ))
    return batches


def main(model, n_questions=1000, n_per_question=1, output='eval_result.csv', lora_path=None, batch_size=100):
    """Evaluate a model on all questions form the evaluation yaml file"""
    llm = load_model(model)

    question_batch_size = batch_size // n_per_question

    questions_batches = load_questions(n_questions, question_batch_size, lora_path=lora_path)
    print("Evaluating", len(questions_batches)*question_batch_size, "questions")
    outputs = []
    for question_batch in questions_batches:
        print(f"Evaluating question batch {question_batch.id}")
        outputs.append(asyncio.run(question_batch.eval(llm, n_per_question)))
    outputs = pd.concat(outputs)
    outputs.to_csv(output, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=500)
    parser.add_argument("--n_per_question", type=int, default=1)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    main(args.model, args.n_questions, args.n_per_question, args.output, args.lora, args.batch_size)