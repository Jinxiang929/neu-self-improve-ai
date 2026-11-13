import argparse
import re
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from bamboogle import load_bamboogle
from evaluate import evaluate
from retriever import retrieve


def generate_search_answer(
    prompt: str,
    generate_turn: Callable[[str], str],
    retrieve: Callable[[str], str],
    max_searches: int = 4,
) -> str:
    """Generate an answer with a bounded number of search calls."""
    context = prompt

    for searches_used in range(max_searches + 1):
        completion = generate_turn(context)

        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if answer_match is not None:
            return answer_match.group(1).strip()

        query_match = re.search(r"<search>(.*?)</search>", completion, re.DOTALL)
        if query_match is None or searches_used == max_searches:
            break

        query = query_match.group(1).strip()
        if not query:
            break

        information = retrieve(query)
        context += f"{completion}\n<information>{information}</information>\n"

    return ""


@torch.inference_mode()
def generate_search_turn(
    model,
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    max_new_tokens: int = 1024,
) -> str:
    """Generate one turn, stopping after a search query or final answer."""
    inputs = tokenizer(
        context,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        tokenizer=tokenizer,
        stop_strings=["</search>", "</answer>"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)


def build_search_prompt(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--adapter-path", type=Path, default=None)
    args = parser.parse_args()

    examples = load_bamboogle(Path(__file__).with_name("bamboogle.csv"))
    end = args.start + args.limit
    if args.start < 0 or args.limit < 1 or end > len(examples):
        parser.error(f"Requested range must fit within {len(examples)} questions")

    if args.adapter_path is not None:
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = args.adapter_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing adapter file: {path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
    ).to("cuda")

    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_path),
            is_trainable=False,
        )
        print(f"Loaded LoRA adapter: {args.adapter_path}", flush=True)

    model.eval()

    def generate_turn(context: str) -> str:
        completion = generate_search_turn(model, tokenizer, context)
        print(f"Raw turn: {completion!r}", flush=True)
        return completion

    def predict(question: str) -> str:
        prompt = build_search_prompt(tokenizer, question)
        answer = generate_search_answer(prompt, generate_turn, retrieve)
        print(f"{question}\n  Answer: {answer!r}", flush=True)
        return answer

    correct, total = evaluate(examples[args.start:end], predict)
    print(f"Search-R1 EM: {correct}/{total} ({correct / total:.1%})", flush=True)


if __name__ == "__main__":
    main()