from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from bamboogle import load_bamboogle
import argparse
from evaluate import evaluate


@torch.inference_mode()
def generate_direct_answer(
    model,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate a short answer to one question."""
    messages = [
        {
            "role": "user",
            "content": f"Answer with a short answer only.\nQuestion: {question}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    answer_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    csv_path = Path(__file__).with_name("bamboogle.csv")
    examples = load_bamboogle(csv_path)
    if args.limit is not None and not 1 <= args.limit <= len(examples):
        parser.error(f"--limit must be between 1 and {len(examples)}")
    examples = examples[:args.limit]

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).to(args.device).eval()

    print(f"Evaluating {len(examples)} questions on {model.device}", flush=True)

    def predict(question: str) -> str:
        prediction = generate_direct_answer(model, tokenizer, question)
        print(f"{question}\n  Prediction: {prediction!r}", flush=True)
        return prediction

    correct, total = evaluate(examples, predict)
    print(f"Direct Inference EM: {correct}/{total} ({correct / total:.1%})")


if __name__ == "__main__":
    main()