import re
from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizerBase


@torch.inference_mode()
def collect_search_rollout(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    retrieve: Callable[[str], str],
    max_searches: int = 4,
    max_new_tokens: int = 1024,
) -> tuple[list[int], list[tuple[list[int], bool]], str]:
    """Return the prompt IDs, rollout segments, and extracted answer."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context_ids = list(prompt_ids)
    segments: list[tuple[list[int], bool]] = []

    for searches_used in range(max_searches + 1):
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=model.device)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            tokenizer=tokenizer,
            stop_strings=["</search>", "</answer>"],
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
        )

        generated_ids = output_ids[0, input_ids.shape[-1]:].tolist()
        segments.append((generated_ids, True))
        context_ids.extend(generated_ids)

        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if answer_match is not None:
            return prompt_ids, segments, answer_match.group(1).strip()

        query_match = re.search(r"<search>(.*?)</search>", completion, re.DOTALL)
        if query_match is None or searches_used == max_searches:
            break

        query = query_match.group(1).strip()
        if not query:
            break

        information = retrieve(query)
        information_ids = tokenizer.encode(
            f"\n<information>{information}</information>\n",
            add_special_tokens=False,
        )
        segments.append((information_ids, False))
        context_ids.extend(information_ids)

    return prompt_ids, segments, ""