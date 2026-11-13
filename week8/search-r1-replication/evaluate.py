from collections.abc import Callable

from reward import exact_match


def evaluate(
    examples: list[dict[str, str | list[str]]],
    predict: Callable[[str], str],
) -> tuple[int, int]:
    """Return the number of exact matches and the number of questions."""
    correct = 0
    for example in examples:
        prediction = predict(example["question"])
        correct += exact_match(prediction, example["answers"])

    return correct, len(examples)