import re
import string


def normalize_answer(text: str) -> str:
    """Normalize an answer before exact-match comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> int:
    """Score 1 if the prediction matches any accepted answer."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0

    return int(
        any(normalized_prediction == normalize_answer(answer) for answer in answers)
    )