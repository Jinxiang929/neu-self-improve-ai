import csv
from pathlib import Path


def load_bamboogle(path: Path) -> list[dict[str, str | list[str]]]:
    """Read each question and its accepted answers from the original CSV."""
    examples = []

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Question", "Answer", "Answer 2"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

        for row in reader:
            question = (row["Question"] or "").strip()
            if not question:
                continue

            answers = []
            for column in ("Answer", "Answer 2"):
                answer = (row[column] or "").strip()
                if answer:
                    answers.append(answer)

            if not answers:
                raise ValueError(f"No accepted answer for: {question}")

            examples.append({
                "id": f"bamboogle-{len(examples) + 1:03d}",
                "question": question,
                "answers": answers,
            })

    return examples


if __name__ == "__main__":
    examples = load_bamboogle(Path(__file__).with_name("bamboogle.csv"))
    assert len(examples) == 125, f"Expected 125 questions, got {len(examples)}"

    with_aliases = [example for example in examples if len(example["answers"]) > 1]
    print(f"Questions: {len(examples)}")
    print(f"Questions with a second answer: {len(with_aliases)}")
    for example in with_aliases[:3]:
        print(example["question"], example["answers"])