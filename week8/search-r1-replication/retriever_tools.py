import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import modal

from modal_common import (
    corpus_path,
    data_dir,
    image,
    index_path,
    repo_path,
    retrieval_server,
    volume,
)


app = modal.App("search-r1-wiki-retriever")


def write_file(destination: Path, command: list[str]) -> None:
    """Make the output visible under its final name after the command succeeds."""
    temporary_path = destination.with_suffix(".partial")
    with temporary_path.open("wb") as output:
        subprocess.run(command, stdout=output, check=True)
    temporary_path.replace(destination)


source_dir = Path(__file__).parent
image = image.add_local_python_source("modal_common")
for filename in (
    "search_inference.py",
    "retriever.py",
    "bamboogle.py",
    "evaluate.py",
    "reward.py",
    "bamboogle.csv",
    "train_grpo.py",
    "grpo.py",
    "rollout.py",
    "trajectory.py",
    "lora_policy.py",
):
    image = image.add_local_file(source_dir / filename, f"/app/{filename}")


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes={"/data": volume})
def prepare_index() -> None:
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    if not index_path.is_file() or not corpus_path.is_file():
        subprocess.run(
            [
                sys.executable, "-u", str(repo_path / "scripts/download.py"),
                "--save_path", str(data_dir),
            ],
            check=True,
        )

    if not index_path.is_file():
        print("Combining FAISS index parts.", flush=True)
        write_file(index_path, [
            "cat", str(data_dir / "part_aa"), str(data_dir / "part_ab"),
        ])

    if not corpus_path.is_file():
        print("Decompressing Wikipedia passages.", flush=True)
        write_file(corpus_path, ["gzip", "-dc", str(data_dir / "wiki-18.jsonl.gz")])

    corpus = load_dataset("json", data_files=str(corpus_path), split="train", num_proc=4)
    volume.commit()
    print(f"Index ready. Wikipedia passages: {len(corpus):,}")


@app.function(
    image=image, gpu="L4", cpu=4, memory=96 * 1024,
    timeout=3600, volumes={"/data": volume},
)
def check_top3(
    model_id: str = "",
    limit: int = 1,
    start: int = 0,
    train_grpo: bool = False,
    adapter_path: str = "",
) -> None:
    with retrieval_server() as base_url:
        if train_grpo:
            subprocess.run(
                [sys.executable, "-u", "/app/train_grpo.py"],
                check=True,
            )
            volume.commit()
            return

        if model_id:
            command = [
                sys.executable, "-u", "/app/search_inference.py",
                "--model-id", model_id,
                "--limit", str(limit),
                "--start", str(start),
            ]
            if adapter_path:
                command.extend(["--adapter-path", adapter_path])
            subprocess.run(command, check=True)
            return

        request = Request(
            f"{base_url}/retrieve",
            data=json.dumps({
                "queries": ["Apollo 11"],
                "topk": 3,
                "return_scores": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            results = json.load(response)["result"][0]

        assert len(results) == 3, f"Expected 3 passages, got {len(results)}"
        for rank, result in enumerate(results, start=1):
            title = result["document"]["contents"].partition("\n")[0]
            print(f"{rank}. {title} (score={result['score']:.4f})")
        print("Official Wikipedia/E5 top-3 check passed.")


@app.local_entrypoint()
def main(limit: int = 125) -> None:
    check_top3.remote(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="/data/ppo-adapters/ppo-short-20261113T020435277459Z",
        limit=limit,
    )