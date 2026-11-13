import subprocess
import sys
from pathlib import Path

import modal

from modal_common import image, retrieval_server, volume


app = modal.App("search-r1-official-ppo")
MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2"
SOURCE_DIR = Path(__file__).parent

image = image.add_local_python_source("modal_common")
for filename in (
    "search_inference.py",
    "retriever.py",
    "bamboogle.py",
    "bamboogle.csv",
    "evaluate.py",
    "reward.py",
):
    image = image.add_local_file(SOURCE_DIR / filename, f"/app/{filename}")


@app.function(
    image=image, gpu="L4", cpu=4, memory=96 * 1024,
    timeout=3600, volumes={"/data": volume},
)
def run_eval(limit: int = 125) -> None:
    with retrieval_server():
        subprocess.run(
            [
                sys.executable, "-u", "/app/search_inference.py",
                "--model-id", MODEL_ID,
                "--limit", str(limit),
            ],
            check=True,
        )