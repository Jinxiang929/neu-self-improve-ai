import subprocess
import sys
from pathlib import Path

import modal

from modal_common import image, retrieval_server, volume


app = modal.App("search-r1-train-grpo")
SOURCE_DIR = Path(__file__).parent

image = image.add_local_python_source("modal_common")
for filename in (
    "train_grpo.py",
    "grpo.py",
    "lora_policy.py",
    "retriever.py",
    "reward.py",
    "rollout.py",
    "trajectory.py",
):
    image = image.add_local_file(SOURCE_DIR / filename, f"/app/{filename}")


@app.function(
    image=image, gpu="L4", cpu=4, memory=96 * 1024,
    timeout=3600, volumes={"/data": volume},
)
def run_grpo() -> None:
    with retrieval_server():
        subprocess.run(
            [sys.executable, "-u", "/app/train_grpo.py"],
            check=True,
        )
        volume.commit()
        print("GRPO checkpoint committed to Modal Volume.", flush=True)