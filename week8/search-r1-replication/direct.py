import subprocess
import sys
from pathlib import Path

import modal


SOURCE_DIR = Path(__file__).parent
app = modal.App("search-r1-direct-inference")
volume = modal.Volume.from_name("search-r1-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install("torch==2.13.0", "transformers==5.19.0")
    .env({"HF_HOME": "/data/huggingface"})
)

for filename in ("direct_inference.py", "bamboogle.py", "evaluate.py", "reward.py", "bamboogle.csv"):
    image = image.add_local_file(SOURCE_DIR / filename, f"/app/{filename}")


@app.function(image=image, gpu="L4", timeout=1800, volumes={"/data": volume})
def run_direct(limit: int = 125) -> None:
    subprocess.run(
        [
            sys.executable, "-u", "/app/direct_inference.py",
            "--device", "cuda", "--limit", str(limit),
        ],
        check=True,
    )
    volume.commit()