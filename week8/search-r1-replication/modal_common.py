import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import modal


volume = modal.Volume.from_name(
    "search-r1-wiki18", create_if_missing=True, version=2
)

data_dir = Path("/data/wiki18")
index_path = data_dir / "e5_Flat.index"
corpus_path = data_dir / "wiki-18.jsonl"
repo_path = Path("/opt/Search-R1")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "gzip")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.19.0",
        "datasets==5.1.0",
        "faiss-cpu",
        "fastapi",
        "uvicorn",
        "peft==0.21.2",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/PeterGriffinJin/Search-R1.git /opt/Search-R1"
    )
    .env({"HF_HOME": "/data/huggingface"})
)


@contextmanager
def retrieval_server() -> Iterator[str]:
    """Start the retriever and yield its base URL once it is ready."""
    if not index_path.is_file() or not corpus_path.is_file():
        raise FileNotFoundError("Run modal_retriever.py::prepare_index first.")

    base_url = "http://127.0.0.1:8000"
    server = subprocess.Popen([
        sys.executable, "-u",
        str(repo_path / "search_r1/search/retrieval_server.py"),
        "--index_path", str(index_path),
        "--corpus_path", str(corpus_path),
        "--retriever_name", "e5",
        "--retriever_model", "intfloat/e5-base-v2",
        "--topk", "3",
    ])

    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Retriever exited with code {server.returncode}")
            try:
                with urlopen(f"{base_url}/openapi.json", timeout=2):
                    break
            except (URLError, TimeoutError):
                time.sleep(2)
        else:
            raise TimeoutError("Retriever did not start within 15 minutes.")

        yield base_url
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)