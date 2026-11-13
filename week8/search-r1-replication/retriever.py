import json
from urllib.request import Request, urlopen


def retrieve(
    query: str,
    url: str = "http://127.0.0.1:8000/retrieve",
) -> str:
    """Fetch and format the top three passages for one query."""
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        documents = json.load(response)["result"][0]

    passages = []
    for index, item in enumerate(documents, start=1):
        title, _, text = item["document"]["contents"].partition("\n")
        passages.append(f"Doc {index}(Title: {title}) {text}")

    return "\n".join(passages)