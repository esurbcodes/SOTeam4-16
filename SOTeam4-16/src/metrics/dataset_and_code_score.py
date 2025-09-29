import time
import logging
from typing import Any, Dict, Tuple

from src.utils.dataset_link_finder import find_datasets_from_resource
from src.utils.github_link_finder import find_github_url_from_hf

logger = logging.getLogger("phase1_cli")


# ---------------------------------------------------------------------
# Backward compatibility shim for old tests
def find_dataset_url_from_hf(model_name: str) -> str | None:
    """
    Old compatibility function for tests.
    Uses new find_datasets_from_resource internally.
    Returns the first dataset URL if available, else None.
    """
    resource = {"name": model_name, "url": f"https://huggingface.co/{model_name}"}
    datasets, _ = find_datasets_from_resource(resource)
    return datasets[0] if datasets else None
# ---------------------------------------------------------------------


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates a score based on the successful discovery of both a linked
    dataset and a linked code repository from a model's card.

    Returns:
        score (float): 
            - 1.0 if both dataset(s) and GitHub repo are found
            - 0.5 if only one is found
            - 0.0 if neither are found
        latency_ms (int): total latency in milliseconds
    """
    start_time = time.perf_counter()
    score = 0.0

    if "huggingface.co" in resource.get("url", ""):
        # Dataset links
        datasets, _ = find_datasets_from_resource(resource)
        has_dataset = bool(datasets)

        # GitHub repo link
        github_url = find_github_url_from_hf(resource.get("name"))
        has_github = bool(github_url)

        # Assign scores
        if has_dataset and has_github:
            score = 1.0
        elif has_dataset or has_github:
            score = 0.5

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return score, latency_ms
