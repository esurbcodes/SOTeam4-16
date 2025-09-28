import time
import logging
from typing import Any, Dict, Tuple
from src.utils.dataset_link_finder import find_dataset_url_from_hf
from src.utils.github_link_finder import find_github_url_from_hf

logger = logging.getLogger("phase1_cli")

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates a score based on the successful discovery of both a linked
    dataset and a linked code repository from a model's card.
    """
    start_time = time.perf_counter()
    score = 0.0

    # This metric only applies to Hugging Face models
    if "huggingface.co" in resource['url']:
        # Use our existing utilities to find the links
        dataset_url = find_dataset_url_from_hf(resource['name'])
        github_url = find_github_url_from_hf(resource['name'])

        # Assign a score only if both are found
        if dataset_url and github_url:
            score = 1.0
        elif dataset_url or github_url:
            # Assign a partial score if only one is found
            score = 0.5

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return score, latency_ms