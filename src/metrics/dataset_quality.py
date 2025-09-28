import time
import logging
from typing import Any, Dict, Tuple
from huggingface_hub import dataset_info
from src.utils.dataset_link_finder import find_dataset_url_from_hf

logger = logging.getLogger("phase1_cli")

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates a dataset quality score by finding a linked dataset
    and assessing its metadata on the Hugging Face Hub.
    """
    start_time = time.perf_counter()
    score = 0.0

    # Step 1: Find a linked dataset URL from the model's card
    dataset_url = find_dataset_url_from_hf(resource['name'])

    if dataset_url:
        try:
            # Extract the dataset ID from the URL
            dataset_id = "/".join(dataset_url.rstrip('/').split('/')[-2:])

            # Step 2: Fetch the dataset's metadata from the Hub
            info = dataset_info(dataset_id)

            # Step 3: Calculate score based on metadata
            card_score = 0.5 if (info.cardData and "dataset_card" in info.cardData) else 0.0
            downloads_score = 0.3 if info.downloads > 1000 else 0.0
            likes_score = 0.2 if info.likes > 10 else 0.0

            score = card_score + downloads_score + likes_score

        except Exception as e:
            logger.error(f"Failed to get info for dataset {dataset_url}: {e}")
            score = 0.0

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return score, latency_ms