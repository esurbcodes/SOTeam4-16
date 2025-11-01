# src/metrics/dataset_quality.py
"""
Calculates a dataset quality score for a model, prioritizing checking model tags
for dataset declarations ('dataset:...') and falling back to README parsing.
Applies a baseline score if datasets are declared via tags.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Dict, Tuple, Set
from urllib.parse import urlparse
# Import HfApi for consistency and access to model_info
from huggingface_hub import HfApi, dataset_info
from huggingface_hub import HfFolder, RepositoryNotFoundError, HFValidationError
# Ensure this import is correct relative to your project structure
from src.utils.dataset_link_finder import find_datasets_from_resource, _normalize_dataset_ref # Import helper

logger = logging.getLogger("phase1_cli")

def get_dataset_id_from_url(url: str) -> str | None:
    """Extracts 'owner/name' or 'name' from a Hugging Face dataset URL."""
    try:
        path_parts = urlparse(url).path.strip("/").split("/")
        if path_parts and path_parts[0] == "datasets":
            # Join potentially 'owner/name' or just 'name'
            if len(path_parts) > 1:
                 # Normalize here to handle potential extra segments
                 return _normalize_dataset_ref("/".join(path_parts[1:3] if len(path_parts) > 2 else path_parts[1:2]))
    except Exception: pass
    return None

def score_single_dataset(dataset_id: str, token: str | None) -> float:
    """Fetches metadata for a single dataset and returns its quality score."""
    # Only attempt to score IDs that look valid (contain '/' for owner/name)
    if not dataset_id or (not '/' in dataset_id and dataset_id not in ['squad', 'glue', 'bookcorpus', 'wikipedia']):
        logger.debug(f"Skipping scoring for potentially invalid or non-scorable dataset ID: '{dataset_id}'")
        return 0.0

    api_dataset_id = dataset_id
    try:
        info = dataset_info(api_dataset_id, token=token)
        card_score = 0.5 if hasattr(info, 'cardData') and info.cardData else 0.0
        downloads_score = 0.3 if info.downloads and info.downloads > 1000 else 0.0
        likes_score = 0.2 if info.likes and info.likes > 10 else 0.0
        score = card_score + downloads_score + likes_score
        logger.debug(f"Scored dataset '{api_dataset_id}': Card={card_score > 0}, Downloads={info.downloads}, Likes={info.likes}, Score={score:.2f}")
        return score
    except RepositoryNotFoundError:
        logger.warning(f"Dataset Quality: Referenced dataset '{api_dataset_id}' not found on Hub.")
        return 0.0
    except HFValidationError as e:
         logger.warning(f"Dataset Quality: Invalid dataset ID format '{api_dataset_id}': {e}")
         return 0.0
    except Exception as e:
        logger.error(f"Dataset Quality: Could not score dataset '{api_dataset_id}' due to API error: {e}", exc_info=False)
        return 0.0

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Finds linked datasets and scores them, checking model_info.tags
    for dataset declarations.
    """
    start_time = time.perf_counter()
    final_score = 0.0
    model_repo_id = resource.get("name")
    # Flag to track if the model *intended* to declare datasets via tags
    datasets_declared_via_tags = False
    all_found_refs: Set[str] = set() # Store all refs found (tags + readme)
    metadata_checked_and_exists = False

    token = HfFolder.get_token()
    if not token: logger.warning("Hugging Face token not found by dataset quality metric.")
    else: logger.debug("Dataset quality metric using detected Hugging Face token.")

    api = HfApi(token=token)

    # Step 1: Check model's metadata tags via API
    if model_repo_id:
        try:
            model_meta = api.model_info(model_repo_id)
            metadata_checked_and_exists = True

            # --- Check the 'tags' attribute for dataset declarations ---
            model_tags = getattr(model_meta, 'tags', [])
            if isinstance(model_tags, list):
                 declared_in_tags = {tag.split(":", 1)[1] for tag in model_tags if isinstance(tag, str) and tag.startswith("dataset:")}
                 if declared_in_tags:
                      datasets_declared_via_tags = True # Mark that declarations were found in tags
                      # Normalize refs found in tags
                      normalized_tag_refs = {norm_ref for ds in declared_in_tags if (norm_ref := _normalize_dataset_ref(ds))}
                      all_found_refs.update(normalized_tag_refs)
                      logger.debug(f"Found declared dataset references in tags for '{model_repo_id}': {normalized_tag_refs}")
                 else:
                     logger.debug(f"Model '{model_repo_id}' tags do not contain any 'dataset:' declarations.")
            else:
                logger.debug(f"model_meta object for '{model_repo_id}' does not have a 'tags' list attribute.")

        except RepositoryNotFoundError:
             logger.error(f"Dataset Quality: Model repository '{model_repo_id}' not found.")
             metadata_checked_and_exists = False
        except Exception as e:
            logger.error(f"Dataset Quality: Could not check model info for '{model_repo_id}' due to API error: {e}", exc_info=False)
            metadata_checked_and_exists = False

    # Step 2: Fallback to README parsing, add unique refs to found set
    # Run README parse regardless, to catch links not mentioned in tags
    readme_dataset_refs: Set[str] = set()
    try:
        dataset_refs_readme, _ = find_datasets_from_resource(resource)
        readme_dataset_refs.update(dataset_refs_readme)
        if readme_dataset_refs:
             # Add newly found refs from README to the main set
             new_refs_from_readme = readme_dataset_refs - all_found_refs # Find refs only in README
             if new_refs_from_readme:
                  logger.debug(f"Found additional dataset references in README for '{model_repo_id}': {new_refs_from_readme}")
                  all_found_refs.update(new_refs_from_readme)
    except Exception as e:
        logger.error(f"Dataset Quality: Error parsing README for '{model_repo_id}': {e}", exc_info=False)

    logger.debug(f"All potential dataset references combined for '{model_repo_id}': {all_found_refs}")

    # Step 3: Score the datasets that are in 'owner/name' format
    if all_found_refs:
        # Filter for scorable IDs (must contain '/')
        scorable_ids = {ref for ref in all_found_refs if ref and '/' in ref}
        if scorable_ids:
             scores = [score_single_dataset(ds_id, token=token) for ds_id in scorable_ids]
             if scores:
                 final_score = max(scores)
                 logger.debug(f"Highest score among scorable datasets for '{model_repo_id}' is {final_score:.2f}")
        else:
            canonical_refs_found = {ref for ref in all_found_refs if ref and '/' not in ref}
            if canonical_refs_found: logger.debug(f"Only canonical dataset names found for '{model_repo_id}': {canonical_refs_found}. Cannot score directly.")
            else: logger.debug(f"No scorable dataset IDs (owner/name format) found among references for '{model_repo_id}'.")


    # Step 4: Apply baseline score logic based on TAGS
    # Apply baseline if metadata check happened, declarations were found *via tags*, AND scoring resulted in 0
    if final_score == 0.0 and metadata_checked_and_exists and datasets_declared_via_tags:
        final_score = 0.5
        logger.debug(f"Applying baseline score of 0.5 for '{model_repo_id}' because datasets were declared in metadata tags but could not be scored higher.")
    elif metadata_checked_and_exists and not datasets_declared_via_tags:
         logger.debug(f"Metadata checked for '{model_repo_id}', but no datasets were declared via tags.")


    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return round(final_score, 2), latency_ms
