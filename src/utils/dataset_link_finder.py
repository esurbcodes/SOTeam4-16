import logging
from huggingface_hub import hf_hub_download
from bs4 import BeautifulSoup

logger = logging.getLogger("phase1_cli")

def find_dataset_url_from_hf(repo_id: str) -> str | None:
    """
    Downloads a model's README.md from Hugging Face and searches for a dataset URL.
    """
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        soup = BeautifulSoup(content, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "huggingface.co/datasets/" in href:
                logger.info(f"Found dataset link for {repo_id}: {href}")
                return href
        
        logger.warning(f"No dataset link found in README for {repo_id}")
        return None
    except FileNotFoundError:
         # This can happen if the repo exists but has no README.md
        logger.warning(f"No README.md found for {repo_id} to find dataset link.")
        return None
    except Exception as e: # Catches other errors, like API failures
        logger.error(f"Could not process README for {repo_id} to find dataset link: {e}")
        return None