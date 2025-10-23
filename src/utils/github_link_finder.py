# src/utils/github_link_finder.py
import logging
import re
from typing import Optional

from huggingface_hub import hf_hub_download

logger = logging.getLogger("phase1_cli")

# Stop characters: whitespace, angle bracket, quote, paren, bracket
_GITHUB_RE_MARKDOWN = re.compile(r'\[[^\]]+\]\((https?://github\.com/[^\'\"\)\>\s\]]+)\)', re.I)
_GITHUB_RE_PLAIN = re.compile(r'(https?://github\.com/[^\'\"\)\>\s\]]+)', re.I)
_GITHUB_RE_BARE = re.compile(r'(github\.com/[^\'\"\)\>\s\]]+)', re.I)

def _normalize_github_href(href: str) -> str:
    href = href.strip()
    if href.startswith("github.com/"):
        return "https://" + href
    return href

def find_github_url_from_hf(repo_id: str) -> Optional[str]:
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = ""

    if not content:
        logger.warning("No GitHub link found in README for %s", repo_id)
        return None

    m = _GITHUB_RE_MARKDOWN.search(content)
    if m:
        url = _normalize_github_href(m.group(1))
        logger.info("Found GitHub link (markdown) for %s: %s", repo_id, url)
        return url

    m = _GITHUB_RE_PLAIN.search(content)
    if m:
        url = _normalize_github_href(m.group(1))
        logger.info("Found GitHub link (plain) for %s: %s", repo_id, url)
        return url

    m = _GITHUB_RE_BARE.search(content)
    if m:
        url = _normalize_github_href(m.group(1))
        logger.info("Found GitHub link (bare) for %s: %s", repo_id, url)
        return url

    logger.warning("No GitHub link found in README for %s", repo_id)
    return None
