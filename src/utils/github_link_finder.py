# src/utils/github_link_finder.py
import logging
import re
from typing import Optional
from functools import lru_cache
import requests  # <-- new
from huggingface_hub import hf_hub_download

logger = logging.getLogger("phase1_cli")

_GITHUB_RE_MARKDOWN = re.compile(r'\[[^\]]+\]\((https?://github\.com/[^\'\"\)\>\s\]]+)\)', re.I)
_GITHUB_RE_PLAIN    = re.compile(r'(https?://github\.com/[^\'\"\)\>\s\]]+)', re.I)
_GITHUB_RE_BARE     = re.compile(r'(github\.com/[^\'\"\)\>\s\]]+)', re.I)

def _normalize_github_href(href: str) -> str:
    href = href.strip()
    if href.startswith("github.com/"):
        return "https://" + href
    return href

def find_github_url_from_hf(repo_id: str) -> Optional[str]:
    """
    Try to extract a GitHub URL from a model's README on Hugging Face.

    Contract for tests:
      - If hf_hub_download succeeds and README has no GitHub link (or is empty/invalid),
        return None (do NOT attempt network fallbacks).
      - Only if hf_hub_download FAILS should we try lightweight HTTP fallbacks with short timeouts.
    """
    content = ""
    used_hf_download = False

    # --- 1) Primary path: hf_hub_download (tests patch this) ---
    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            token=None,
            repo_type="model",
            etag_timeout=5,
            timeout=5,
        )
        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        used_hf_download = True
    except Exception:
        content = ""
        used_hf_download = False

    # If we successfully fetched via hf_hub_download, parse ONLY that content.
    if used_hf_download:
        text = (content or "").strip()
        if not text:
            return None

        m = _GITHUB_RE_MARKDOWN.search(text)
        if m:
            return _normalize_github_href(m.group(1))

        m = _GITHUB_RE_PLAIN.search(text)
        if m:
            return _normalize_github_href(m.group(1))

        m = _GITHUB_RE_BARE.search(text)
        if m:
            return _normalize_github_href(m.group(1))

        logger.warning("No GitHub link found in README for %s", repo_id)
        return None


    # --- 2) Fallbacks ONLY if hf_hub_download failed ---
    # Try raw README via HTTP (main/master) with short timeouts
    try:
        import requests
        for url in (
            f"https://huggingface.co/{repo_id}/raw/main/README.md",
            f"https://huggingface.co/{repo_id}/raw/master/README.md",
        ):
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200 and r.text.strip():
                    text = r.text
                    m = _GITHUB_RE_MARKDOWN.search(text) or _GITHUB_RE_PLAIN.search(text) or _GITHUB_RE_BARE.search(text)
                    if m:
                        return _normalize_github_href(m.group(1))
                    # README exists but contains no link → stop here
                    return None
            except Exception:
                pass
        # Fallback: scan model page HTML (lightweight)
        try:
            page = requests.get(f"https://huggingface.co/{repo_id}", timeout=5)
            if page.status_code == 200 and page.text:
                m = re.search(r"https?://github\.com/[^\s)\"'>]+", page.text, re.I)
                if m:
                    return _normalize_github_href(m.group(0))
        except Exception:
            pass
    except Exception:
        pass

    # Nothing found
    return None
