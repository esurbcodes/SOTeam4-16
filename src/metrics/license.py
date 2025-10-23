# src/metrics/license.py
"""
Calculates a license compatibility score for a model's repository.

This metric first attempts to find and read a LICENSE file in the local repository.
It calculates a score based on two methods:
1.  Heuristic Fallback: A fast, local check for common permissive license keywords (MIT, Apache, etc.).
2.  LLM-Powered Analysis: If a `GEN_AI_STUDIO_API_KEY` environment variable is set, it calls the
    Gemini API with the license text, asking for a `compatibility_score` in a JSON format.

The LLM-powered score is preferred. If the API call fails, is disabled, or returns an
invalid format, the metric gracefully falls back to the heuristic score.
"""
from __future__ import annotations
import os
import time
import logging
import json
import re
import requests
from typing import Any, Dict, Tuple

logger = logging.getLogger("phase1_cli")

# --- Heuristic Scoring Logic ---

# A dictionary mapping license keywords to their compatibility scores and labels.
LICENSE_KEYWORDS = {
    "mit": (1.0, "MIT"),
    "apache": (0.95, "Apache-2.0"),
    "bsd": (0.9, "BSD"),
    "mozilla": (0.75, "MPL"),
    "mpl": (0.75, "MPL"),
    "lgpl": (0.6, "LGPL"),
    "creative commons": (0.5, "CC-BY"),
    "cc-by": (0.5, "CC-BY"),
    "gpl": (0.4, "GPL"),
}

def heuristic_license_score(text: str) -> Tuple[float, str, str]:
    """
    Performs a fast, heuristic-based scoring of license text by searching for keywords.
    Returns a tuple of (score, label, method).
    """
    if not text:
        return 0.0, "Unknown", "Heuristic"

    lower_text = text.lower()
    for keyword, (score, label) in LICENSE_KEYWORDS.items():
        if keyword in lower_text:
            return score, label, "Heuristic"
    
    return 0.0, "Unknown", "Heuristic"

# --- Helper Functions ---

def _read_license_file(local_dir: str) -> str | None:
    """
    Searches for common license filenames in a directory and returns the content
    of the first one found.
    """
    if not local_dir or not os.path.isdir(local_dir):
        return None
    
    for filename in ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"]:
        path = os.path.join(local_dir, filename)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                logger.debug(f"Could not read license file {path}: {e}")
    return None

def _extract_json_from_assistant(content: str) -> Dict[str, Any] | None:
    """
    Safely extracts a JSON object from a string, handling common LLM response formats
    like markdown code fences and single quotes.
    """
    if not content:
        return None
    
    # Find the JSON block, which might be inside a markdown fence
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
        
    json_str = match.group(0)
    try:
        # First attempt: parse directly
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Second attempt: handle single quotes
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from LLM response after attempting to fix quotes.")
            return None

# --- Main Metric Function ---

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates the license score, using an LLM if available, otherwise falling back
    to a keyword-based heuristic.
    """
    start_time = time.perf_counter()
    final_score = 0.0

    # Step 1: Read the license file from the local repository path
    local_dir = resource.get("local_path") or resource.get("local_dir")
    license_text = _read_license_file(local_dir)

    # Step 2: Calculate the heuristic score as a reliable fallback
    if license_text:
        final_score, _, _ = heuristic_license_score(license_text)

    # Step 3: Attempt to get a more accurate score from the Gemini LLM
    api_key = os.environ.get("GEN_AI_STUDIO_API_KEY")
    if api_key and license_text:
        prompt = (
            "Analyze the following license text and determine its compatibility for commercial reuse "
            "in an open-source project. Respond ONLY with a single valid JSON object containing one key, "
            f"'compatibility_score', with a float value between 0.0 (very restrictive) and 1.0 (very permissive).\n\n"
            f"LICENSE TEXT:\n\"\"\"\n{license_text[:4000]}\n\"\"\""
        )
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15,
            )
            
            if response.status_code == 200:
                data = response.json()
                llm_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                if (parsed_json := _extract_json_from_assistant(llm_text)) and isinstance(score := parsed_json.get("compatibility_score"), (float, int)):
                    final_score = float(max(0.0, min(1.0, score))) # Clamp score to [0,1]
            else:
                 logger.debug(f"LLM API returned status {response.status_code}, using heuristic score.")

        except (requests.RequestException, KeyError, IndexError, TypeError) as e:
            logger.debug(f"LLM call for license metric failed, using heuristic score. Error: {e}")
            # On any error, the 'final_score' remains the heuristic one.

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return round(final_score, 2), latency_ms
