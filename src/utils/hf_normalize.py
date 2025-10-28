from urllib.parse import urlparse

def normalize_hf_id(url_or_id: str) -> str:
    """
    Normalize a Hugging Face model or dataset identifier.

    Examples:
        "https://huggingface.co/openai/whisper-tiny/tree/main" -> "openai/whisper-tiny"
        "facebook/wav2vec2-base" -> "facebook/wav2vec2-base"
    """
    if "huggingface.co" not in url_or_id:
        return url_or_id.split("/tree/")[0]
    p = urlparse(url_or_id)
    parts = [x for x in p.path.strip("/").split("/") if x]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return url_or_id.split("/tree/")[0]
