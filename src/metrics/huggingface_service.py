# huggingface_service.py (only showing updated main block at bottom)

import re

def extract_model_id(url: str) -> str | None:
    """
    Extracts the Hugging Face model or dataset ID from a URL.
    Returns None if it's not a Hugging Face URL.
    """
    if "huggingface.co" not in url:
        return None

    # strip query params, anchors, /tree/main, etc.
    clean_url = re.sub(r"(\?|#).*", "", url).strip()
    clean_url = clean_url.replace("/tree/main", "").rstrip("/")

    # get everything after huggingface.co/
    try:
        after_domain = clean_url.split("huggingface.co/")[1]
    except IndexError:
        return None

    # skip "datasets/" prefix
    if after_domain.startswith("datasets/"):
        after_domain = after_domain.replace("datasets/", "", 1)

    return after_domain if after_domain else None


if __name__ == "__main__":
    import sys

    # pass text file path as argument, e.g.
    # python huggingface_service.py urls.txt
    file_path = sys.argv[1] if len(sys.argv) > 1 else "urls.txt"

    service = HuggingFaceService()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            model_id = extract_model_id(url)
            if model_id is None:
                print(f"Skipping non-HuggingFace URL: {url}")
                continue

            print(f"\nFetching metadata for: {model_id} (from {url})")
            metadata = service.fetch_model_metadata(model_id)
            if metadata:
                print("  Name:", metadata.modelName)
                print("  Category:", metadata.modelCategory)
                print("  Size:", metadata.modelSize, "bytes")
                print("  License:", metadata.license)
                print("  Downloads:", metadata.timesDownloaded)
                print("  Likes:", metadata.modelLikes)
                print("  Last modified:", metadata.lastModified)
                print("  Files:", len(metadata.files))
            else:
                print(f"  Failed to fetch metadata for {model_id}")
