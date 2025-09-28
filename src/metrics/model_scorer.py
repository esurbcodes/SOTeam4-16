from huggingface_service import HuggingFaceService, ModelMetadata


def score_model_size(metadata: ModelMetadata) -> int:
    """Return a score based on model size in bytes."""
    size_mb = metadata.modelSize / (1024 * 1024)  # convert to MB
    size_gb = size_mb / 1024

    if size_mb < 500:
        return 1
    elif size_mb < 1000:
        return 2
    elif 1 <= size_gb <= 6:
        return 3
    else:  # > 6 GB
        return 2


# -------------------------
# Temporary test block
# -------------------------
"""
if __name__ == "__main__":
    service = HuggingFaceService()
    model_id = "bert-base-uncased"  # example model
    metadata = service.fetch_model_metadata(model_id)

    if metadata:
        score = score_model_size(metadata)
        print("✅ Test run — Model scoring")
        print("Model:", metadata.modelName)
        print("Size:", metadata.pretty_size())
        print("Score:", score)
"""
