# src/metrics/structure_metric.py
from typing import Optional
try:
    from services.huggingface_service import HuggingFaceService
except Exception:
    from huggingface_service import HuggingFaceService


class StructureMetricRaw:
    @staticmethod
    def calculate_structure_score(model_info) -> int:
        """
        Score based on the number of files (siblings) inside raw ModelInfo.
        """
        siblings = getattr(model_info, "siblings", []) or []
        file_count = len(siblings)

        if file_count == 0:
            return 0
        if file_count < 5:
            return 1
        if file_count < 20:
            return 2
        if file_count < 50:
            return 3
        return 4

    @staticmethod
    def score_model(model_id: str, token: Optional[str] = None) -> Optional[int]:
        service = HuggingFaceService(token=token)
        # directly pull raw ModelInfo
        model_info = service.api.model_info(model_id)
        if model_info is None:
            print(f"Could not fetch model info for '{model_id}'")
            return None
        return StructureMetricRaw.calculate_structure_score(model_info)


if __name__ == "__main__":
    import sys

    model_id = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
    score = StructureMetricRaw.score_model(model_id)
    print(f"Structure score for {model_id}: {score}")
