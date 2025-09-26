# structure_metric.py
from huggingface_service import HuggingFaceService

class StructureMetric:
    @staticmethod
    def calculate_structure_score(model_info: dict) -> int:
        """
        Calculates a score based on the model's file structure.
        
        Rules (example, tweak as you like):
        - If model has < 5 files → score 1
        - If model has 5–20 files → score 2
        - If model has 20–50 files → score 3
        - If model has > 50 files → score 4
        """

        if "siblings" not in model_info:
            return 0  # no file info available

        file_list = model_info["siblings"]
        file_count = len(file_list)

        if file_count < 5:
            return 1
        elif file_count < 20:
            return 2
        elif file_count < 50:
            return 3
        else:
            return 4

if __name__ == "__main__":
    # Example usage
    service = HuggingFaceService()
    model_id = "bert-base-uncased"
    model_info = service.get_model_info(model_id)

    score = StructureMetric.calculate_structure_score(model_info)
    print(f"Structure score for {model_id}: {score}")
