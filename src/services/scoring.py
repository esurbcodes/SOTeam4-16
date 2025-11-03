from __future__ import annotations
from typing import Dict, Any
import time

# Import your existing metrics/utilities
from src.metrics import reviewedness, dataset_quality, dataset_and_code_score, treescore
# reproducibility metric file name may differ; adapt import as needed
# from src.metrics import reproducibility
# utils used by dataset/code metrics are already in your repo (seen in CURRENT CODE)
# - src/utils/dataset_link_finder.py, src/utils/github_link_finder.py

NON_LATENCY = ("reviewedness", "dataset_quality", "dataset_and_code_score", "treescore")

class ScoringService:
    def rate(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        subs = {}
        lat_total = 0
        # Call each existing metric's metric(resource) -> (score, latency_ms)
        for name, mod in {
            "reviewedness": reviewedness,
            "dataset_quality": dataset_quality,
            "dataset_and_code_score": dataset_and_code_score,
            "treescore": treescore,
            # "reproducibility": reproducibility,
        }.items():
            try:
                score, lat = mod.metric(resource)  # current signature in your code
                subs[name] = float(score)
                lat_total += int(lat)
            except Exception:
                subs[name] = 0.0
        # Simple average as placeholder; adapt to your Phase 1 net-score formula
        net = sum(subs.values()) / max(len(subs), 1)
        return {"net": round(net, 4), "subs": subs, "latency_ms": lat_total}
