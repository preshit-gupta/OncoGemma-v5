"""
OncoGemma Stage v4.3 - Pure Nottingham Mitotic Scoring Engine.
Computes point-in-circle containment of mitotic figures within virtual HPFs,
calculates standardized area-normalized density (mitoses/mm²), and assigns
Elston-Ellis Nottingham Mitotic Scores (Score 1, 2, or 3).
"""
import math
import os
from typing import List, Dict, Any, Tuple, Optional
import yaml


def calculate_hpf_mitosis_counts(
    candidates: List[Dict[str, Any]],
    hpfs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Computes which mitotic candidates fall inside each virtual HPF circle.
    Updates the 'count' field on each HPF and returns updated HPF list + total count.
    
    A candidate is considered a confirmed mitosis if label == "mitosis".
    """
    updated_hpfs = []
    mitoses_in_any_hpf = set()

    for hpf in hpfs:
        hpf_copy = dict(hpf)
        cx, cy = hpf_copy["center_um"]
        r = float(hpf_copy.get("radius_um", 262.0))
        r_sq = r * r

        hpf_mitosis_count = 0
        for cand in candidates:
            if cand.get("label") != "mitosis":
                continue

            cand_x, cand_y = cand["centroid_um"]
            dist_sq = (cand_x - cx) ** 2 + (cand_y - cy) ** 2
            if dist_sq <= r_sq:
                hpf_mitosis_count += 1
                mitoses_in_any_hpf.add(cand.get("id"))

        hpf_copy["count"] = hpf_mitosis_count
        updated_hpfs.append(hpf_copy)

    total_count = len(mitoses_in_any_hpf)
    return updated_hpfs, total_count


def load_scoring_config() -> Dict[str, Any]:
    """Loads configs/scoring.yaml or configs/mitosis.yaml."""
    cfg_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/scoring.yaml")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/mitosis.yaml")),
    ]
    for p in cfg_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        return data
            except Exception:
                pass
    return {}


def compute_nottingham_mitotic_score(
    count_total: int,
    n_hpf: int = 10,
    radius_um: float = 262.0,
    config_dict: Optional[Dict[str, Any]] = None,
    hpfs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Computes Nottingham Mitotic Score based on standardized mm² area normalization.
    
    Standard thresholds (Elston-Ellis):
      - Score 1: < 3.65 mitoses/mm²  (Classic: 0 - 9 / 2.74 mm²)
      - Score 2: 3.65 - 7.30 mitoses/mm² (Classic: 10 - 19 / 2.74 mm²)
      - Score 3: >= 7.30 mitoses/mm² (Classic: >= 20 / 2.74 mm²)
    """
    if config_dict is None:
        config_dict = load_scoring_config()

    score2_min = 3.65
    score3_min = 7.30
    classic_area_mm2 = 2.74

    if config_dict:
        # Check 'mitotic_score' (scoring.yaml) or 'scoring' (mitosis.yaml)
        m_cfg = config_dict.get("mitotic_score") or config_dict.get("scoring") or {}
        thresh = m_cfg.get("thresholds", {})
        score2_min = float(thresh.get("score2_min", score2_min))
        score3_min = float(thresh.get("score3_min", score3_min))
        classic_area_mm2 = float(m_cfg.get("classic_area_mm2", classic_area_mm2))

    # Calculate actual cumulative HPF inspection area summing per-HPF radius (#764)
    if hpfs and len(hpfs) > 0:
        n_fields = len(hpfs)
        area_mm2 = sum(math.pi * ((float(h.get("radius_um", radius_um)) / 1000.0) ** 2) for h in hpfs)
    else:
        n_fields = n_hpf
        single_hpf_area_mm2 = math.pi * ((radius_um / 1000.0) ** 2)
        area_mm2 = float(n_fields * single_hpf_area_mm2)

    # Clean zero-HPF state handling (#373)
    if n_fields <= 0 or area_mm2 <= 0.0:
        return {
            "count_total": int(count_total),
            "n_hpf": 0,
            "area_mm2": 0.0,
            "per_mm2": 0.0,
            "mitoses_per_mm2": 0.0,
            "classic_per_10hpf": 0.0,
            "mitotic_score": 1,
            "score": 1
        }

    density = float(count_total) / area_mm2
    classic_per_10hpf = density * classic_area_mm2

    # Determine score
    if density >= score3_min:
        mitotic_score = 3
    elif density >= score2_min:
        mitotic_score = 2
    else:
        mitotic_score = 1

    return {
        "count_total": int(count_total),
        "n_hpf": int(n_fields),
        "area_mm2": round(area_mm2, 3),
        "per_mm2": round(density, 2),
        "mitoses_per_mm2": round(density, 2),
        "classic_per_10hpf": round(classic_per_10hpf, 1),
        "mitotic_score": mitotic_score,
        "score": mitotic_score
    }
