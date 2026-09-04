"""
Pure Zero-LLM Nottingham Histologic Grading Aggregation Engine.

All arithmetic, median/mode voting, tie-breaking, and Nottingham grade synthesis
are strictly calculated in pure deterministic Python code. The LLM never computes
any numbers or aggregates.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import yaml

# Default configuration parameters
DEFAULT_CONF_WEIGHTS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
    "default": 1.0
}
DEFAULT_TUBULE_SCORE1_MIN = 75.0
DEFAULT_TUBULE_SCORE2_MIN = 10.0
DEFAULT_GRADE1_MAX_SUM = 5
DEFAULT_GRADE2_MAX_SUM = 7
DEFAULT_MIN_TUMOR_PATCHES = 8
DEFAULT_MAX_DISP = 0.30


def load_scoring_config() -> Dict[str, Any]:
    """Load scoring thresholds and weights from configs/scoring.yaml if present."""
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/scoring.yaml"))
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def weighted_median(values: List[float], weights: List[float]) -> float:
    """
    Compute deterministic weighted median for continuous values.
    
    Args:
        values: List of numeric values (e.g., tubule percentages).
        weights: Corresponding positive weights.
        
    Returns:
        Weighted median value as float.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    
    # Pair and sort by value ascending
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total_weight = sum(w for _, w in paired)
    if total_weight <= 0:
        return float(values[len(values) // 2])
    
    half_weight = total_weight / 2.0
    cumulative_weight = 0.0
    
    for i, (val, w) in enumerate(paired):
        cumulative_weight += w
        if cumulative_weight >= half_weight:
            # Check for exact midpoint tie across distinct values
            if cumulative_weight == half_weight and i + 1 < len(paired):
                return float((val + paired[i + 1][0]) / 2.0)
            return float(val)
            
    return float(paired[-1][0])


def weighted_mode(values: List[int], weights: List[float], tie_breaker=max) -> Tuple[int, float]:
    """
    Compute deterministic weighted mode for discrete categories (e.g. pleomorphism scores 1, 2, 3).
    Ties resolve via tie_breaker (default max: conservative clinical rule favoring worse grade).
    
    Args:
        values: List of discrete scores (e.g. [1, 2, 3]).
        weights: Corresponding positive weights.
        tie_breaker: Function to resolve ties among candidate scores with equal max weight.
        
    Returns:
        Tuple of (winning_score, disagreement_ratio).
    """
    if not values:
        return 2, 1.0  # Default to moderate score if empty
        
    weight_totals: Dict[int, float] = {}
    for val, w in zip(values, weights):
        weight_totals[val] = weight_totals.get(val, 0.0) + w
        
    total_weight = sum(weights)
    if total_weight <= 0:
        return tie_breaker(values), 0.0
        
    max_w = max(weight_totals.values())
    candidates = [val for val, w in weight_totals.items() if abs(w - max_w) < 1e-9]
    
    winning_score = tie_breaker(candidates)
    winning_weight = weight_totals[winning_score]
    disagreement_ratio = 1.0 - (winning_weight / total_weight)
    
    return winning_score, disagreement_ratio


def calculate_tubule_score(tubule_percent: float, cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    Map tubule formation percentage to Elston-Ellis Nottingham score:
    - Score 1: > 75%
    - Score 2: 10% - 75%
    - Score 3: < 10%
    """
    score1_min = DEFAULT_TUBULE_SCORE1_MIN
    score2_min = DEFAULT_TUBULE_SCORE2_MIN
    
    if cfg and "tubule_formation" in cfg:
        tf_cfg = cfg["tubule_formation"].get("thresholds", {})
        score1_min = tf_cfg.get("score1_min_percent", DEFAULT_TUBULE_SCORE1_MIN)
        score2_min = tf_cfg.get("score2_min_percent", DEFAULT_TUBULE_SCORE2_MIN)
        
    if tubule_percent > score1_min:
        return 1
    elif tubule_percent >= score2_min:
        return 2
    else:
        return 3


def calculate_nottingham_grade(
    tubule_score: int,
    pleo_score: int,
    mitotic_score: int,
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[int, int]:
    """
    Calculate Nottingham sum and final Nottingham Histological Grade.
    
    Grade 1: Sum 3-5 (Well differentiated)
    Grade 2: Sum 6-7 (Moderately differentiated)
    Grade 3: Sum 8-9 (Poorly differentiated)
    
    Returns:
        (nottingham_sum, grade)
    """
    nottingham_sum = tubule_score + pleo_score + mitotic_score
    
    g1_max = DEFAULT_GRADE1_MAX_SUM
    g2_max = DEFAULT_GRADE2_MAX_SUM
    if cfg and "nottingham_grading" in cfg:
        ng_cfg = cfg["nottingham_grading"]
        g1_max = ng_cfg.get("grade1_max_sum", DEFAULT_GRADE1_MAX_SUM)
        g2_max = ng_cfg.get("grade2_max_sum", DEFAULT_GRADE2_MAX_SUM)
        
    if nottingham_sum <= g1_max:
        grade = 1
    elif nottingham_sum <= g2_max:
        grade = 2
    else:
        grade = 3
        
    return nottingham_sum, grade


def validate_grading_invariants(
    tubule_score: int,
    pleo_score: int,
    mitotic_score: int,
    nottingham_sum: int,
    grade: int
) -> None:
    """
    The v3/v4 Guard: Ensure all mathematical invariants hold strictly before DB write.
    Raises ValueError if any invariant is violated.
    """
    for name, val in [("tubule_score", tubule_score), ("pleo_score", pleo_score), ("mitotic_score", mitotic_score)]:
        if val not in (1, 2, 3):
            raise ValueError(f"Invariant Violation: {name} must be in [1, 2, 3], got {val}")
            
    expected_sum = tubule_score + pleo_score + mitotic_score
    if nottingham_sum != expected_sum:
        raise ValueError(f"Invariant Violation: nottingham_sum ({nottingham_sum}) != sum of sub-scores ({expected_sum})")
        
    expected_grade = 1 if expected_sum <= 5 else (2 if expected_sum <= 7 else 3)
    if grade != expected_grade:
        raise ValueError(f"Invariant Violation: grade ({grade}) does not match expected Nottingham Grade ({expected_grade}) for sum {expected_sum}")


def calculate_mitotic_score_from_hpfs(
    hpf_mitotic_counts: List[int],
    cfg: Optional[Dict[str, Any]] = None,
    radius_um: float = 262.0
) -> Tuple[int, int]:
    """
    Calculate total mitoses and Nottingham Mitotic Score (1, 2, or 3) across standard HPFs
    using area-normalized density (mitoses/mm²).
    
    Standard Cutoffs for 10 HPFs (0.2157 mm² per HPF, 2.157 mm² total):
    - Score 1: < 3.65 / mm² (< 8 mitoses in 10 standard HPFs)
    - Score 2: 3.65 - 7.30 / mm² (8 - 15 mitoses in 10 standard HPFs)
    - Score 3: >= 7.30 / mm² (>= 16 mitoses in 10 standard HPFs)
    
    Returns:
        (total_mitoses, mitotic_score)
    """
    total_mitoses = sum(hpf_mitotic_counts) if hpf_mitotic_counts else 0
    from pipeline.scoring import compute_nottingham_mitotic_score
    n_hpf = len(hpf_mitotic_counts) if hpf_mitotic_counts else 10
    summary = compute_nottingham_mitotic_score(
        count_total=total_mitoses,
        n_hpf=n_hpf,
        radius_um=radius_um,
        config_dict=cfg
    )
    return total_mitoses, summary["mitotic_score"]


def calculate_mitotic_score_from_detections_and_hpfs(
    detections: List[Dict[str, Any]],
    hpfs: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[int, int]:
    """
    Calculate total mitoses and Nottingham Mitotic Score (1, 2, or 3) across virtual HPFs,
    ensuring that mitoses falling inside overlapping HPF circles are counted ONCE (no double counting).
    Uses standardized area-normalized density (mitoses/mm²).
    
    Returns:
        (unique_total_mitoses, mitotic_score)
    """
    from pipeline.scoring import calculate_hpf_mitosis_counts, compute_nottingham_mitotic_score
    updated_hpfs, unique_total = calculate_hpf_mitosis_counts(detections, hpfs)
    r_um = updated_hpfs[0].get("radius_um", 262.0) if updated_hpfs else 262.0
    summary = compute_nottingham_mitotic_score(
        count_total=unique_total,
        n_hpf=len(updated_hpfs) if updated_hpfs else 10,
        radius_um=r_um,
        config_dict=cfg
    )
    return unique_total, summary["mitotic_score"]


def aggregate_grading_findings(
    tubule_responses: List[Dict[str, Any]],
    pleo_responses: List[Dict[str, Any]],
    mitotic_score: int,
    cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Full end-to-end pure code aggregation pipeline.
    
    Args:
        tubule_responses: List of per-patch dicts with {tubule_percent, tumor_present, confidence, [user_tubule_percent], [user_tumor_present]}
        pleo_responses: List of per-patch dicts with {pleomorphism_score, rationale, confidence, [user_pleo_score]}
        mitotic_score: Confirmed mitotic score (1, 2, or 3)
        cfg: Scoring config dict
        
    Returns:
        Dict containing:
            tubule_percent, tubule_score, pleo_score, mitotic_score,
            nottingham_sum, grade, flags, patch_counts
    """
    if cfg is None:
        cfg = load_scoring_config()
        
    weights_map = cfg.get("grading", {}).get("confidence_weights", DEFAULT_CONF_WEIGHTS)
    min_tumor_patches = cfg.get("grading", {}).get("min_tumor_patches", DEFAULT_MIN_TUMOR_PATCHES)
    max_disp = cfg.get("grading", {}).get("max_disp", DEFAULT_MAX_DISP)
    
    # 1. Filter tumor-containing patches for Tubule assessment (accounting for pathologist overrides)
    tumor_tubule = []
    for r in tubule_responses:
        tumor_present = r.get("user_tumor_present") if r.get("user_tumor_present") is not None else r.get("tumor_present", True)
        if tumor_present:
            tumor_tubule.append(r)
    
    if tumor_tubule:
        tubule_vals = [
            float(r.get("user_tubule_percent") if r.get("user_tubule_percent") is not None else r.get("tubule_percent", 0.0))
            for r in tumor_tubule
        ]
        # Pathologist-reviewed/modified patches receive highest confidence weight
        tubule_w = [
            1.5 if r.get("user_tubule_percent") is not None else weights_map.get(str(r.get("confidence", "medium")).lower(), 1.0)
            for r in tumor_tubule
        ]
        derived_tubule_percent = round(weighted_median(tubule_vals, tubule_w), 1)
    else:
        derived_tubule_percent = 0.0
        
    tubule_score = calculate_tubule_score(derived_tubule_percent, cfg)
    
    # 2. Pleomorphism mode calculation across all valid responses (accounting for pathologist overrides)
    pleo_vals = [
        int(r.get("user_pleo_score") if r.get("user_pleo_score") is not None else r.get("pleomorphism_score", 2))
        for r in pleo_responses
    ]
    pleo_w = [
        1.5 if r.get("user_pleo_score") is not None else weights_map.get(str(r.get("confidence", "medium")).lower(), 1.0)
        for r in pleo_responses
    ]
    
    pleo_score, pleo_dispersion = weighted_mode(pleo_vals, pleo_w, tie_breaker=max)
    
    # 3. Overall Nottingham Grade Calculation
    nottingham_sum, grade = calculate_nottingham_grade(tubule_score, pleo_score, mitotic_score, cfg)
    
    # 4. Quality & Consistency Flags
    flags: List[str] = []
    if len(tumor_tubule) < min_tumor_patches:
        flags.append("insufficient_tumor_patches")
    if pleo_dispersion > max_disp:
        flags.append("pleo_high_variance")
        
    # Validate invariants before returning
    validate_grading_invariants(tubule_score, pleo_score, mitotic_score, nottingham_sum, grade)
    
    return {
        "tubule_percent": derived_tubule_percent,
        "tubule_score": tubule_score,
        "pleo_score": pleo_score,
        "mitotic_score": mitotic_score,
        "nottingham_sum": nottingham_sum,
        "grade": grade,
        "flags": flags,
        "tumor_patch_count": len(tumor_tubule),
        "total_patch_count": max(len(tubule_responses), len(pleo_responses)),
        "pleo_dispersion": round(pleo_dispersion, 3)
    }

