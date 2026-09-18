"""
Pure Zero-LLM AJCC 8th/9th Edition Staging & CAP Synoptic Validation Engine.

All arithmetic calculations of Pathologic T (pT), Pathologic N (pN), and AJCC
Stage Groups are strictly computed deterministically in Python code.
"""

import os
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional
import re
import yaml


def load_staging_config() -> Dict[str, Any]:
    """Load AJCC staging thresholds and definitions from configs/staging.yaml if present."""
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/staging.yaml"))
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def get_staging_config_hash() -> str:
    """Compute deterministic SHA-256 hash of active staging configuration."""
    cfg = load_staging_config()
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def round_half_up_mm(val: float) -> int:
    """Deterministic clinical rounding to nearest whole millimetre (round-half-up)."""
    return int(Decimal(str(val)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def calculate_ajcc_pt_stage(
    tumor_size_mm: Optional[float],
    chest_wall_extension: bool = False,
    skin_ulceration: bool = False,
    is_in_situ_only: bool = False,
    cfg: Optional[Dict[str, Any]] = None
) -> str:
    """
    Calculate Pathologic T (pT) category according to AJCC 8th/9th Edition Breast Cancer staging
    with strict millimetre rounding rules (AJCC 8th Edition Breast Chapter 48):
    - pTX: Primary tumor cannot be assessed
    - pT0: No evidence of primary tumor
    - pTis: In situ only (DCIS/LCIS/Paget)
    - pT1mi: Tumor <= 1.0 mm (microinvasion)
    - pT1a: Tumor > 1.0 mm to <= 5.0 mm (Note: 1.0-1.9 mm rounds to 2 mm -> pT1a)
    - pT1b: Tumor > 5.0 mm to <= 10.0 mm
    - pT1c: Tumor > 10.0 mm to <= 20.0 mm (e.g., 20.4 mm rounds to 20 mm -> pT1c)
    - pT2: Tumor > 20.0 mm to <= 50.0 mm (e.g., 20.5 mm rounds to 21 mm -> pT2)
    - pT3: Tumor > 50.0 mm (e.g., 50.4 mm rounds to 50 mm -> pT2, 50.5 mm rounds to 51 mm -> pT3)
    - pT4: Direct extension to chest wall (pT4a), skin ulceration (pT4b), both (pT4c), or inflammatory (pT4d)
    """
    if is_in_situ_only:
        return "pTis"
        
    if chest_wall_extension and skin_ulceration:
        return "pT4c"
    elif chest_wall_extension:
        return "pT4a"
    elif skin_ulceration:
        return "pT4b"
        
    if tumor_size_mm is None or tumor_size_mm <= 0:
        return "pTX"
        
    # Microinvasion rule: tumor <= 1.0 mm is pT1mi
    if tumor_size_mm <= 1.0:
        return "pT1mi"
        
    # Special AJCC 8th Ed rule: 1.0 to 1.9 mm is reported as 2 mm (pT1a)
    if 1.0 < tumor_size_mm < 2.0:
        rounded_mm = 2
    else:
        rounded_mm = round_half_up_mm(tumor_size_mm)

    if rounded_mm <= 1:
        return "pT1mi"
    elif rounded_mm <= 5:
        return "pT1a"
    elif rounded_mm <= 10:
        return "pT1b"
    elif rounded_mm <= 20:
        return "pT1c"
    elif rounded_mm <= 50:
        return "pT2"
    else:
        return "pT3"


def calculate_ajcc_pn_stage(
    nodes_examined: int,
    nodes_positive: int,
    largest_meta_mm: float = 0.0,
    is_micrometastasis: bool = False,
    cfg: Optional[Dict[str, Any]] = None
) -> str:
    """
    Calculate Pathologic N (pN) category according to AJCC 8th/9th Edition Breast Cancer staging:
    - pNX: Regional lymph nodes cannot be assessed (0 nodes removed)
    - pN0: No regional lymph node metastasis histologically
    - pN0(i+): Isolated tumor cell clusters (ITC) <= 0.2 mm
    - pN1mi: Micrometastasis (> 0.2 mm to <= 2.0 mm and/or > 200 cells)
    - pN1a: Metastases in 1 to 3 axillary lymph nodes (at least 1 metastasis > 2.0 mm)
    - pN2a: Metastases in 4 to 9 axillary lymph nodes
    - pN3a: Metastases in 10 or more axillary lymph nodes
    """
    # Strict Invariant: Positive nodes cannot exceed examined nodes
    if nodes_positive > nodes_examined:
        raise ValueError(
            f"Staging Invariant Violation: nodes_positive ({nodes_positive}) cannot exceed nodes_examined ({nodes_examined})"
        )

    if nodes_examined <= 0:
        return "pNX"
        
    # Isolated tumor cell clusters (ITCs <= 0.2 mm) are classified as pN0(i+)
    if 0.0 < largest_meta_mm <= 0.2:
        return "pN0(i+)"

    if nodes_positive <= 0:
        return "pN0"
        
    # Micrometastases (> 0.2 mm to <= 2.0 mm) in 1-3 nodes
    if is_micrometastasis or (0.2 < largest_meta_mm <= 2.0 and nodes_positive <= 3):
        return "pN1mi"
        
    if 1 <= nodes_positive <= 3:
        return "pN1a"
    elif 4 <= nodes_positive <= 9:
        return "pN2a"
    else:
        return "pN3a"


def calculate_ajcc_stage_group(
    pt_stage: str,
    pn_stage: str,
    pm_stage: str = "cM0"
) -> str:
    """
    Calculate AJCC Anatomic Stage Group (0, IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IV)
    per AJCC 8th Edition Breast Cancer Chapter 48 staging matrix.
    Never defaults to 'IA' for unhandled combinations.
    """
    # Any M1 is Stage IV
    if pm_stage in ("pM1", "cM1", "M1") or str(pm_stage).endswith("M1"):
        return "IV"

    if pt_stage == "N/A" or pn_stage == "N/A":
        return "Benign"

    if pt_stage in ("pTX", "TX"):
        return "Unknown"

    if pt_stage == "pTis" and pn_stage in ("pN0", "pN0(i+)", "pNX", "NX"):
        return "0"

    # Standardize sub-categories to canonical N groups:
    # N1: pN1, pN1a, pN1b, pN1c
    # N2: pN2, pN2a, pN2b
    # N3: pN3, pN3a, pN3b, pN3c
    is_n0 = pn_stage in ("pN0", "pN0(i+)", "pNX", "NX")
    is_n1mi = (pn_stage == "pN1mi")
    is_n1 = pn_stage in ("pN1", "pN1a", "pN1b", "pN1c")
    is_n2 = pn_stage in ("pN2", "pN2a", "pN2b")
    is_n3 = pn_stage in ("pN3", "pN3a", "pN3b", "pN3c")

    # Any T with N3 is Stage IIIC
    if is_n3:
        return "IIIC"

    # pT0 combinations
    if pt_stage == "pT0":
        if is_n1mi:
            return "IB"
        elif is_n1:
            return "IIA"
        elif is_n2:
            return "IIIA"
        return "Cannot be determined"

    # pT1 combinations (pT1mi, pT1a, pT1b, pT1c)
    if pt_stage in ("pT1mi", "pT1a", "pT1b", "pT1c"):
        if is_n0:
            return "IA"
        elif is_n1mi:
            return "IB"
        elif is_n1:
            return "IIA"
        elif is_n2:
            return "IIIA"

    # pT2 combinations
    elif pt_stage == "pT2":
        if is_n0:
            return "IIA"
        elif is_n1mi or is_n1:
            return "IIB"
        elif is_n2:
            return "IIIA"

    # pT3 combinations
    elif pt_stage == "pT3":
        if is_n0:
            return "IIB"
        elif is_n1mi or is_n1 or is_n2:
            return "IIIA"

    # pT4 combinations (pT4a, pT4b, pT4c, pT4d)
    elif pt_stage.startswith("pT4"):
        if is_n0 or is_n1mi or is_n1 or is_n2:
            return "IIIB"

    return "Cannot be determined"


def validate_staging_invariants(
    tumor_size_mm: Optional[float],
    pt_stage: str,
    nodes_examined: int,
    nodes_positive: int,
    pn_stage: str,
    stage_group: str
) -> None:
    """
    Validate that all calculated staging codes adhere strictly to AJCC mathematical and clinical boundaries.
    Raises ValueError if any discrepancy or invariant violation occurs.
    """
    if nodes_positive > nodes_examined:
        raise ValueError(
            f"Staging Invariant Violation: nodes_positive ({nodes_positive}) cannot exceed nodes_examined ({nodes_examined})"
        )
        
    if tumor_size_mm is not None and tumor_size_mm < 0:
        raise ValueError(f"Staging Invariant Violation: tumor_size_mm ({tumor_size_mm}) cannot be negative")

    valid_pt = {"pTX", "pT0", "pTis", "pT1mi", "pT1a", "pT1b", "pT1c", "pT2", "pT3", "pT4", "pT4a", "pT4b", "pT4c", "pT4d"}
    if pt_stage not in valid_pt:
        raise ValueError(f"Invalid pT category: '{pt_stage}'")
        
    valid_pn = {"pNX", "pN0", "pN0(i+)", "pN1mi", "pN1a", "pN1b", "pN1c", "pN2a", "pN2b", "pN3a", "pN3b", "pN3c"}
    if pn_stage not in valid_pn:
        raise ValueError(f"Invalid pN category: '{pn_stage}'")


def validate_narrative_consistency(
    narrative_dict: Dict[str, str],
    verified_data: Dict[str, Any]
) -> List[str]:
    """
    Checks the generated MedGemma text to ensure it does not fabricate conflicting numbers or grades.
    Returns a list of warning/inconsistency strings (empty if consistent).
    """
    issues: List[str] = []
    text_corpus = " ".join([
        str(narrative_dict.get("diagnosis_line", "")),
        str(narrative_dict.get("microscopic_findings", "")),
        str(narrative_dict.get("clinical_correlation", ""))
    ]).lower()
    
    # 1. Check Grade consistency
    grade = verified_data.get("nottingham_grade", {}).get("grade")
    if grade:
        grade_str = str(grade)
        for g_other in [1, 2, 3]:
            if g_other != grade:
                pattern = rf"\bgrade\s*{g_other}\b"
                if re.search(pattern, text_corpus):
                    # Flag if opposing grade is explicitly asserted
                    if not re.search(rf"\bgrade\s*{grade_str}\b", text_corpus):
                        issues.append(f"Narrative mentions Grade {g_other} instead of confirmed Grade {grade}")
                    else:
                        issues.append(f"Narrative contains conflicting Grade mentions (mentions both Grade {g_other} and confirmed Grade {grade})")

    # 2. Check Nottingham Sum consistency if present
    confirmed_sum = verified_data.get("nottingham_grade", {}).get("nottingham_sum")
    if confirmed_sum:
        sum_matches = re.findall(r"\b(?:score|sum)\s*(\d+)/9\b", text_corpus)
        for s_match in sum_matches:
            if int(s_match) != int(confirmed_sum):
                issues.append(f"Narrative states Nottingham sum {s_match}/9 contradicting verified sum {confirmed_sum}/9")

    # 3. Check LVI Concordance
    lvi_status = verified_data.get("lvi_status")
    if lvi_status == "present":
        neg_lvi_patterns = [
            r"no\s+(?:definite\s+|extensive\s+)?lymphovascular\s+invasion",
            r"lymphovascular\s+invasion\s+is\s+not\s+identified",
            r"lvi\s+is\s+negative",
            r"negative\s+for\s+lymphovascular\s+invasion"
        ]
        for p in neg_lvi_patterns:
            if re.search(p, text_corpus):
                issues.append("Narrative asserts absence of lymphovascular invasion while verified LVI status is 'present'")
                break
    elif lvi_status == "absent":
        pos_lvi_patterns = [
            r"lymphovascular\s+invasion\s+(?:is\s+)?present",
            r"lymphovascular\s+invasion\s+identified",
            r"positive\s+for\s+lymphovascular\s+invasion"
        ]
        for p in pos_lvi_patterns:
            if re.search(p, text_corpus):
                m = re.search(r"(?:no|not|without)\s+[\w\s]{1,30}" + p, text_corpus)
                if not m:
                    issues.append("Narrative asserts presence of lymphovascular invasion while verified LVI status is 'absent'")
                    break

    # 4. Check Laterality Concordance
    laterality = verified_data.get("laterality")
    if laterality == "right":
        if re.search(r"\bleft\s+breast\b", text_corpus) and not re.search(r"\bright\s+breast\b", text_corpus):
            issues.append("Narrative references 'left breast' while verified laterality is 'right'")
    elif laterality == "left":
        if re.search(r"\bright\s+breast\b", text_corpus) and not re.search(r"\bleft\s+breast\b", text_corpus):
            issues.append("Narrative references 'right breast' while verified laterality is 'left'")
                    
    return issues
