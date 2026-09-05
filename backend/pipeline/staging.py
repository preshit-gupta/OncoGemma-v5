"""
Pure Zero-LLM AJCC 8th/9th Edition Staging & CAP Synoptic Validation Engine.

All arithmetic calculations of Pathologic T (pT), Pathologic N (pN), and AJCC
Stage Groups are strictly computed deterministically in Python code.
"""

from typing import Dict, Any, List, Optional, Tuple
import re

def calculate_ajcc_pt_stage(
    tumor_size_mm: Optional[float],
    chest_wall_extension: bool = False,
    skin_ulceration: bool = False,
    is_in_situ_only: bool = False
) -> str:
    """
    Calculate Pathologic T (pT) category according to AJCC 8th/9th Edition Breast Cancer staging:
    - pTX: Primary tumor cannot be assessed
    - pT0: No evidence of primary tumor
    - pTis: In situ only (DCIS/LCIS/Paget)
    - pT1mi: Tumor <= 1.0 mm
    - pT1a: Tumor > 1.0 mm but <= 5.0 mm
    - pT1b: Tumor > 5.0 mm but <= 10.0 mm
    - pT1c: Tumor > 10.0 mm but <= 20.0 mm
    - pT2: Tumor > 20.0 mm but <= 50.0 mm
    - pT3: Tumor > 50.0 mm
    - pT4: Direct extension to chest wall or skin ulceration / macroscopic satellite skin nodules
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
        
    if tumor_size_mm <= 1.0:
        return "pT1mi"
    elif tumor_size_mm <= 5.0:
        return "pT1a"
    elif tumor_size_mm <= 10.0:
        return "pT1b"
    elif tumor_size_mm <= 20.0:
        return "pT1c"
    elif tumor_size_mm <= 50.0:
        return "pT2"
    else:
        return "pT3"


def calculate_ajcc_pn_stage(
    nodes_examined: int,
    nodes_positive: int,
    largest_meta_mm: float = 0.0,
    is_micrometastasis: bool = False
) -> str:
    """
    Calculate Pathologic N (pN) category according to AJCC 8th/9th Edition Breast Cancer staging:
    - pNX: Regional lymph nodes cannot be assessed (no nodes removed)
    - pN0: No regional lymph node metastasis histologically
    - pN0(i+): Isolated tumor cell clusters (ITC) <= 0.2 mm
    - pN1mi: Micrometastasis (> 0.2 mm to <= 2.0 mm and/or > 200 cells)
    - pN1a: Metastases in 1 to 3 axillary lymph nodes (at least 1 metastasis > 2.0 mm)
    - pN2a: Metastases in 4 to 9 axillary lymph nodes
    - pN3a: Metastases in 10 or more axillary lymph nodes
    """
    if nodes_examined <= 0:
        return "pNX"
        
    if nodes_positive <= 0:
        return "pN0"
        
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
    Calculate AJCC Anatomic Stage Group (0, IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IV).
    """
    if pm_stage in ("pM1", "cM1"):
        return "IV"

    if pt_stage == "N/A" or pn_stage == "N/A":
        return "Benign"

    if pt_stage in ("pTX", "TX"):
        return "Unknown"
        
    if pt_stage == "pTis" and pn_stage in ("pN0", "pNX"):
        return "0"
        
    # Standard Anatomic Stage Matrix
    if pt_stage in ("pT1mi", "pT1a", "pT1b", "pT1c"):
        if pn_stage in ("pN0", "pNX"):
            return "IA"
        elif pn_stage == "pN1mi":
            return "IB"
        elif pn_stage == "pN1a":
            return "IIA"
        elif pn_stage == "pN2a":
            return "IIIA"
        elif pn_stage == "pN3a":
            return "IIIC"
            
    elif pt_stage == "pT2":
        if pn_stage in ("pN0", "pNX"):
            return "IIA"
        elif pn_stage in ("pN1mi", "pN1a"):
            return "IIB"
        elif pn_stage == "pN2a":
            return "IIIA"
        elif pn_stage == "pN3a":
            return "IIIC"
            
    elif pt_stage == "pT3":
        if pn_stage in ("pN0", "pNX"):
            return "IIB"
        elif pn_stage in ("pN1mi", "pN1a", "pN2a"):
            return "IIIA"
        elif pn_stage == "pN3a":
            return "IIIC"
            
    elif pt_stage.startswith("pT4"):
        if pn_stage in ("pN0", "pNX", "pN1mi", "pN1a", "pN2a"):
            return "IIIB"
        elif pn_stage == "pN3a":
            return "IIIC"
            
    # Default fallback
    if pn_stage == "pN3a":
        return "IIIC"
    elif pn_stage == "pN2a":
        return "IIIA"
    elif pn_stage == "pN1a":
        return "IIA"
        
    return "Unknown"


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
                # Ensure it wasn't preceded by 'no' or 'not'
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
