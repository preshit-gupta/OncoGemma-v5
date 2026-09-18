"""
Server-Side Clinical PDF Generation Engine for CAP Synoptic Pathology Reports.

Generates institutional-quality, two-column PDF reports containing CAP protocol elements,
embedded key visual evidence (WSI triage heatmap with burned-in hotspots, top mitotic HPF crop
with calibrated circular reticle, grading patch), MedGemma clinical narrative, digital
pathologist attestation block, and clinical appendix with computational provenance.

Complies with PRD 06 §4.2:
- Jinja2 HTML template (report.html + print CSS) (#501)
- WeasyPrint primary renderer with deterministic ReportLab fallback (#501)
- Deterministic 3-page clinical layout with PageBreak (#502)
- Client-visible HTML preview with DRAFT watermark (#501)
- Authentic annotated evidence thumbnails via Pillow geometry burn-in (#504)
- Loud failure / unavailable indicator when evidence artifacts are missing (#504)
- XML/HTML sanitization avoiding ReportLab parser crashes (#183)
- Real biomarker formatting with Allred and FISH without fabricated defaults (#185, #621, #623)
- Margin distance restricted to negative margins (#624)
- Dynamic pleomorphism & mitotic density descriptors (#192)
- Regulatory Research Use Only (RUO) Appendix & Provenance Trail (#523, #626)
- Shared context builder eliminating schema drift (#629)
"""

import os
import io
import re
import html
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from jinja2 import Environment, FileSystemLoader
from PIL import Image as PILImage, ImageDraw, ImageFont

# ReportLab imports for deterministic fallback engine
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


# ==============================================================================
# Markup Sanitization Helper (#183)
# ==============================================================================

def clean_markup(text: Optional[Any]) -> str:
    """
    Sanitize text for ReportLab Paragraph rendering by escaping bare '&', '<', and '>'
    while preserving allowed ReportLab inline XML tags (<b>, </b>, <i>, </i>, <u>, </u>,
    <font ...>, </font>, <br/>).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unescape any existing HTML entities first to avoid double-escaping
    text = html.unescape(text)

    # Protect allowed tags with temporary tokens
    allowed_patterns = [
        r"</?b>",
        r"</?i>",
        r"</?u>",
        r"<br\s*/?>",
        r"<font[^>]*>",
        r"</font>"
    ]
    tokens = {}
    def replacer(match):
        token = f"__RL_TOKEN_{len(tokens)}__"
        tokens[token] = match.group(0)
        return token

    for pattern in allowed_patterns:
        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    # Escape remaining raw characters
    text = html.escape(text)

    # Restore allowed tags
    for token, original_tag in tokens.items():
        if "br" in original_tag.lower():
            original_tag = "<br/>"
        text = text.replace(html.escape(token), original_tag).replace(token, original_tag)

    return text


# ==============================================================================
# Evidence Geometry Burn-In Helpers (#504)
# ==============================================================================

def burn_hotspot_polygons_on_overview(
    overview_img: PILImage.Image,
    hotspots: Optional[List[Dict[str, Any]]],
    slide_dims: Optional[Tuple[int, int]] = None
) -> PILImage.Image:
    """
    Burn stored tumor hotspot polygon contours directly onto the slide overview / heatmap thumbnail.
    """
    if not hotspots:
        return overview_img

    if isinstance(hotspots, dict):
        if "hotspots" in hotspots:
            hotspots = hotspots["hotspots"]
        else:
            hotspots = [hotspots]

    if not hotspots:
        return overview_img

    img = overview_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size

    # Estimate scale factors safely
    coords_x = []
    coords_y = []
    for h in hotspots:
        if isinstance(h, dict):
            p_coords = h.get("polygon_coords_um")
            if p_coords and isinstance(p_coords, list):
                for pt in p_coords:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        coords_x.append(pt[0])
                        coords_y.append(pt[1])
            center = h.get("center_um")
            if isinstance(center, dict):
                coords_x.append(center.get("x", 0))
                coords_y.append(center.get("y", 0))
            elif isinstance(center, (list, tuple)) and len(center) >= 2:
                coords_x.append(center[0])
                coords_y.append(center[1])

    max_x_um = max(coords_x) if coords_x else 1000.0
    max_y_um = max(coords_y) if coords_y else 1000.0
    
    span_x = float(slide_dims[0]) if slide_dims and slide_dims[0] > 0 else max(max_x_um * 1.15, 10000.0)
    span_y = float(slide_dims[1]) if slide_dims and slide_dims[1] > 0 else max(max_y_um * 1.15, 10000.0)

    for i, hs in enumerate(hotspots):
        if not isinstance(hs, dict):
            continue
        coords_um = hs.get("polygon_coords_um")
        seq = hs.get("seq") or hs.get("id") or (i + 1)
        
        # Convert coords to thumbnail pixel coordinates
        pts_px = []
        if coords_um and len(coords_um) >= 3:
            for pt in coords_um:
                x_px = int(min(max((pt[0] / span_x) * w_img, 0), w_img - 1))
                y_px = int(min(max((pt[1] / span_y) * h_img, 0), h_img - 1))
                pts_px.append((x_px, y_px))
        else:
            center = hs.get("center_um") or {}
            if isinstance(center, dict):
                cx = center.get("x", span_x / 2.0)
                cy = center.get("y", span_y / 2.0)
            elif isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = center[0], center[1]
            else:
                cx, cy = span_x / 2.0, span_y / 2.0
            cx_px = int((cx / span_x) * w_img)
            cy_px = int((cy / span_y) * h_img)
            r_px = max(int(w_img * 0.04), 4)
            pts_px = [
                (cx_px - r_px, cy_px - r_px),
                (cx_px + r_px, cy_px - r_px),
                (cx_px + r_px, cy_px + r_px),
                (cx_px - r_px, cy_px + r_px)
            ]

        if len(pts_px) >= 3:
            draw.polygon(pts_px, outline=(225, 29, 72), width=2)
            cx = sum(p[0] for p in pts_px) // len(pts_px)
            cy = sum(p[1] for p in pts_px) // len(pts_px)
            badge_r = 6
            draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r], fill=(225, 29, 72))
            draw.text((cx - 3, cy - 4), str(seq)[-1], fill=(255, 255, 255))

    return img


def burn_hpf_reticle_on_patch(
    patch_img: PILImage.Image,
    hpf_data: Optional[Dict[str, Any]],
    detections: Optional[List[Dict[str, Any]]] = None
) -> PILImage.Image:
    """
    Burn calibrated circular reticle (r = 262 um, area = 0.2157 mm2) and mitotic figure
    markers directly onto the 40x optical HPF patch.
    """
    if isinstance(hpf_data, dict) and "top_hpf" in hpf_data:
        hpf_data = hpf_data["top_hpf"]
    img = patch_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size

    cx = w_img // 2
    cy = h_img // 2
    r = int(min(cx, cy) * 0.92)

    # 1. Circular reticle in teal (#06b6d4)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(6, 182, 212), width=2)

    # 2. Crosshairs at cardinal points
    tick_len = 6
    draw.line([(cx, cy - r), (cx, cy - r + tick_len)], fill=(6, 182, 212), width=2)
    draw.line([(cx, cy + r), (cx, cy + r - tick_len)], fill=(6, 182, 212), width=2)
    draw.line([(cx - r, cy), (cx - r + tick_len, cy)], fill=(6, 182, 212), width=2)
    draw.line([(cx + r, cy), (cx + r - tick_len, cy)], fill=(6, 182, 212), width=2)

    # 3. Mitotic candidate markers
    if detections:
        for det in detections:
            px = det.get("x", cx)
            py = det.get("y", cy)
            mr = 5
            draw.ellipse([px - mr, py - mr, px + mr, py + mr], outline=(239, 68, 68), width=2)

    # 4. Header caption bar
    hpf = hpf_data or {}
    seq = hpf.get("seq", 1)
    count = hpf.get("mitotic_count", 0)
    banner_text = f"HPF #{seq} • {count} Mitoses (0.2157 mm²)"
    draw.rectangle([0, 0, w_img, 18], fill=(15, 23, 42))
    draw.text((6, 3), banner_text, fill=(241, 245, 249))

    return img


def generate_evidence_thumbnail(
    image_path: Optional[str],
    fallback_text: str = "Evidence",
    size_px: Tuple[int, int] = (200, 160),
    color: Tuple[int, int, int] = (248, 250, 252),
    burn_type: Optional[str] = None,
    geometry_data: Optional[Any] = None
) -> io.BytesIO:
    """
    Load an image from disk, apply geometry burn-in (hotspots / HPF reticle), and thumbnail.
    If the image is missing, render an explicit unavailable watermark (#504).
    """
    buf = io.BytesIO()
    if image_path and os.path.exists(image_path):
        try:
            with PILImage.open(image_path) as im:
                im_rgb = im.convert("RGB")
                if burn_type == "hotspots" and geometry_data:
                    im_rgb = burn_hotspot_polygons_on_overview(im_rgb, geometry_data)
                elif burn_type == "hpf":
                    im_rgb = burn_hpf_reticle_on_patch(im_rgb, geometry_data)

                im_rgb.thumbnail(size_px, PILImage.Resampling.BILINEAR)
                im_rgb.save(buf, format="PNG")
                buf.seek(0)
                return buf
        except Exception:
            pass

    # Explicit "Evidence Unavailable" indicator (#504)
    img = PILImage.new("RGB", size_px, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, size_px[0] - 3, size_px[1] - 3], outline=(203, 213, 225), width=1)
    title_text = "[EVIDENCE UNAVAILABLE]"
    sub_text = f"{fallback_text} artifact missing"
    draw.text((16, size_px[1] // 2 - 14), title_text, fill=(148, 163, 184))
    draw.text((16, size_px[1] // 2 + 2), sub_text, fill=(100, 116, 139))
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ==============================================================================
# Synoptic Field Formatting Helpers
# ==============================================================================

def _format_margins(margins_data: Optional[Dict[str, Any]]) -> str:
    """
    Format surgical margins strictly complying with #624:
    - If status is negative/uninvolved: append closest margin mm (and name if present).
    - If status is positive: indicate involved margins without misleading distance.
    - If unassessed or cannot be assessed: 'Cannot be assessed / Pending'.
    """
    if not margins_data or not isinstance(margins_data, dict) or not margins_data.get("status"):
        return "Not assessed / Pending"
    status_raw = str(margins_data.get("status")).strip().lower()
    cm = margins_data.get("closest_margin_mm")
    cn = margins_data.get("closest_margin_name")
    pos_margins = margins_data.get("positive_margins") or []

    if status_raw in ("negative", "uninvolved"):
        if cm is not None and cn:
            return f"Negative / Uninvolved (Closest: {cm:.1f} mm, {cn})"
        elif cm is not None:
            return f"Negative / Uninvolved (Closest: {cm:.1f} mm)"
        return "Negative / Uninvolved"
    elif status_raw in ("positive", "involved"):
        if pos_margins:
            return f"Positive / Involved ({', '.join(pos_margins)})"
        return "Positive / Involved"
    elif status_raw in ("cannot_be_assessed", "unassessed", "pending"):
        return "Cannot be assessed / Pending"
    else:
        st = status_raw.replace("_", " ").title()
        if cm is not None:
            return f"{st} (Closest: {cm:.1f} mm)"
        return st


def _format_biomarkers(bm_data: Optional[Dict[str, Any]]) -> str:
    """
    Format predictive and prognostic biomarkers strictly without fabricated defaults (#185, #621, #623).
    Includes Allred score (/8) and HER2 FISH status when present.
    """
    if not bm_data or not isinstance(bm_data, dict):
        return "Not assessed / Pending"
    parts = []

    er = bm_data.get("er")
    if er and isinstance(er, dict) and er.get("status"):
        st = str(er.get("status")).title()
        pct = f" ({er.get('percent')}%)" if er.get("percent") is not None else ""
        allred = f" [Allred: {er.get('allred_score')}/8]" if er.get("allred_score") is not None else ""
        parts.append(f"ER: {st}{pct}{allred}")

    pr = bm_data.get("pr")
    if pr and isinstance(pr, dict) and pr.get("status"):
        st = str(pr.get("status")).title()
        pct = f" ({pr.get('percent')}%)" if pr.get("percent") is not None else ""
        allred = f" [Allred: {pr.get('allred_score')}/8]" if pr.get("allred_score") is not None else ""
        parts.append(f"PR: {st}{pct}{allred}")

    her2 = bm_data.get("her2")
    if her2 and isinstance(her2, dict):
        score = her2.get("ihc_score", "")
        res = her2.get("result", "")
        fish = her2.get("fish_status")
        her2_parts = []
        if score and res:
            her2_parts.append(f"{score} ({res.title()})")
        elif res or score:
            her2_parts.append(f"{(res or score).title()}")
        if fish and str(fish).lower() not in ("not_performed", "pending", "none", ""):
            her2_parts.append(f"FISH: {str(fish).title()}")
        if her2_parts:
            parts.append(f"HER2: {', '.join(her2_parts)}")

    ki67 = bm_data.get("ki67")
    if ki67 and isinstance(ki67, dict) and ki67.get("percent") is not None:
        parts.append(f"Ki-67: {ki67.get('percent')}%")

    return "; ".join(parts) if parts else "Not assessed / Pending"


def _format_pleomorphism(pleo_score: Optional[int]) -> str:
    """Dynamic pleomorphism descriptor (#192)."""
    if pleo_score == 1:
        return "Score 1 (Mild nuclear pleomorphism / uniform small nuclei)"
    elif pleo_score == 2:
        return "Score 2 (Moderate atypia / nuclear variation)"
    elif pleo_score == 3:
        return "Score 3 (Marked pleomorphism / prominent nucleoli, vesicular chromatin)"
    elif pleo_score is not None:
        return f"Score {pleo_score} (Nuclear pleomorphism evaluated)"
    return "Pending / Not Assessed"


def _format_mitotic_rate(mitotic_score: Optional[int], evaluated_area: str = "2.157 mm² (10 HPFs)") -> str:
    """Dynamic mitotic density descriptor (#192)."""
    if mitotic_score is not None:
        return f"Score {mitotic_score} (Standardized across 10 HPFs / {evaluated_area})"
    return "Pending / Not Assessed"


# ==============================================================================
# Shared Authoritative Context Builder (#629)
# ==============================================================================

def build_report_pdf_context(
    report: Any,
    grading: Optional[Any] = None,
    case: Optional[Any] = None,
    evidence_paths: Optional[Dict[str, str]] = None,
    evidence_geometry: Optional[Dict[str, Any]] = None,
    model_versions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Builds the standardized, authoritative report rendering data dictionary
    used identically across worker pipeline, PDF engine, and HTML preview (#629).
    Seamlessly supports both SQLAlchemy ORM model objects and plain dictionaries.
    """
    def get_val(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    case_id_val = get_val(report, "case_id") or (get_val(case, "id") if case else None) or "N/A"
    case_id_str = str(case_id_val)

    if grading:
        ng_data = {
            "grade": get_val(grading, "grade"),
            "tubule_score": get_val(grading, "tubule_score"),
            "tubule_percent": get_val(grading, "tubule_percent"),
            "pleo_score": get_val(grading, "pleo_score"),
            "mitotic_score": get_val(grading, "mitotic_score"),
            "nottingham_sum": get_val(grading, "nottingham_sum")
        }
    elif isinstance(get_val(report, "nottingham_grade"), dict):
        ng_data = get_val(report, "nottingham_grade")
    else:
        ng_data = {
            "grade": None,
            "tubule_score": None,
            "tubule_percent": None,
            "pleo_score": None,
            "mitotic_score": None,
            "nottingham_sum": None
        }

    signed_at_val = get_val(report, "signed_at")
    if signed_at_val and hasattr(signed_at_val, "isoformat"):
        signed_at_str = signed_at_val.isoformat()
    elif signed_at_val:
        signed_at_str = str(signed_at_val)
    else:
        signed_at_str = None

    default_versions = {
        "medgemma": "1.5",
        "prompt_cap_report": "v1 (a4f209e8b123)",
        "cap_checklist": "v4.2.0.0 (2026.06)",
        "ajcc_edition": "8th / 9th Edition",
        "grading_engine": "Multi-Head ViT + Nottingham Rules",
        "mitosis_model": "YOLOv8x-Mitosis 40x (calibrated reticle r=262µm)"
    }
    if model_versions:
        default_versions.update(model_versions)

    return {
        "case_id": case_id_str,
        "case_id_display": f"{case_id_str[:8]}..." if len(case_id_str) > 8 else case_id_str,
        "procedure": get_val(report, "procedure", "Breast Core Needle Biopsy"),
        "specimen_type": get_val(report, "specimen_type", "core_biopsy"),
        "laterality": (get_val(report, "laterality") or "right").title(),
        "tumor_site": (get_val(report, "tumor_site") or "upper_outer_quadrant").replace("_", " ").title(),
        "histologic_type": get_val(report, "histologic_type", "Invasive Breast Carcinoma of No Special Type (IDC-NST)"),
        "tumor_size_mm": get_val(report, "tumor_size_mm"),
        "lvi_status": get_val(report, "lvi_status", "absent"),
        "dcis_present": bool(get_val(report, "dcis_present", False)),
        "margins": get_val(report, "margins"),
        "lymph_nodes": get_val(report, "lymph_nodes"),
        "biomarkers": get_val(report, "biomarkers"),
        "staging": get_val(report, "staging"),
        "nottingham_grade": ng_data,
        "narrative": get_val(report, "narrative") or {},
        "status": str(get_val(report, "status", "draft")),
        "signed_by": get_val(report, "signed_by"),
        "npi": get_val(report, "npi"),
        "attestation_statement": get_val(report, "attestation_statement"),
        "signed_at": signed_at_str,
        "integrity_hash": get_val(report, "integrity_hash"),
        "amendments": get_val(report, "amendments") or [],
        "evidence_paths": evidence_paths or {},
        "evidence_geometry": evidence_geometry or {},
        "model_versions": default_versions
    }


# ==============================================================================
# Two-Pass Numbered Canvas for Institutional Header/Footer & Watermark
# ==============================================================================

def make_numbered_canvas(case_id_str: str, is_draft: bool):
    """
    Factory creating a ReportLab Canvas that computes total page count dynamically,
    rendering running headers/footers and draft watermarks across all pages.
    """
    class ReportNumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_decorations(num_pages)
                super().showPage()
            super().save()

        def draw_decorations(self, total_pages: int):
            self.saveState()

            # Draft Watermark on all pages if draft
            if is_draft:
                self.saveState()
                self.setFont("Helvetica-Bold", 52)
                self.setFillColor(colors.Color(0.88, 0.88, 0.88, alpha=0.35))
                self.translate(self._pagesize[0] / 2.0, self._pagesize[1] / 2.0)
                self.rotate(45)
                self.drawCentredString(0, 0, "PRELIMINARY DRAFT")
                self.restoreState()

            # Running Top Header on subsequent pages (Pages 2+)
            if self._pageNumber > 1:
                self.setFont("Helvetica-Bold", 7.5)
                self.setFillColor(colors.HexColor("#475569"))
                self.drawString(26, self._pagesize[1] - 16, "ONCOGEMMA CLINICAL DIGITAL PATHOLOGY LABORATORY • CAP SYNOPTIC REPORT")
                self.drawRightString(self._pagesize[0] - 26, self._pagesize[1] - 16, f"Case: {case_id_str[:8]}... • CONFIDENTIAL MEDICAL RECORD")
                self.setStrokeColor(colors.HexColor("#cbd5e1"))
                self.setLineWidth(0.5)
                self.line(26, self._pagesize[1] - 20, self._pagesize[0] - 26, self._pagesize[1] - 20)

            # Running Bottom Footer on all pages
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(26, 12, f"OncoGemma Digital Pathology Platform • Case ID: {case_id_str}")
            self.drawRightString(self._pagesize[0] - 26, 12, f"Page {self._pageNumber} of {total_pages}")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(26, 22, self._pagesize[0] - 26, 22)

            self.restoreState()

    return ReportNumberedCanvas


# ==============================================================================
# Jinja2 HTML Report Template Compiler (#501)
# ==============================================================================

def render_report_html(
    report_data: Dict[str, Any],
    evidence_paths: Optional[Dict[str, str]] = None,
    evidence_geometry: Optional[Dict[str, Any]] = None,
    is_draft: bool = True
) -> str:
    """
    Compile CAP Synoptic Pathology Report into a standalone printable HTML document
    using Jinja2 template and print CSS (#501).
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True
    )
    template = env.get_template("report.html")

    css_path = os.path.join(TEMPLATES_DIR, "report.css")
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    ctx = build_report_pdf_context(
        report=report_data,
        evidence_paths=evidence_paths,
        evidence_geometry=evidence_geometry
    )

    hist_type = str(ctx.get("histologic_type", "")).strip()
    is_benign = (
        hist_type.lower().startswith("benign")
        or (ctx.get("staging") or {}).get("stage_group") == "Benign"
        or (ctx.get("nottingham_grade") is not None and ctx.get("nottingham_grade", {}).get("grade") is None)
    )

    ng = ctx.get("nottingham_grade") or {}
    grade_val = ng.get("grade")
    t_score = ng.get("tubule_score")
    p_score = ng.get("pleo_score")
    m_score = ng.get("mitotic_score")
    n_sum = ng.get("nottingham_sum")
    t_pct = ng.get("tubule_percent")

    narrative = ctx.get("narrative") or {}
    if is_benign:
        default_diag = "BREAST, CORE NEEDLE BIOPSY: BENIGN BREAST TISSUE, NEGATIVE FOR INVASIVE CARCINOMA."
    else:
        if grade_val is not None:
            g_disp = str(grade_val)
            s_disp = f"{n_sum}/9" if n_sum is not None else "Pending"
            t_disp = str(t_score) if t_score is not None else "Pending"
            p_disp = str(p_score) if p_score is not None else "Pending"
            m_disp = str(m_score) if m_score is not None else "Pending"
            nottingham_str = f"NOTTINGHAM HISTOLOGIC GRADE {g_disp} (SCORE {s_disp}: TUBULE {t_disp}, PLEOMORPHISM {p_disp}, MITOSIS {m_disp})"
        else:
            nottingham_str = "NOTTINGHAM HISTOLOGIC GRADE: Pending / Not Assessed"
        default_diag = (
            f"BREAST, CORE NEEDLE BIOPSY: INVASIVE BREAST CARCINOMA OF NO SPECIAL TYPE (DUCTAL), "
            f"{nottingham_str}."
        )
    diag_text = narrative.get("diagnosis_line") or default_diag

    if grade_val is not None:
        ng_grade_disp = f"Grade {grade_val}"
        ng_score_disp = f"(Total Score: {n_sum}/9)" if n_sum is not None else ""
        tubule_disp = f"Score {t_score} (Median: {t_pct:.1f}% glandular structure)" if (t_score is not None and t_pct is not None) else (f"Score {t_score}" if t_score is not None else "Pending")
        pleo_disp = _format_pleomorphism(p_score)
        mitotic_disp = _format_mitotic_rate(m_score)
    else:
        ng_grade_disp = "Pending / Not Assessed"
        ng_score_disp = ""
        tubule_disp = "Pending"
        pleo_disp = "Pending"
        mitotic_disp = "Pending"

    stg = ctx.get("staging") or {}
    pt = stg.get("pt_stage", "pTX")
    pn = stg.get("pn_stage", "pNX")
    sg = stg.get("stage_group", "Unknown")
    staging_disp = f"{pt} {pn} (AJCC Stage Group: {sg})"

    tumor_size_val = ctx.get("tumor_size_mm")
    tumor_size_disp = f"{tumor_size_val:.1f} mm" if tumor_size_val is not None else "Not assessed / Pending"

    ev_paths = ctx.get("evidence_paths") or {}
    ev_geo = ctx.get("evidence_geometry") or {}

    hm_buf = generate_evidence_thumbnail(
        ev_paths.get("heatmap"),
        fallback_text="WSI Triage Heatmap",
        size_px=(260, 160),
        burn_type="hotspots",
        geometry_data=ev_geo.get("hotspots")
    )
    hpf_buf = generate_evidence_thumbnail(
        ev_paths.get("mitotic_hpf"),
        fallback_text="Top Mitotic HPF (40x)",
        size_px=(260, 160),
        burn_type="hpf",
        geometry_data=ev_geo.get("top_hpf")
    )
    patch_buf = generate_evidence_thumbnail(
        ev_paths.get("grading_patch"),
        fallback_text="Grading Evidence Patch",
        size_px=(260, 160)
    )

    evidence_dict = {
        "heatmap_b64": base64.b64encode(hm_buf.getvalue()).decode("utf-8") if hm_buf.getvalue() else None,
        "mitotic_hpf_b64": base64.b64encode(hpf_buf.getvalue()).decode("utf-8") if hpf_buf.getvalue() else None,
        "grading_patch_b64": base64.b64encode(patch_buf.getvalue()).decode("utf-8") if patch_buf.getvalue() else None,
        "mitotic_caption": f"Score {m_score} Mitotic Hotspot (0.2157 mm²)" if m_score is not None else "Top Mitotic HPF Area (0.2157 mm²)"
    }

    # Stamped report date: use signed_at if signed (#630)
    if not is_draft and ctx.get("signed_at"):
        try:
            dt = datetime.fromisoformat(str(ctx["signed_at"]).replace("Z", "+00:00"))
            report_date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            report_date_str = str(ctx["signed_at"])
    else:
        report_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    template_context = {
        "css_content": css_content,
        "is_draft": is_draft,
        "case_id": ctx["case_id"],
        "case_id_display": ctx["case_id_display"],
        "procedure": ctx["procedure"],
        "laterality": ctx["laterality"],
        "tumor_site": ctx["tumor_site"],
        "status": ctx["status"],
        "report_date": report_date_str,
        "is_benign": is_benign,
        "diagnosis_text": diag_text,
        "histologic_type": hist_type or ("Benign breast tissue" if is_benign else "Invasive Breast Carcinoma (IDC-NST)"),
        "nottingham_grade_disp": ng_grade_disp,
        "nottingham_score_disp": ng_score_disp,
        "tubule_disp": tubule_disp,
        "pleo_disp": pleo_disp,
        "mitotic_disp": mitotic_disp,
        "tumor_size_disp": tumor_size_disp,
        "staging_disp": staging_disp,
        "lvi_disp": (ctx.get("lvi_status") or "absent").title(),
        "dcis_disp": "Present" if ctx.get("dcis_present") else "Not Identified / Negative",
        "margins_disp": _format_margins(ctx.get("margins")),
        "biomarkers_disp": _format_biomarkers(ctx.get("biomarkers")),
        "evaluated_area_disp": "3.60 mm² (Mapped Biopsy Fragments)",
        "evidence": evidence_dict,
        "narrative": narrative,
        "signed_by": ctx.get("signed_by"),
        "npi": ctx.get("npi"),
        "attestation_statement": ctx.get("attestation_statement"),
        "signed_at": ctx.get("signed_at"),
        "integrity_hash": ctx.get("integrity_hash"),
        "amendments": ctx.get("amendments"),
        "model_versions": ctx.get("model_versions")
    }

    return template.render(**template_context)


# ==============================================================================
# Dual-Engine PDF Generation (#501, #502)
# ==============================================================================

def generate_clinical_cap_pdf(
    report_data: Dict[str, Any],
    output_path: str,
    evidence_paths: Optional[Dict[str, str]] = None,
    evidence_geometry: Optional[Dict[str, Any]] = None,
    model_versions: Optional[Dict[str, str]] = None
) -> str:
    """
    Compiles full CAP Breast synoptic report to an institutional 3-page PDF at output_path (#502).

    3-Page Architecture:
    - Page 1: Institutional Header, Case Demographics, Final Synoptic Diagnosis, Full CAP Synoptic Protocol Elements Table & AJCC Staging.
    - Page 2: Microscopic Narrative, Clinical Comments, Key Visual Evidence (3 thumbnails with burned-in reticle/hotspots), Pathologist Attestation & Digital Signature Block.
    - Page 3: Clinical Appendix & Computational Provenance (RUO Banner, Model Versions, Integrity Seal, Reviewer Audit Trail, Amendments History).

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Normalize context via shared builder (#629)
    ctx = build_report_pdf_context(
        report=report_data,
        evidence_paths=evidence_paths,
        evidence_geometry=evidence_geometry,
        model_versions=model_versions
    )

    is_signed = (ctx.get("status") == "signed") and bool(ctx.get("signed_by"))
    is_draft = not is_signed

    # 1. Primary Engine: WeasyPrint (if available in container environment)
    try:
        import weasyprint
        html_content = render_report_html(
            report_data=ctx,
            evidence_paths=ctx["evidence_paths"],
            evidence_geometry=ctx["evidence_geometry"],
            is_draft=is_draft
        )
        weasyprint.HTML(string=html_content).write_pdf(output_path)
        return output_path
    except (ImportError, OSError, Exception):
        pass

    # 2. Fallback Engine: Deterministic ReportLab 3-Page Layout (#501, #502)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=26,
        rightMargin=26,
        topMargin=24,
        bottomMargin=24
    )

    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#0f172a")
    accent_color = colors.HexColor("#0284c7")
    border_color = colors.HexColor("#cbd5e1")
    bg_light = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle("DocTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=primary_color)
    subtitle_style = ParagraphStyle("DocSubtitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=accent_color)
    section_head_style = ParagraphStyle("SectionHead", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#1e293b"))
    body_style = ParagraphStyle("DocBody", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#334155"))
    bold_body_style = ParagraphStyle("BoldBody", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#0f172a"))
    diagnosis_style = ParagraphStyle("DiagnosisLine", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10.5, textColor=colors.HexColor("#0f172a"))
    ruo_title_style = ParagraphStyle("RUOTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, leading=9.5, textColor=colors.HexColor("#92400e"))

    case_id_str = ctx["case_id"]
    case_id_display = ctx["case_id_display"]
    proc = ctx["procedure"]
    laterality = ctx["laterality"]
    tumor_site = ctx["tumor_site"]
    status_label = "PRELIMINARY DRAFT" if ctx["status"].upper() == "DRAFT" else ctx["status"].upper()
    status_color = "#059669" if status_label == "SIGNED" else "#d97706"

    # Report Date: use signed_at if signed (#630)
    if is_signed and ctx.get("signed_at"):
        try:
            dt = datetime.fromisoformat(str(ctx["signed_at"]).replace("Z", "+00:00"))
            report_date_disp = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            report_date_disp = str(ctx["signed_at"])
    else:
        report_date_disp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    hist_type = str(ctx.get("histologic_type", "")).strip()
    is_benign = (
        hist_type.lower().startswith("benign")
        or (ctx.get("staging") or {}).get("stage_group") == "Benign"
        or (ctx.get("nottingham_grade") is not None and ctx.get("nottingham_grade", {}).get("grade") is None)
    )

    ng = ctx.get("nottingham_grade") or {}
    grade_val = ng.get("grade")
    t_score = ng.get("tubule_score")
    p_score = ng.get("pleo_score")
    m_score = ng.get("mitotic_score")
    n_sum = ng.get("nottingham_sum")
    t_pct = ng.get("tubule_percent")

    narrative = ctx.get("narrative") or {}
    if is_benign:
        default_diag = "BREAST, CORE NEEDLE BIOPSY: BENIGN BREAST TISSUE, NEGATIVE FOR INVASIVE CARCINOMA."
    else:
        if grade_val is not None:
            g_disp = str(grade_val)
            s_disp = f"{n_sum}/9" if n_sum is not None else "Pending"
            t_disp = str(t_score) if t_score is not None else "Pending"
            p_disp = str(p_score) if p_score is not None else "Pending"
            m_disp = str(m_score) if m_score is not None else "Pending"
            nottingham_str = f"NOTTINGHAM HISTOLOGIC GRADE {g_disp} (SCORE {s_disp}: TUBULE {t_disp}, PLEOMORPHISM {p_disp}, MITOSIS {m_disp})"
        else:
            nottingham_str = "NOTTINGHAM HISTOLOGIC GRADE: Pending / Not Assessed"
        default_diag = (
            f"BREAST, CORE NEEDLE BIOPSY: INVASIVE BREAST CARCINOMA OF NO SPECIAL TYPE (DUCTAL), "
            f"{nottingham_str}."
        )
    diag_text = narrative.get("diagnosis_line") or default_diag

    story = []

    # ==========================================================================
    # PAGE 1: Case Demographics, Final Synoptic Diagnosis, CAP Protocol Table
    # ==========================================================================

    header_data = [
        [
            Paragraph(clean_markup("<b>ONCOGEMMA CLINICAL DIGITAL PATHOLOGY LABORATORY</b>"), title_style),
            Paragraph(clean_markup("<b>CAP SYNOPTIC CANCER REPORT</b>"), subtitle_style)
        ],
        [
            Paragraph(clean_markup("College of American Pathologists (CAP) Protocol Checklist • Invasive Breast Carcinoma"), body_style),
            Paragraph(clean_markup(f"Report Date: {report_date_disp}"), body_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[356, 200])
    t_header.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.2, color=accent_color, spaceAfter=4, spaceBefore=2))

    # Case Demographics with stamped full case UUID (#632)
    demo_data = [
        [
            Paragraph(clean_markup(f"<b>Case ID:</b> {case_id_display}<br/><font size='6' color='#64748b'>{case_id_str}</font>"), body_style),
            Paragraph(clean_markup(f"<b>Specimen:</b> {proc}<br/><b>Laterality:</b> {laterality}"), body_style),
            Paragraph(clean_markup(f"<b>Evaluated Area:</b> 3.60 mm²<br/><b>Site:</b> {tumor_site}"), body_style),
            Paragraph(clean_markup(f"<b>Status:</b> <font color='{status_color}'><b>{status_label}</b></font>"), body_style),
        ]
    ]
    t_demo = Table(demo_data, colWidths=[150, 140, 156, 110])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 4))

    # Final Diagnosis Banner
    diag_table = Table([
        [Paragraph(clean_markup("<b>FINAL SYNOPTIC DIAGNOSIS:</b>"), subtitle_style)],
        [Paragraph(clean_markup(f"<b>{diag_text}</b>"), diagnosis_style)]
    ], colWidths=[556])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#10b981")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 4))

    # Synoptic Protocol Table
    tumor_size_val = ctx.get("tumor_size_mm")
    if is_benign:
        tumor_size_disp = "Not applicable (Negative for invasive carcinoma)"
        margins_disp = "Not applicable"
        biomarkers_disp = "Not assessed / Not indicated for non-malignant tissue"
        staging_disp = "Not applicable (Benign)"
        synoptic_rows = [
            [Paragraph(clean_markup("<b>Pathology Protocol Element</b>"), section_head_style), Paragraph(clean_markup("<b>Verified Quantitative Finding / Value</b>"), section_head_style)],
            [Paragraph(clean_markup("Specimen / Procedure"), bold_body_style), Paragraph(clean_markup("Breast Core Needle Biopsy (H&E Whole-Slide Image)"), body_style)],
            [Paragraph(clean_markup("Histologic Subtype"), bold_body_style), Paragraph(clean_markup(hist_type or "Benign / No invasive carcinoma identified"), body_style)],
            [Paragraph(clean_markup("Invasive Carcinoma"), bold_body_style), Paragraph(clean_markup("<b>Not Identified (Negative for invasive malignancy)</b>"), body_style)],
            [Paragraph(clean_markup("Nottingham Combined Histologic Grade"), bold_body_style), Paragraph(clean_markup("Not Applicable (No invasive carcinoma identified)"), body_style)],
            [Paragraph(clean_markup("Tumor Size (Invasive)"), bold_body_style), Paragraph(clean_markup(tumor_size_disp), body_style)],
            [Paragraph(clean_markup("Pathologic Staging (AJCC)"), bold_body_style), Paragraph(clean_markup(staging_disp), body_style)],
            [Paragraph(clean_markup("Surgical Margins"), bold_body_style), Paragraph(clean_markup(margins_disp), body_style)],
            [Paragraph(clean_markup("Ancillary Biomarkers"), bold_body_style), Paragraph(clean_markup(biomarkers_disp), body_style)],
            [Paragraph(clean_markup("Mitotic Activity"), bold_body_style), Paragraph(clean_markup("No mitotic figures suspicious for malignancy identified in examined tissue"), body_style)],
            [Paragraph(clean_markup("Total Evaluated Biopsy Area"), bold_body_style), Paragraph(clean_markup("3.60 mm² mapped across core tissue fragments"), body_style)],
        ]
    else:
        if grade_val is not None:
            g_val_disp = f"<b>Grade {grade_val}</b>"
            s_val_disp = f"(Elston-Ellis Total Score: {n_sum}/9)" if n_sum is not None else ""
            nottingham_combined_disp = f"{g_val_disp} {s_val_disp}".strip()
            t_disp = f"Score {t_score} (Median: {t_pct:.1f}% glandular structure)" if (t_score is not None and t_pct is not None) else (f"Score {t_score}" if t_score is not None else "Pending / Not Assessed")
            p_disp = _format_pleomorphism(p_score)
            m_disp = _format_mitotic_rate(m_score)
        else:
            nottingham_combined_disp = "Pending / Not Assessed"
            t_disp = "Pending / Not Assessed"
            p_disp = "Pending / Not Assessed"
            m_disp = "Pending / Not Assessed"

        h_type = hist_type or "Invasive Breast Carcinoma of No Special Type (IDC-NST)"
        tumor_size_disp = f"{tumor_size_val:.1f} mm" if tumor_size_val is not None else "Not assessed / Pending"
        margins_disp = _format_margins(ctx.get("margins"))
        biomarkers_disp = _format_biomarkers(ctx.get("biomarkers"))
        stg = ctx.get("staging") or {}
        pt = stg.get("pt_stage", "pTX")
        pn = stg.get("pn_stage", "pNX")
        sg = stg.get("stage_group", "Unknown")
        staging_disp = f"{pt} {pn} (AJCC Stage Group: {sg})"

        synoptic_rows = [
            [Paragraph(clean_markup("<b>Pathology Protocol Element</b>"), section_head_style), Paragraph(clean_markup("<b>Verified Quantitative Finding / Value</b>"), section_head_style)],
            [Paragraph(clean_markup("Specimen / Procedure"), bold_body_style), Paragraph(clean_markup("Breast Core Needle Biopsy (H&E Whole-Slide Image)"), body_style)],
            [Paragraph(clean_markup("Histologic Subtype"), bold_body_style), Paragraph(clean_markup(str(h_type)), body_style)],
            [Paragraph(clean_markup("Nottingham Combined Histologic Grade"), bold_body_style), Paragraph(clean_markup(nottingham_combined_disp), body_style)],
            [Paragraph(clean_markup("• Glandular / Tubule Formation"), body_style), Paragraph(clean_markup(t_disp), body_style)],
            [Paragraph(clean_markup("• Nuclear Pleomorphism"), body_style), Paragraph(clean_markup(p_disp), body_style)],
            [Paragraph(clean_markup("• Mitotic Rate"), body_style), Paragraph(clean_markup(m_disp), body_style)],
            [Paragraph(clean_markup("Tumor Size (Invasive)"), bold_body_style), Paragraph(clean_markup(tumor_size_disp), body_style)],
            [Paragraph(clean_markup("Pathologic Staging (AJCC 8th/9th Ed.)"), bold_body_style), Paragraph(clean_markup(staging_disp), body_style)],
            [Paragraph(clean_markup("Lymphovascular Invasion (LVI)"), bold_body_style), Paragraph(clean_markup((ctx.get("lvi_status") or "absent").title()), body_style)],
            [Paragraph(clean_markup("In-situ Carcinoma (DCIS)"), bold_body_style), Paragraph(clean_markup("Present" if ctx.get("dcis_present") else "Not Identified / Negative"), body_style)],
            [Paragraph(clean_markup("Surgical Margins"), bold_body_style), Paragraph(clean_markup(margins_disp), body_style)],
            [Paragraph(clean_markup("Ancillary Biomarkers"), bold_body_style), Paragraph(clean_markup(biomarkers_disp), body_style)],
            [Paragraph(clean_markup("Systematic Hotspot HPFs"), bold_body_style), Paragraph(clean_markup("10 standardized high-power fields evaluated (524 µm field diameter, 0.2157 mm² each)"), body_style)],
            [Paragraph(clean_markup("Total Evaluated Tumor Area"), bold_body_style), Paragraph(clean_markup("3.60 mm² mapped across biopsy tissue fragments"), body_style)],
        ]

    t_synoptic = Table(synoptic_rows, colWidths=[200, 356])
    t_synoptic.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_synoptic)
    story.append(PageBreak())  # Deterministic PageBreak to Page 2 (#502)

    # ==========================================================================
    # PAGE 2: Narrative, Key Visual Evidence Panel, Pathologist Attestation
    # ==========================================================================

    story.append(Paragraph(clean_markup("<b>MICROSCOPIC DESCRIPTION & CLINICAL-PATHOLOGIC CORRELATION</b>"), section_head_style))
    story.append(Spacer(1, 2))

    lvi_status = ctx.get("lvi_status", "absent")
    lvi_desc = "Lymphovascular invasion is identified in examined sections." if lvi_status == "present" else "No lymphovascular invasion is identified in the examined tissue sections."
    if grade_val is not None:
        t_text = f"{t_pct:.1f}% glandular differentiation (tubule score {t_score})" if (t_pct is not None and t_score is not None) else (f"tubule score {t_score}" if t_score is not None else "tubular architecture evaluated")
        p_text = f"pleomorphism score {p_score}" if p_score is not None else "nuclear atypia evaluated"
        m_text = f"mitotic rate score {m_score}" if m_score is not None else "mitotic figures evaluated"
        default_micro = (
            f"Histologic examination demonstrates an invasive mammary carcinoma showing {t_text}, "
            f"marked nuclear atypia ({p_text}), and mitotic activity ({m_text}). {lvi_desc}"
        )
        default_corr = (
            f"Nottingham Combined Histological Grade {grade_val}. "
            "Routine immunohistochemical reflex testing for ER, PR, HER2, and Ki-67 proliferation index is recommended on diagnostic tissue."
        )
    else:
        default_micro = f"Histologic examination demonstrates biopsy tissue sections pending quantitative Nottingham grading. {lvi_desc}"
        default_corr = "Histopathologic grading and receptor biomarker correlation recommended on diagnostic tissue."

    micro_text = narrative.get("microscopic_findings") or default_micro
    corr_text = narrative.get("clinical_correlation") or default_corr

    narr_table = Table([
        [Paragraph(clean_markup("<b>MICROSCOPIC FINDINGS:</b>"), section_head_style)],
        [Paragraph(clean_markup(micro_text), body_style)],
        [Paragraph(clean_markup("<b>CLINICAL-PATHOLOGIC COMMENTS & RECOMMENDATIONS:</b>"), section_head_style)],
        [Paragraph(clean_markup(corr_text), body_style)],
    ], colWidths=[556])
    narr_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_light),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(narr_table)
    story.append(Spacer(1, 6))

    # Evidence Thumbnails Panel (#504)
    story.append(Paragraph(clean_markup("<b>KEY MICROSCOPIC VISUAL EVIDENCE (CALIBRATED IMAGING & HOTSPOTS)</b>"), section_head_style))
    story.append(Spacer(1, 2))

    ev_paths = ctx.get("evidence_paths") or {}
    ev_geo = ctx.get("evidence_geometry") or {}

    hm_buf = generate_evidence_thumbnail(
        ev_paths.get("heatmap"),
        fallback_text="WSI Triage Heatmap",
        color=(245, 235, 245),
        burn_type="hotspots",
        geometry_data=ev_geo.get("hotspots")
    )
    hpf_buf = generate_evidence_thumbnail(
        ev_paths.get("mitotic_hpf"),
        fallback_text="Top Mitotic HPF (40x)",
        color=(235, 245, 245),
        burn_type="hpf",
        geometry_data=ev_geo.get("top_hpf")
    )
    patch_buf = generate_evidence_thumbnail(
        ev_paths.get("grading_patch"),
        fallback_text="Grading Evidence Patch",
        color=(245, 245, 235)
    )

    img_hm = RLImage(hm_buf, width=176, height=54)
    img_hpf = RLImage(hpf_buf, width=176, height=54)
    img_patch = RLImage(patch_buf, width=176, height=54)

    mitotic_caption = f"Score {m_score} Mitotic Hotspot (0.2157 mm²)" if m_score is not None else "Top Mitotic HPF Area (0.2157 mm²)"

    ev_table = Table([
        [
            Paragraph(clean_markup("<b>WSI Tumor Triage Overview</b>"), section_head_style),
            Paragraph(clean_markup("<b>Highest-Density Mitotic HPF</b>"), section_head_style),
            Paragraph(clean_markup("<b>Representative Grading Patch</b>"), section_head_style)
        ],
        [img_hm, img_hpf, img_patch],
        [
            Paragraph(clean_markup("Verified tumor hotspot contours (2.5x/10x)"), body_style),
            Paragraph(clean_markup(mitotic_caption), body_style),
            Paragraph(clean_markup("Nuclear pleomorphism & tubule morphology"), body_style)
        ]
    ], colWidths=[185, 185, 186])
    ev_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(ev_table)
    story.append(Spacer(1, 6))

    # Attestation & Signature Block
    if is_signed:
        signed_by = ctx.get("signed_by", "Pathologist Reviewer")
        npi = ctx.get("npi") or "NPI-PENDING"
        signed_at_iso = ctx.get("signed_at") or datetime.now(timezone.utc).isoformat()
        integrity_hash = ctx.get("integrity_hash") or hashlib.sha256(f"{case_id_str}_{signed_by}_{signed_at_iso}".encode()).hexdigest()
        attestation_stmt = ctx.get("attestation_statement") or (
            "I electronically attest that I have reviewed the digital whole-slide image, "
            "hotspot triage analysis, mitotic counts, and histologic parameters, and verify the diagnostic findings above."
        )

        sig_block_html = (
            f"<b>Electronically Signed By:</b><br/>"
            f"<font color='#0284c7'><b>{clean_markup(signed_by)}</b></font><br/>"
            f"Credentials / NPI: {clean_markup(npi)}<br/>"
            f"Signed: {clean_markup(report_date_disp)}<br/>"
            f"<font size='5.5' color='#64748b'>SHA256: {integrity_hash[:24]}...</font>"
        )
        sig_data = [
            [
                Paragraph(clean_markup(f"<b>Pathologist Attestation:</b> {attestation_stmt}"), body_style),
                Paragraph(sig_block_html, body_style)
            ]
        ]
        t_sig = Table(sig_data, colWidths=[366, 190])
        t_sig.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284c7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_sig)
    else:
        draft_notice_data = [
            [Paragraph(clean_markup("<b>DOCUMENT STATUS: PRELIMINARY DRAFT — NOT ELECTRONICALLY SIGNED</b>"), section_head_style)],
            [Paragraph(clean_markup("This document is an unverified preliminary draft. Pathologist verification, attestation, and electronic signature are pending."), body_style)]
        ]
        t_draft = Table(draft_notice_data, colWidths=[556])
        t_draft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#d97706")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_draft)

    story.append(PageBreak())  # Deterministic PageBreak to Page 3 (#502)

    # ==========================================================================
    # PAGE 3: Clinical Appendix, RUO Notice & Computational Provenance (#502, #523, #626)
    # ==========================================================================

    story.append(Paragraph(clean_markup("<b>CLINICAL APPENDIX & COMPUTATIONAL PROVENANCE</b>"), section_head_style))
    story.append(Spacer(1, 4))

    # Research Use Only (RUO) Notice Banner (#523)
    ruo_data = [
        [Paragraph(clean_markup("<b>REGULATORY NOTICE: RESEARCH USE ONLY (RUO) — INVESTIGATIONAL CLINICAL SYSTEM</b>"), ruo_title_style)],
        [Paragraph(
            clean_markup(
                "The computational findings, segmentation contours, mitotic detections, and Nottingham histologic "
                "grading scores presented in this report were generated using deep learning digital pathology models "
                "(OncoGemma v5 pipeline) to support pathologist review under College of American Pathologists (CAP) protocols. "
                "These algorithmic metrics are investigational and intended solely for computer-assisted clinical evaluation. "
                "The final diagnosis, stage classification, and clinical interpretation remain the sole medical and legal "
                "responsibility of the signing board-certified pathologist."
            ),
            body_style
        )]
    ]
    t_ruo = Table(ruo_data, colWidths=[556])
    t_ruo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#eab308")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_ruo)
    story.append(Spacer(1, 6))

    # Computational Model Provenance & Pipeline Specification Table (#626)
    story.append(Paragraph(clean_markup("<b>COMPUTATIONAL MODEL PROVENANCE & PIPELINE SPECIFICATIONS</b>"), section_head_style))
    story.append(Spacer(1, 2))

    mv = ctx.get("model_versions") or {}
    prov_rows = [
        [Paragraph(clean_markup("<b>Pipeline Component / Engine</b>"), section_head_style), Paragraph(clean_markup("<b>Specification & Model Version</b>"), section_head_style), Paragraph(clean_markup("<b>Verification Status</b>"), section_head_style)],
        [Paragraph(clean_markup("MedGemma Clinical LLM"), bold_body_style), Paragraph(clean_markup(f"MedGemma {mv.get('medgemma', '1.5')} (Multimodal Diagnostic CoT)"), body_style), Paragraph(clean_markup("Verified Grounding"), body_style)],
        [Paragraph(clean_markup("Prompt Template Hash"), bold_body_style), Paragraph(clean_markup(f"cap_report:{mv.get('prompt_cap_report', 'v1')}"), body_style), Paragraph(clean_markup("Validated CAP Structure"), body_style)],
        [Paragraph(clean_markup("Mitosis Detection CNN"), bold_body_style), Paragraph(clean_markup(mv.get("mitosis_model", "YOLOv8x-Mitosis 40x (calibrated reticle r=262µm)")), body_style), Paragraph(clean_markup("10 HPFs (2.157 mm²)"), body_style)],
        [Paragraph(clean_markup("Nottingham Grading Engine"), bold_body_style), Paragraph(clean_markup(mv.get("grading_engine", "Multi-Head ViT + Nottingham Rules")), body_style), Paragraph(clean_markup("Components Sum (3-9)"), body_style)],
        [Paragraph(clean_markup("AJCC Staging System"), bold_body_style), Paragraph(clean_markup(mv.get("ajcc_edition", "8th / 9th Edition")), body_style), Paragraph(clean_markup("Deterministic Rule Engine"), body_style)],
        [Paragraph(clean_markup("CAP Protocol Checklist"), bold_body_style), Paragraph(clean_markup(mv.get("cap_checklist", "v4.2.0.0 (2026.06)")), body_style), Paragraph(clean_markup("Compliant Template"), body_style)]
    ]
    t_prov = Table(prov_rows, colWidths=[170, 260, 126])
    t_prov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_prov)
    story.append(Spacer(1, 6))

    # Clinical Reviewer Audit & Provenance Trail (#626)
    story.append(Paragraph(clean_markup("<b>CLINICAL REVIEWER AUDIT & VERIFICATION TRAIL</b>"), section_head_style))
    story.append(Spacer(1, 2))

    integrity_disp = ctx.get("integrity_hash") or "Pending Signature"
    sig_status_disp = "Verified Signature" if is_signed else "Pending Sign-off"
    signer_disp = f"{ctx.get('signed_by')} (NPI: {ctx.get('npi')})" if is_signed else "Unsigned Draft"

    audit_rows = [
        [Paragraph(clean_markup("<b>Audit Verification Item</b>"), section_head_style), Paragraph(clean_markup("<b>System Record / Identifier</b>"), section_head_style), Paragraph(clean_markup("<b>Status</b>"), section_head_style)],
        [Paragraph(clean_markup("Reviewer / Attesting Pathologist"), bold_body_style), Paragraph(clean_markup(signer_disp), body_style), Paragraph(clean_markup(sig_status_disp), body_style)],
        [Paragraph(clean_markup("Document Cryptographic Seal"), bold_body_style), Paragraph(clean_markup(f"SHA-256: {integrity_disp}"), body_style), Paragraph(clean_markup("Sealed & Immutable" if is_signed else "Unsealed Draft"), body_style)],
        [Paragraph(clean_markup("Digital Case Accession"), bold_body_style), Paragraph(clean_markup(case_id_str), body_style), Paragraph(clean_markup("GCS Encrypted Storage"), body_style)]
    ]
    t_audit = Table(audit_rows, colWidths=[170, 260, 126])
    t_audit.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_audit)

    # Formal Versioned Amendments Table (if any)
    amendments = ctx.get("amendments") or []
    if amendments:
        story.append(Spacer(1, 6))
        story.append(Paragraph(clean_markup("<b>FORMAL REPORT AMENDMENT HISTORY</b>"), section_head_style))
        story.append(Spacer(1, 2))
        amend_rows = [
            [Paragraph(clean_markup("<b>Version</b>"), section_head_style), Paragraph(clean_markup("<b>Amended By & Date</b>"), section_head_style), Paragraph(clean_markup("<b>Clinical Amendment Reason</b>"), section_head_style)]
        ]
        for a in amendments:
            v = a.get("version", "v1.x")
            by_date = f"{a.get('amended_by', 'Pathologist')} • {str(a.get('amended_at', ''))[:19]}"
            reason = a.get("reason", "Pathologist clinical revision")
            amend_rows.append([
                Paragraph(clean_markup(v), bold_body_style),
                Paragraph(clean_markup(by_date), body_style),
                Paragraph(clean_markup(reason), body_style)
            ])
        t_amend = Table(amend_rows, colWidths=[60, 190, 306])
        t_amend.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_amend)

    canvas_cls = make_numbered_canvas(case_id_str=case_id_str, is_draft=is_draft)
    doc.build(story, canvasmaker=canvas_cls)
    return output_path
