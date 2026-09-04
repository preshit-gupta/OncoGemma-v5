import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.gcs import download_blob_as_bytes, blob_exists, get_gcs_client
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.detection import Detection
from app.models.hpf_site import HpfSite


def to_uuid(val: Any) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        return uuid.uuid4()


def rehydrate_case_from_gcs(case_id_val: Any, db: Session) -> Case | None:
    try:
        case_uuid = to_uuid(case_id_val)
        case_str = str(case_uuid)
    except Exception:
        return None

    bucket = settings.GCS_ARTIFACTS_BUCKET
    if not (blob_exists(bucket, f"cases/{case_str}/ingest_output.json") or
            blob_exists(bucket, f"cases/{case_str}/triage/output.json") or
            blob_exists(bucket, f"cases/{case_str}/preprocess/output.json")):
        return None

    case_obj = db.get(Case, case_uuid)
    if not case_obj:
        case_obj = Case(
            id=case_uuid,
            created_by="pathologist_01",
            status="open",
            created_at=datetime.now(timezone.utc)
        )
        db.add(case_obj)
        db.flush()

    existing_slide = db.scalars(select(Slide).where(Slide.case_id == case_uuid)).first()
    if not existing_slide:
        ingest_data = {}
        if blob_exists(bucket, f"cases/{case_str}/ingest_output.json"):
            try:
                raw_bytes = download_blob_as_bytes(bucket, f"cases/{case_str}/ingest_output.json")
                ingest_data = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                pass

        slide_uuid = to_uuid(ingest_data.get("slide_id", uuid.uuid4()))
        dims = ingest_data.get("dimensions", [52842, 142079])
        mpp = ingest_data.get("mpp_x", 0.265018)

        slide_obj = Slide(
            id=slide_uuid,
            case_id=case_uuid,
            gcs_uri_original=f"gs://{settings.GCS_RAW_BUCKET}/cases/{case_str}/slide.svs",
            gcs_uri_pyramid=ingest_data.get("gcs_uri_pyramid", f"gs://{settings.GCS_PYRAMIDS_BUCKET}/{slide_uuid}/orig/"),
            checksum_sha256=ingest_data.get("checksum", ""),
            mpp_x=mpp,
            mpp_y=mpp,
            width_px=dims[0],
            height_px=dims[1],
            created_at=datetime.now(timezone.utc)
        )
        db.add(slide_obj)
        db.flush()

    stages_to_check = [
        ("ingest", f"cases/{case_str}/ingest_output.json", "completed"),
        ("preprocess", f"cases/{case_str}/preprocess/output.json", "confirmed"),
        ("qc", f"cases/{case_str}/qc/output.json", "confirmed"),
        ("triage", f"cases/{case_str}/triage/output.json", "awaiting_review"),
        ("mitosis", f"cases/{case_str}/mitosis/output.json", "awaiting_review"),
        ("grading", f"cases/{case_str}/grading/output.json", "awaiting_review"),
        ("report", f"cases/{case_str}/report/output.json", "awaiting_review"),
    ]

    for stage_name, gcs_path, default_status in stages_to_check:
        existing_stage = db.scalars(
            select(StageExecution).where(
                StageExecution.case_id == case_uuid,
                StageExecution.stage == stage_name
            )
        ).first()

        if not existing_stage and blob_exists(bucket, gcs_path):
            output_ref = f"gs://{bucket}/{gcs_path}"
            stage_exec = StageExecution(
                id=uuid.uuid4(),
                case_id=case_uuid,
                stage=stage_name,
                attempt=1,
                status=default_status,
                output_ref=output_ref,
                input_ref={},
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )
            db.add(stage_exec)
            db.flush()

            if stage_name == "triage":
                try:
                    raw_bytes = download_blob_as_bytes(bucket, gcs_path)
                    triage_json = json.loads(raw_bytes.decode("utf-8"))
                    for hs in triage_json.get("hotspots", []):
                        hs_id = str(hs.get("id") or f"hs_{uuid.uuid4().hex[:8]}")
                        hs_obj = Hotspot(
                            id=hs_id,
                            case_id=case_uuid,
                            stage_execution_id=stage_exec.id,
                            polygon_um=hs.get("polygon_um", []),
                            area_mm2=hs.get("area_mm2", 0.36),
                            prob_mean=hs.get("prob_mean", 0.8),
                            prob_max=hs.get("prob_max", 0.9),
                            source=hs.get("source", "model")
                        )
                        db.add(hs_obj)
                except Exception as e:
                    print(f"[Rehydrate Triage Hotspots Note] {e}")

            elif stage_name == "mitosis":
                try:
                    raw_bytes = download_blob_as_bytes(bucket, gcs_path)
                    mitosis_json = json.loads(raw_bytes.decode("utf-8"))
                    for c in mitosis_json.get("candidates", []):
                        det_obj = Detection(
                            id=str(c.get("id") or f"m_{uuid.uuid4().hex[:8]}"),
                            case_id=case_uuid,
                            hotspot_id=c.get("hotspot_id"),
                            centroid_um=c.get("centroid_um", [0.0, 0.0]),
                            det_conf=c.get("det_conf"),
                            ver_conf=c.get("ver_conf"),
                            label=c.get("label", "unreviewed"),
                            label_source=c.get("label_source", "model"),
                            medgemma_verdict=c.get("medgemma_verdict"),
                            medgemma_rationale=c.get("medgemma_rationale"),
                            medgemma_confidence=c.get("medgemma_confidence"),
                            crop_uri=c.get("crop_uri"),
                            crop_orig_uri=c.get("crop_orig_uri"),
                        )
                        db.add(det_obj)
                    for h in mitosis_json.get("hpfs", []):
                        hpf_obj = HpfSite(
                            case_id=case_uuid,
                            seq=int(h.get("seq", 1)),
                            center_um=h.get("center_um", [0.0, 0.0]),
                            radius_um=float(h.get("radius_um", 262.0)),
                            mitotic_count=int(h.get("count", 0)),
                            source=str(h.get("source", "model")),
                        )
                        db.add(hpf_obj)
                except Exception as e:
                    print(f"[Rehydrate Mitosis Note] {e}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Rehydrate Commit Error] {e}")

    return db.get(Case, case_uuid)
