import uuid
import os
import tempfile
from io import BytesIO
from datetime import datetime, timezone
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Response
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.db import get_db
from app.core.auth import get_current_user, CurrentUser
from app.core.config import settings
from app.core.gcs import (
    upload_blob_from_file,
    download_blob_to_filename,
    generate_signed_upload_url,
    get_gcs_tile_template_url,
    parse_gcs_uri
)
from app.core.openslide_lock import OPENSLIDE_GLOBAL_LOCK
from app.core.cloud_tasks import dispatch_stage_task
from app.models.case import Case
from app.models.slide import Slide
from app.models.stage_execution import StageExecution
from app.models.hotspot import Hotspot
from app.models.detection import Detection
from app.models.hpf_site import HpfSite
from app.models.grading import Grading
from app.models.report import Report
from app.models.audit import AuditEvent
from app.core.rehydrate import rehydrate_case_from_gcs
from app.schemas.case import (
    CaseResponse,
    SlideUploadUrlRequest,
    SlideUploadUrlResponse,
    SlideFinalizeRequest,
    SlideMppUpdateRequest,
    CaseDetailResponse
)

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    if user.role not in ("admin", "pathologist"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Pathologist or Admin role required to create cases.")
    case_obj = Case(created_by=user.id)
    db.add(case_obj)
    db.commit()
    db.refresh(case_obj)

    audit = AuditEvent(
        case_id=str(case_obj.id),
        actor=user.id,
        event_type="case_created",
        payload={"created_by": user.id}
    )
    db.add(audit)
    db.commit()

    return case_obj

@router.get("", response_model=list[CaseResponse])
def list_cases(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    stmt = select(Case).order_by(Case.created_at.desc())
    cases = db.scalars(stmt).all()
    if not cases:
        try:
            from app.core.gcs import get_gcs_client
            client = get_gcs_client()
            bucket = client.bucket(settings.GCS_ARTIFACTS_BUCKET)
            blobs = bucket.list_blobs(prefix="cases/", delimiter="/")
            list(blobs)
            for prefix in getattr(blobs, "prefixes", []):
                parts = prefix.strip("/").split("/")
                if len(parts) >= 2:
                    rehydrate_case_from_gcs(parts[1], db)
        except Exception as e:
            print(f"[List Cases Rehydrate Note] {e}")
        cases = db.scalars(stmt).all()
    return cases

def delete_single_case_data(case_id: uuid.UUID, db: Session):
    """
    Safely delete a case and all associated child entities in strict dependency order,
    preventing any foreign key constraint violations, and cleans up associated GCS storage artifacts.
    """
    case_str = str(case_id)

    # 1. Delete leaf entities (mitosis detections and virtual HPF sites)
    db.query(Detection).filter(Detection.case_id == case_id).delete(synchronize_session=False)
    db.query(HpfSite).filter(HpfSite.case_id == case_id).delete(synchronize_session=False)

    # 2. Delete Hotspots (which reference stage_executions and cases)
    db.query(Hotspot).filter(Hotspot.case_id == case_id).delete(synchronize_session=False)

    # 3. Delete StageExecutions
    db.query(StageExecution).filter(StageExecution.case_id == case_id).delete(synchronize_session=False)

    # 4. Delete Grading and Report
    db.query(Grading).filter(Grading.case_id == case_id).delete(synchronize_session=False)
    db.query(Report).filter(Report.case_id == case_id).delete(synchronize_session=False)

    # 5. Delete Slide records
    db.query(Slide).filter(Slide.case_id == case_id).delete(synchronize_session=False)

    # 6. Delete AuditEvents
    db.query(AuditEvent).filter(AuditEvent.case_id == case_str).delete(synchronize_session=False)

    # 7. Delete Case record
    db.query(Case).filter(Case.id == case_id).delete(synchronize_session=False)
    db.commit()

    # 8. Clean up GCS artifacts and raw slide blobs
    try:
        from app.core.gcs import get_gcs_client
        client = get_gcs_client()
        for bname in [settings.GCS_ARTIFACTS_BUCKET, settings.GCS_RAW_BUCKET, settings.GCS_PYRAMIDS_BUCKET]:
            try:
                bucket = client.bucket(bname)
                blobs = list(bucket.list_blobs(prefix=f"cases/{case_str}/"))
                if blobs:
                    bucket.delete_blobs(blobs)
                    print(f"[Delete Case GCS] Deleted {len(blobs)} blobs from {bname} for case {case_str}")
            except Exception as b_err:
                print(f"[Delete Case GCS Note] Could not delete blobs from {bname}: {b_err}")
    except Exception as gcs_err:
        print(f"[Delete Case GCS Exception] {gcs_err}")


@router.delete("", status_code=status.HTTP_200_OK)
def clear_all_cases(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """Clear all diagnostic cases, associated relational child data, and storage artifacts (Admin only)."""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required to clear all cases.")
    cases = db.scalars(select(Case)).all()
    count = len(cases)
    for c in cases:
        delete_single_case_data(c.id, db)

    return {"status": "cleared", "deleted_count": count}


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """Delete a single diagnostic case and all associated child data (Pathologist or Admin only)."""
    if user.role not in ("admin", "pathologist"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Pathologist or Admin role required to delete a case.")
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")
    
    delete_single_case_data(case_id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{case_id}/slide/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_slide_file(
    case_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """Direct file upload endpoint streaming directly to GCS bucket with zero local persistence."""
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    file_uuid = uuid.uuid4()
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else "svs"
    
    blob_name = f"cases/{case_id}/{file_uuid}.{ext}"
    gcs_uri = f"gs://{settings.GCS_RAW_BUCKET}/{blob_name}"

    try:
        await file.seek(0)
        upload_blob_from_file(
            bucket_name=settings.GCS_RAW_BUCKET,
            blob_name=blob_name,
            file_obj=file.file,
            content_type=file.content_type or "application/octet-stream"
        )
        print(f"[Upload Success] Directly uploaded slide to {gcs_uri}")
    except Exception as save_err:
        print(f"[Upload Error] GCS bucket upload failed: {save_err}")
        raise HTTPException(status_code=500, detail=f"GCS bucket upload failed: {str(save_err)}")

    # Create slide record
    slide_obj = Slide(
        case_id=case_id,
        gcs_uri_original=gcs_uri
    )
    db.add(slide_obj)
    db.flush()
    
    # Queue 'ingest' stage_execution
    stage_exec = StageExecution(
        case_id=case_id,
        stage="ingest",
        attempt=1,
        status="queued",
        input_ref={
            "gcs_uri_original": gcs_uri,
            "slide_id": str(slide_obj.id),
            "blob_name": blob_name
        }
    )
    db.add(stage_exec)
    
    # Audit event
    audit = AuditEvent(
        case_id=str(case_id),
        actor=user.id,
        event_type="slide_uploaded",
        stage="ingest",
        payload={"gcs_uri": gcs_uri, "slide_id": str(slide_obj.id), "filename": file.filename}
    )
    db.add(audit)
    
    db.commit()

    dispatch_stage_task(
        case_id=str(case_id),
        stage="ingest",
        stage_exec_id=str(stage_exec.id),
        payload={"gcs_uri": gcs_uri, "slide_id": str(slide_obj.id)}
    )

    return {
        "status": "queued",
        "slide_id": str(slide_obj.id),
        "stage_execution_id": str(stage_exec.id),
        "gcs_uri": gcs_uri
    }

@router.post("/{case_id}/stages/{stage_name}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_case_stage(
    case_id: uuid.UUID,
    stage_name: str,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """Re-queue execution attempt for a specific pipeline stage."""
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    slide_obj = db.scalars(select(Slide).where(Slide.case_id == case_id)).first()
    if not slide_obj:
        raise HTTPException(status_code=404, detail="Slide not found")

    stmt = (
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == stage_name)
        .order_by(StageExecution.attempt.desc())
    )
    existing_stage = db.scalars(stmt).first()
    next_attempt = (existing_stage.attempt + 1) if existing_stage else 1

    new_stage = StageExecution(
        case_id=case_id,
        stage=stage_name,
        attempt=next_attempt,
        status="queued",
        input_ref={"gcs_uri_original": slide_obj.gcs_uri_original, "slide_id": str(slide_obj.id)}
    )
    db.add(new_stage)
    
    audit = AuditEvent(
        case_id=str(case_id),
        actor=user.id,
        event_type="stage_retried",
        stage=stage_name,
        payload={"attempt": next_attempt}
    )
    db.add(audit)
    db.commit()
    db.refresh(new_stage)

    dispatch_stage_task(
        case_id=str(case_id),
        stage=stage_name,
        stage_exec_id=str(new_stage.id),
        payload={"slide_id": str(slide_obj.id)}
    )

    return {
        "status": "queued",
        "stage_execution_id": str(new_stage.id),
        "attempt": next_attempt
    }

@router.post("/{case_id}/stages/{stage_name}/approve", status_code=status.HTTP_202_ACCEPTED)
def approve_case_stage(
    case_id: uuid.UUID,
    stage_name: str,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """
    Approve pipeline stage output by Pathologist and trigger the next stage execution (e.g. v4.2 Hotspot Triage).
    """
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    slide_obj = db.scalars(select(Slide).where(Slide.case_id == case_id)).first()
    if not slide_obj:
        raise HTTPException(status_code=404, detail="Slide not found")

    # Mark current stage execution as confirmed
    stmt = (
        select(StageExecution)
        .where(StageExecution.case_id == case_id, StageExecution.stage == stage_name)
        .order_by(StageExecution.attempt.desc())
    )
    current_stage = db.scalars(stmt).first()
    if current_stage:
        current_stage.status = "confirmed"

    # If approving preprocess, also mark associated QC stage as confirmed
    if stage_name == "preprocess":
        qc_stage = db.scalars(
            select(StageExecution)
            .where(StageExecution.case_id == case_id, StageExecution.stage == "qc")
            .order_by(StageExecution.attempt.desc())
        ).first()
        if qc_stage and qc_stage.status in ("awaiting_review", "done"):
            qc_stage.status = "confirmed"

    # Determine next stage name
    next_stage_map = {
        "preprocess": "triage",
        "triage": "mitosis",
        "mitosis": "grading",
        "grading": "report"
    }
    next_stage_name = next_stage_map.get(stage_name)
    
    new_stage = None
    if next_stage_name:
        stmt_next = (
            select(StageExecution)
            .where(StageExecution.case_id == case_id, StageExecution.stage == next_stage_name)
            .order_by(StageExecution.attempt.desc())
        )
        existing_next = db.scalars(stmt_next).first()
        next_attempt = (existing_next.attempt + 1) if existing_next else 1

        new_stage = StageExecution(
            case_id=case_id,
            stage=next_stage_name,
            attempt=next_attempt,
            status="queued",
            input_ref={"slide_id": str(slide_obj.id), "gcs_uri_original": slide_obj.gcs_uri_original}
        )
        db.add(new_stage)

    case_obj.status = "open"
    
    audit = AuditEvent(
        case_id=str(case_id),
        actor=user.id,
        event_type="stage_approved",
        stage=stage_name,
        payload={"approved_by": user.id, "next_stage": next_stage_name}
    )
    db.add(audit)
    db.commit()

    if new_stage:
        dispatch_stage_task(
            case_id=str(case_id),
            stage=next_stage_name,
            stage_exec_id=str(new_stage.id),
            payload={"slide_id": str(slide_obj.id), "gcs_uri_original": slide_obj.gcs_uri_original}
        )

    return {
        "status": "approved",
        "approved_stage": stage_name,
        "next_stage": next_stage_name,
        "next_stage_execution_id": str(new_stage.id) if new_stage else None
    }

@router.post("/{case_id}/slide/upload-url", response_model=SlideUploadUrlResponse)
def get_slide_upload_url(
    case_id: uuid.UUID,
    req: SlideUploadUrlRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    file_uuid = uuid.uuid4()
    ext = req.filename.rsplit(".", 1)[-1].lower() if "." in req.filename else "svs"
    
    blob_name = f"cases/{case_id}/{file_uuid}.{ext}"
    gcs_uri = f"gs://{settings.GCS_RAW_BUCKET}/{blob_name}"
    upload_url = generate_signed_upload_url(settings.GCS_RAW_BUCKET, blob_name)

    return SlideUploadUrlResponse(
        upload_url=upload_url,
        gcs_uri=gcs_uri
    )

@router.post("/{case_id}/slide/finalize", status_code=status.HTTP_202_ACCEPTED)
def finalize_slide_upload(
    case_id: uuid.UUID,
    req: SlideFinalizeRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    case_obj = db.get(Case, case_id)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    slide_obj = Slide(
        case_id=case_id,
        gcs_uri_original=req.gcs_uri,
        checksum_sha256=req.client_sha256
    )
    db.add(slide_obj)
    db.flush()
    
    stage_exec = StageExecution(
        case_id=case_id,
        stage="ingest",
        attempt=1,
        status="queued",
        input_ref={"gcs_uri_original": req.gcs_uri, "slide_id": str(slide_obj.id)}
    )
    db.add(stage_exec)
    
    audit = AuditEvent(
        case_id=str(case_id),
        actor=user.id,
        event_type="slide_uploaded",
        stage="ingest",
        payload={"gcs_uri": req.gcs_uri, "slide_id": str(slide_obj.id)}
    )
    db.add(audit)
    
    db.commit()
    db.refresh(slide_obj)
    db.refresh(stage_exec)

    dispatch_stage_task(
        case_id=str(case_id),
        stage="ingest",
        stage_exec_id=str(stage_exec.id),
        payload={"gcs_uri_original": req.gcs_uri, "slide_id": str(slide_obj.id)}
    )

    return {
        "status": "queued",
        "slide_id": str(slide_obj.id),
        "stage_execution_id": str(stage_exec.id)
    }

@router.get("/{case_id}/thumbnail")
def get_case_thumbnail(
    case_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Returns a high-speed whole-slide macro thumbnail (e.g. 256x256) of the case biopsy directly from GCS.
    """
    stmt = select(Slide).where(Slide.case_id == case_id).limit(1)
    slide_obj = db.scalars(stmt).first()
    if not slide_obj:
        raise HTTPException(status_code=404, detail="Slide not found")

    gcs_uri = slide_obj.gcs_uri_original
    if not gcs_uri:
        raise HTTPException(status_code=404, detail="Slide GCS URI not set")

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    ext = os.path.splitext(blob_name)[1] or ".svs"

    temp_file = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="thumb_")
        os.close(temp_fd)
        temp_file = temp_path

        download_blob_to_filename(bucket_name, blob_name, temp_file)

        try:
            import openslide
            with OPENSLIDE_GLOBAL_LOCK:
                oslide = openslide.OpenSlide(temp_file)
                thumb = oslide.get_thumbnail((256, 256)).convert("RGB")
                oslide.close()
        except Exception:
            try:
                with Image.open(temp_file) as pil_img:
                    thumb = pil_img.copy()
                    thumb.thumbnail((256, 256))
                    thumb = thumb.convert("RGB")
            except Exception:
                thumb = Image.new("RGB", (256, 256), color=(240, 225, 235))

        buf = BytesIO()
        thumb.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case_detail(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    case_obj = db.get(Case, case_id)
    if not case_obj:
        case_obj = rehydrate_case_from_gcs(str(case_id), db)
    if not case_obj:
        raise HTTPException(status_code=404, detail="Case not found")

    slides = db.scalars(select(Slide).where(Slide.case_id == case_id)).all()
    stages = db.scalars(select(StageExecution).where(StageExecution.case_id == case_id).order_by(StageExecution.started_at.asc())).all()

    slides_data = [
        {
            "id": str(s.id),
            "status": getattr(s, "status", "ready") or "ready",
            "gcs_uri_original": s.gcs_uri_original,
            "gcs_uri_pyramid": s.gcs_uri_pyramid,
            "format": s.format,
            "scanner": s.scanner,
            "mpp_x": s.mpp_x,
            "mpp_y": s.mpp_y,
            "base_mag": s.base_mag,
            "width_px": s.width_px,
            "height_px": s.height_px,
            "checksum_sha256": s.checksum_sha256,
            "label_stripped_at": s.label_stripped_at.isoformat() if s.label_stripped_at else None
        }
        for s in slides
    ]

    stages_data = [
        {
            "id": str(st.id),
            "stage": st.stage,
            "attempt": st.attempt,
            "status": st.status,
            "output_ref": st.output_ref,
            "error": st.error,
            "started_at": st.started_at.isoformat() if st.started_at else None,
            "completed_at": st.completed_at.isoformat() if st.completed_at else None
        }
        for st in stages
    ]

    primary_slide_id = str(slides[0].id) if slides else None
    tile_template = None
    if primary_slide_id:
        tile_template = get_gcs_tile_template_url(primary_slide_id, "{layer}")

    return CaseDetailResponse(
        id=case_obj.id,
        created_by=case_obj.created_by,
        status=case_obj.status,
        created_at=case_obj.created_at,
        slides=slides_data,
        stages=stages_data,
        tile_url_template=tile_template,
        cdn_base_url=settings.CDN_BASE_URL
    )


@router.patch("/{case_id}/slides/{slide_id}/mpp", status_code=status.HTTP_200_OK)
@router.put("/{case_id}/slides/{slide_id}/mpp", status_code=status.HTTP_200_OK)
def update_slide_mpp(
    case_id: uuid.UUID,
    slide_id: uuid.UUID,
    req: SlideMppUpdateRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user)
):
    """
    Allow pathologist or admin to manually provide valid MPP for a slide marked 'needs_mpp'.
    Unblocks downstream processing by setting slide status to 'ready' and chaining preprocess stage.
    """
    if user.role not in ("admin", "pathologist"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Pathologist or Admin role required.")

    if req.mpp_x <= 0 or (req.mpp_y is not None and req.mpp_y <= 0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MPP values must be positive numbers.")

    slide = db.get(Slide, slide_id)
    if not slide:
        slide = db.scalars(select(Slide).where(Slide.id == str(slide_id))).first()
    if not slide or str(slide.case_id) != str(case_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slide not found in case.")

    slide.mpp_x = float(req.mpp_x)
    slide.mpp_y = float(req.mpp_y) if req.mpp_y is not None else float(req.mpp_x)
    slide.status = "ready"
    db.flush()

    # If case status was needs_mpp and all slides now have valid MPP, restore open status
    case_str = str(case_id)
    case_obj = db.get(Case, case_id)
    if not case_obj:
        case_obj = db.scalars(select(Case).where(Case.id == case_str)).first()

    if case_obj and case_obj.status == "needs_mpp":
        case_slides = db.scalars(
            select(Slide).where((Slide.case_id == case_id) | (Slide.case_id == case_str))
        ).all()
        remaining = [s for s in case_slides if s.status == "needs_mpp"]
        if not remaining:
            case_obj.status = "open"

    db.commit()
    db.refresh(slide)

    # If ingest output exists and preprocess is not yet queued, auto-queue preprocess
    existing_prep = db.scalars(
        select(StageExecution).where(
            StageExecution.case_id == case_id,
            StageExecution.stage == "preprocess"
        )
    ).first()

    if not existing_prep:
        output_ref = f"gs://{settings.GCS_ARTIFACTS_BUCKET}/cases/{case_id}/ingest_output.json"
        next_prep_stage = StageExecution(
            case_id=case_id,
            stage="preprocess",
            attempt=1,
            status="queued",
            input_ref={"slide_id": str(slide.id), "ingest_output_ref": output_ref}
        )
        db.add(next_prep_stage)
        db.commit()
        db.refresh(next_prep_stage)

        from app.core.cloud_tasks import dispatch_stage_task
        dispatch_stage_task(
            case_id=str(case_id),
            stage="preprocess",
            stage_exec_id=str(next_prep_stage.id),
            payload={"slide_id": str(slide.id), "ingest_output_ref": output_ref}
        )

    return {
        "slide_id": str(slide.id),
        "status": slide.status,
        "mpp_x": slide.mpp_x,
        "mpp_y": slide.mpp_y
    }


