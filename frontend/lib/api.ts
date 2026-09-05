export const API_BASE = "";

export interface Case {
  id: string;
  created_by: string;
  status: string;
  created_at: string;
}

export interface CaseDetail extends Case {
  slides: Array<{
    id: string;
    gcs_uri_original: string;
    gcs_uri_pyramid?: string;
    format?: string;
    scanner?: string;
    status?: string;
    mpp_x?: number;
    mpp_y?: number;
    base_mag?: number;
    width_px?: number;
    height_px?: number;
    checksum_sha256?: string;
    label_stripped_at?: string;
  }>;
  stages: Array<{
    id: string;
    stage: string;
    attempt: number;
    status: string;
    output_ref?: string;
    error?: string;
    started_at?: string;
    completed_at?: string;
  }>;
  tile_url_template?: string | null;
}

export async function fetchCases(): Promise<Case[]> {
  const res = await fetch(`${API_BASE}/api/v1/cases`, {
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error("Failed to fetch cases");
  return res.json();
}

export async function createCase(): Promise<Case> {
  const res = await fetch(`${API_BASE}/api/v1/cases`, {
    method: "POST",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error("Failed to create case");
  return res.json();
}

export async function uploadSlideDirectToGCS(
  caseId: string,
  file: File,
  onProgress?: (percent: number) => void
): Promise<any> {
  // 1. Request Signed Upload URL from FastAPI control plane
  const urlRes = await fetch(`${API_BASE}/api/v1/cases/${caseId}/slide/upload-url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({
      filename: file.name,
      size_bytes: file.size,
      content_type: file.type || "application/octet-stream"
    })
  });

  if (!urlRes.ok) {
    let errDetail = urlRes.statusText;
    try {
      const body = await urlRes.json();
      if (body.detail) errDetail = body.detail;
    } catch (_) {}
    throw new Error(`Failed to acquire direct upload URL (HTTP ${urlRes.status}): ${errDetail}`);
  }

  const { upload_url, gcs_uri } = await urlRes.json();

  // 2. Upload file directly from browser to GCS bucket via Signed URL
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    if (xhr.upload && onProgress) {
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          onProgress(percent);
        }
      });
    }

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`Direct GCS upload failed with status ${xhr.status}`));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network connection error during direct GCS upload")));
    xhr.addEventListener("abort", () => reject(new Error("Direct GCS upload aborted")));

    xhr.open("PUT", upload_url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.send(file);
  });

  // 3. Finalize upload with API to record slide metadata and trigger cloud pipeline stage
  const finalizeRes = await fetch(`${API_BASE}/api/v1/cases/${caseId}/slide/finalize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({ gcs_uri })
  });

  if (!finalizeRes.ok) {
    throw new Error("Failed to finalize slide registration in cloud");
  }

  return finalizeRes.json();
}

export async function uploadSlideFile(
  caseId: string,
  file: File,
  onProgress?: (percent: number) => void
): Promise<any> {
  // First attempt zero-server-transit direct GCS upload
  try {
    return await uploadSlideDirectToGCS(caseId, file, onProgress);
  } catch (directErr) {
    if (file.size > 25 * 1024 * 1024) {
      // Cloud Run HTTP body limit is 32MB; large WSI files cannot be proxied through the API
      throw directErr;
    }
    console.warn("Direct GCS upload attempt failed, falling back to API proxy upload:", directErr);
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    if (xhr.upload && onProgress) {
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          onProgress(percent);
        }
      });
    }

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (_) {
          resolve({});
        }
      } else {
        let errorMsg = `HTTP Upload Error (${xhr.status})`;
        try {
          const body = JSON.parse(xhr.responseText);
          if (body.detail) errorMsg = body.detail;
        } catch (_) {}
        reject(new Error(errorMsg));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network connection error during file upload")));
    xhr.addEventListener("abort", () => reject(new Error("Slide upload aborted")));

    xhr.open("POST", `${API_BASE}/api/v1/cases/${caseId}/slide/upload`);
    xhr.setRequestHeader("X-User-Role", "pathologist");
    xhr.send(formData);
  });
}

export async function retryStage(caseId: string, stageName: string) {
  const res = await fetch(`${API_BASE}/api/v1/cases/${caseId}/stages/${stageName}/retry`, {
    method: "POST",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error("Failed to retry stage execution");
  return res.json();
}

export async function approveStage(caseId: string, stageName: string) {
  const res = await fetch(`${API_BASE}/api/v1/cases/${caseId}/stages/${stageName}/approve`, {
    method: "POST",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error("Failed to approve stage");
  return res.json();
}

export async function deleteCase(caseId: string) {
  const res = await fetch(`${API_BASE}/api/v1/cases/${caseId}`, {
    method: "DELETE",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => null);
    throw new Error(errData?.detail || `Failed to delete case (${res.status})`);
  }
}

export async function clearAllCases() {
  const res = await fetch(`${API_BASE}/api/v1/cases`, {
    method: "DELETE",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => null);
    throw new Error(errData?.detail || `Failed to clear cases (${res.status})`);
  }
  return res.json();
}

export async function fetchCaseDetail(caseId: string): Promise<CaseDetail> {
  const res = await fetch(`${API_BASE}/api/v1/cases/${caseId}`, {
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error("Failed to fetch case detail");
  return res.json();
}

// Stage 4: Mitosis Detection & Virtual HPFs Interfaces
export interface MitosisCandidate {
  id: string;
  hotspot_id?: string | null;
  centroid_um: [number, number];
  det_conf?: number | null;
  ver_conf?: number | null;
  label: "mitosis" | "not_mitosis" | "unreviewed";
  label_source: string;
  medgemma_verdict?: string | null;
  medgemma_rationale?: string | null;
  medgemma_confidence?: "low" | "medium" | "high" | null;
  crop_uri?: string | null;
  crop_orig_uri?: string | null;
}

export interface VirtualHpfSite {
  seq: number;
  center_um: [number, number];
  radius_um: number;
  count: number;
  source?: string;
}

export interface MitoticScoreSummary {
  count_total: number;
  n_hpf: number;
  area_mm2: number;
  per_mm2: number;
  classic_per_10hpf: number;
  mitotic_score: number; // 1, 2, or 3
}

export interface MitosisStageData {
  case_id: string;
  stage_execution_id: string;
  status: string;
  candidates: MitosisCandidate[];
  hpfs: VirtualHpfSite[];
  summary: MitoticScoreSummary;
  slide?: { width_px: number; height_px: number; mpp_x: number; mpp_y: number };
  model_versions: Record<string, string>;
  reviewed_at?: string | null;
  reviewed_by?: string | null;
}

export async function fetchMitosisStageData(caseId: string): Promise<MitosisStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/${caseId}`, {
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error(`Failed to fetch mitosis stage data (Status: ${res.status})`);
  return res.json();
}

export async function recomputeMitosis(payload: {
  case_id: string;
  candidate_labels?: Record<string, string>;
  hpfs?: Array<{ seq: number; center_um: [number, number]; radius_um?: number; source?: string }>;
  audit_toggle?: { id: string; from: string; to: string };
}): Promise<{ case_id: string; hpfs: VirtualHpfSite[]; summary: MitoticScoreSummary }> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/recompute`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error("Failed to recompute mitosis score");
  return res.json();
}

export async function addPathologistMitosis(
  caseId: string,
  centroidUm: [number, number],
  label: string = "mitosis",
  reviewedBy: string = "pathologist_01"
): Promise<{ status: string; candidate: MitosisCandidate }> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/add_candidate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({
      case_id: caseId,
      centroid_um: centroidUm,
      label,
      reviewed_by: reviewedBy
    })
  });
  if (!res.ok) throw new Error("Failed to add candidate mitosis");
  return res.json();
}

export async function bulkRejectUnreviewedMitosis(
  caseId: string,
  reviewedBy: string = "pathologist_01"
): Promise<MitosisStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/bulk_action`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({
      case_id: caseId,
      action: "reject_remaining_unreviewed",
      reviewed_by: reviewedBy
    })
  });
  if (!res.ok) throw new Error("Failed to bulk reject unreviewed candidates");
  return res.json();
}

export async function replaceMitosisHpfs(caseId: string): Promise<MitosisStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/re_place_hpfs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({
      case_id: caseId,
      action: "re_place_hpfs"
    })
  });
  if (!res.ok) throw new Error("Failed to re-place HPF sites");
  return res.json();
}

export async function confirmMitosisStage(
  caseId: string,
  reviewedBy: string = "pathologist_01"
): Promise<any> {
  const res = await fetch(`${API_BASE}/api/v1/stages/mitosis/confirm`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({
      case_id: caseId,
      reviewed_by: reviewedBy
    })
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to confirm mitosis stage");
  }
  return res.json();
}

// Stage 5: Nottingham Grading & Architectural Synthesis Interfaces
export interface GradingPatch {
  id: string;
  index: number;
  hotspot_id?: string;
  tissue_density?: number;
  source?: string;
  center_um?: [number, number];
  center_x_px: number;
  center_y_px: number;
  tumor_probability: number;
  image_url: string;
  tubule: {
    tubule_percent: number;
    tumor_present: boolean;
    confidence: "low" | "medium" | "high";
  };
  pleo: {
    pleomorphism_score: 1 | 2 | 3;
    rationale: string;
    confidence: "low" | "medium" | "high";
  };
  review_status: "suggested" | "approved" | "modified";
  user_tubule_percent?: number | null;
  user_tumor_present?: boolean | null;
  user_pleo_score?: (1 | 2 | 3) | null;
  user_notes?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
}

export interface HpfGradingSite {
  seq: number;
  center_um: [number, number];
  radius_um: number;
  mitotic_count: number;
  density_mm2: number;
  review_status: "suggested" | "approved" | "modified";
  user_mitotic_count?: number | null;
  user_notes?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
}

export interface ReviewSummary {
  total_patches: number;
  approved_patches: number;
  all_patches_reviewed: boolean;
  total_hpfs: number;
  approved_hpfs: number;
  all_hpfs_reviewed: boolean;
  is_type_confirmed: boolean;
  can_confirm: boolean;
}

export interface GradingSubscores {
  tubule_percent?: number;
  tubule_score: number;
  pleo_score: number;
  mitotic_score: number;
  nottingham_sum: number;
  grade: number;
  flags?: string[];
  is_overridden?: boolean;
}

export interface HistologicTypeMeta {
  proposed_type: string;
  differential: string[];
  rationale: string;
  confidence: "low" | "medium" | "high";
  confirmed_type: string;
  type_confirmed_by: string;
  is_confirmed: boolean;
}

export interface GradingStageData {
  case_id: string;
  slide_id?: string | null;
  status: string;
  patches: GradingPatch[];
  hpfs: HpfGradingSite[];
  review_summary: ReviewSummary;
  machine: GradingSubscores;
  current: GradingSubscores;
  histologic_type: HistologicTypeMeta;
  narrative: string;
  overrides: Record<string, any>;
  mitotic_summary: {
    total_mitoses: number;
    mitotic_score: number;
    evaluated_hpfs: number;
  };
  model_versions: Record<string, any>;
}

export interface SinglePatchReview {
  patch_id: string;
  tubule_percent?: number;
  tumor_present?: boolean;
  pleomorphism_score?: 1 | 2 | 3;
  status: "suggested" | "approved" | "modified";
  notes?: string;
}

export interface PatchReviewPayload {
  case_id: string;
  reviewed_by?: string;
  action: "update" | "approve_all" | "reset_all";
  reviews?: SinglePatchReview[];
}

export interface SingleHpfReview {
  seq: number;
  mitotic_count?: number;
  status: "suggested" | "approved" | "modified";
  notes?: string;
}

export interface HpfReviewPayload {
  case_id: string;
  reviewed_by?: string;
  action: "update" | "approve_all" | "reset_all";
  reviews?: SingleHpfReview[];
}

export async function fetchGradingStageData(caseId: string): Promise<GradingStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/${caseId}`, {
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error(`Failed to fetch grading stage data (Status: ${res.status})`);
  return res.json();
}

export async function reviewGradingPatches(payload: PatchReviewPayload): Promise<GradingStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/patches/review`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to update patch reviews");
  }
  return res.json();
}

export async function reviewGradingHpfs(payload: HpfReviewPayload): Promise<GradingStageData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/hpfs/review`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to update HPF reviews");
  }
  return res.json();
}

export async function recomputeGradingPreview(payload: {
  case_id: string;
  tubule_score?: number;
  tubule_percent?: number;
  pleo_score?: number;
  mitotic_score?: number;
}): Promise<{
  tubule_score: number;
  pleo_score: number;
  mitotic_score: number;
  nottingham_sum: number;
  grade: number;
  is_overridden: boolean;
}> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/recompute`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error("Failed to recompute grade preview");
  return res.json();
}

export async function confirmHistologicType(payload: {
  case_id: string;
  histologic_type: string;
  justification?: string;
  reviewed_by?: string;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/type/confirm`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to confirm histologic subtype");
  }
  return res.json();
}

export async function confirmGradingStage(payload: {
  case_id: string;
  reviewed_by?: string;
  histologic_type: string;
  type_confirmed: boolean;
  overrides: Record<string, any>;
  tubule_score: number;
  tubule_percent?: number;
  pleo_score: number;
  mitotic_score: number;
  nottingham_sum: number;
  grade: number;
}): Promise<any> {
  const res = await fetch(`${API_BASE}/api/v1/stages/grading/confirm`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to confirm grading stage");
  }
  return res.json();
}

// Stage 6: CAP-Compliant Synoptic Reporting Interfaces
export interface CapReportData {
  case_id: string;
  slide_id?: string | null;
  status: "draft" | "in_review" | "signed" | "amended";
  stage_status: string;
  specimen_type: string;
  procedure: string;
  laterality: string;
  tumor_site: string;
  histologic_type: string;
  tumor_size_mm: number;
  lvi_status: "absent" | "present" | "indeterminate";
  dcis_present: boolean;
  margins: {
    status: "negative" | "positive" | "cannot_be_assessed";
    closest_margin_mm?: number;
    closest_margin_name?: string;
    positive_margins?: string[];
  };
  lymph_nodes: {
    examined_count: number;
    positive_count: number;
    extranodal_extension: boolean;
    largest_metastasis_mm?: number;
  };
  biomarkers: {
    er: { status: string; percent: number; allred_score: number };
    pr: { status: string; percent: number; allred_score: number };
    her2: { ihc_score: string; fish_status: string; result: string };
    ki67: { percent: number };
  };
  staging: {
    ajcc_version: string;
    pt_stage: string;
    pn_stage: string;
    pm_stage: string;
    stage_group: string;
  };
  nottingham_grade: {
    grade: number;
    tubule_score: number;
    tubule_percent: number;
    pleo_score: number;
    mitotic_score: number;
    nottingham_sum: number;
    histologic_type: string;
    type_confirmed_by: string;
  };
  narrative: {
    diagnosis_line: string;
    microscopic_findings: string;
    clinical_correlation: string;
  };
  visual_evidence: {
    has_heatmap: boolean;
    has_mitotic_hpf: boolean;
    has_grading_patch: boolean;
  };
  pdf_url: string;
  json_url: string;
  signed_by?: string | null;
  npi?: string | null;
  attestation_statement?: string | null;
  signed_at?: string | null;
  integrity_hash?: string | null;
  amendments: Array<{
    version: string;
    amended_by: string;
    amended_at: string;
    reason: string;
    previous_hash?: string;
    updated_fields?: Record<string, any>;
  }>;
  can_sign: boolean;
}

export async function fetchReportData(caseId: string): Promise<CapReportData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/report/${caseId}`, {
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) throw new Error(`Failed to fetch CAP report data (Status: ${res.status})`);
  return res.json();
}

export async function updateReportData(payload: Partial<CapReportData> & { case_id: string }): Promise<CapReportData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/report/${payload.case_id}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to update report data");
  }
  return res.json();
}

export async function regenerateReportNarrative(caseId: string): Promise<{ status: string; narrative: any; warnings: string[] }> {
  const res = await fetch(`${API_BASE}/api/v1/stages/report/${caseId}/regenerate-narrative`, {
    method: "POST",
    headers: { "X-User-Role": "pathologist" }
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to regenerate diagnostic narrative");
  }
  return res.json();
}

export async function signReport(payload: {
  case_id: string;
  signed_by: string;
  npi?: string;
  attestation_statement: string;
  password_or_pin?: string;
}): Promise<CapReportData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/report/sign`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    let msg = "Failed to sign and finalize report";
    if (typeof errData.detail === "string") {
      msg = errData.detail;
    } else if (errData.detail && typeof errData.detail === "object") {
      if (Array.isArray(errData.detail.missing_items)) {
        msg = `${errData.detail.error || "Preconditions not met"}: ${errData.detail.missing_items.join("; ")}`;
      } else {
        msg = errData.detail.error || errData.detail.message || JSON.stringify(errData.detail);
      }
    }
    const err: any = new Error(msg);
    err.status = res.status;
    err.detail = errData.detail;
    throw err;
  }
  return res.json();
}

export async function amendReport(payload: {
  case_id: string;
  amended_by: string;
  amendment_reason: string;
  updated_fields?: Record<string, any>;
}): Promise<CapReportData> {
  const res = await fetch(`${API_BASE}/api/v1/stages/report/amend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to submit report amendment");
  }
  return res.json();
}

export async function updateSlideMpp(
  caseId: string,
  slideId: string,
  mppX: number,
  mppY?: number
): Promise<any> {
  const res = await fetch(`${API_BASE}/api/v1/cases/${caseId}/slides/${slideId}/mpp`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "X-User-Role": "pathologist"
    },
    body: JSON.stringify({ mpp_x: mppX, mpp_y: mppY })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to update MPP" }));
    throw new Error(err.detail || "Failed to update MPP");
  }
  return res.json();
}



