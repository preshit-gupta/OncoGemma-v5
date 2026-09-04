"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, RefreshCcw, Info, X, Microscope, AlertTriangle, CheckCircle2, RotateCcw } from "lucide-react";
import { fetchCaseDetail, CaseDetail, retryStage, approveStage, updateSlideMpp } from "@/lib/api";
import { formatISTDateTime } from "@/lib/utils";
import dynamic from "next/dynamic";
import { StageRail } from "@/components/viewer/StageRail";

const OpenSeadragonViewer = dynamic(
  () => import("@/components/viewer/OpenSeadragonViewer").then((mod) => mod.OpenSeadragonViewer),
  { ssr: false }
);

const TriageViewer = dynamic(
  () => import("@/components/viewer/TriageViewer").then((mod) => mod.TriageViewer),
  { ssr: false }
);

const MitosisViewer = dynamic(
  () => import("@/components/viewer/MitosisViewer").then((mod) => mod.MitosisViewer),
  { ssr: false }
);

const GradingReviewWorkspace = dynamic(
  () => import("@/components/viewer/GradingReviewWorkspace").then((mod) => mod.GradingReviewWorkspace),
  { ssr: false }
);

const ReportWorkspace = dynamic(
  () => import("@/components/viewer/ReportWorkspace").then((mod) => mod.ReportWorkspace),
  { ssr: false }
);

export default function CaseWorkspacePage({ params }: { params: { id: string } }) {
  const caseId = params.id;

  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeStage, setActiveStage] = useState<string>("ingest");
  const [showSlideDetails, setShowSlideDetails] = useState<boolean>(false);
  const [actionLoading, setActionLoading] = useState<boolean>(false);

  const [hasUserNavigated, setHasUserNavigated] = useState<boolean>(false);

  const loadData = async () => {
    try {
      const data = await fetchCaseDetail(caseId);
      setCaseDetail(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2500);
    return () => clearInterval(interval);
  }, [caseId]);

  // Controlled auto-advance: advances to highest active/ready stage
  useEffect(() => {
    if (!caseDetail?.stages) return;
    const stages = caseDetail.stages;

    const prepStage = stages.find((s) => s.stage === "preprocess");
    const triageStage = stages.find((s) => s.stage === "triage");
    const mitosisStage = stages.find((s) => s.stage === "mitosis");
    const gradingStage = stages.find((s) => s.stage === "grading");
    const reportStage = stages.find((s) => s.stage === "report");

    if (!hasUserNavigated) {
      if (reportStage && (reportStage.status === "running" || reportStage.status === "done" || reportStage.status === "confirmed" || reportStage.status === "awaiting_review")) {
        setActiveStage("report");
      } else if (gradingStage && (gradingStage.status === "running" || gradingStage.status === "done" || gradingStage.status === "confirmed" || gradingStage.status === "awaiting_review")) {
        setActiveStage("grading");
      } else if (mitosisStage && (mitosisStage.status === "running" || mitosisStage.status === "done" || mitosisStage.status === "confirmed" || mitosisStage.status === "awaiting_review")) {
        setActiveStage("mitosis");
      } else if (triageStage && (triageStage.status === "running" || triageStage.status === "done" || triageStage.status === "confirmed" || triageStage.status === "awaiting_review") && (prepStage?.status === "confirmed" || prepStage?.status === "done")) {
        setActiveStage("triage");
      } else if (prepStage && (prepStage.status === "running" || prepStage.status === "done" || prepStage.status === "confirmed" || prepStage.status === "awaiting_review" || prepStage.status === "queued")) {
        setActiveStage("preprocess");
      }
    }
  }, [caseDetail, hasUserNavigated]);

  const slide = caseDetail?.slides?.[0];
  const ingestStage = caseDetail?.stages?.find((s) => s.stage === "ingest");
  const preprocessStage = caseDetail?.stages?.find((s) => s.stage === "preprocess");
  const triageStage = caseDetail?.stages?.find((s) => s.stage === "triage");

  const isIngestDone = ingestStage?.status === "completed" || ingestStage?.status === "done";
  const isIngestRunning = ingestStage?.status === "running" || ingestStage?.status === "queued" || !ingestStage;
  const isIngestFailed = ingestStage?.status === "failed";
  const isNeedsMpp = slide?.status === "needs_mpp" || caseDetail?.status === "needs_mpp" || (isIngestDone && (!slide?.mpp_x || slide?.mpp_x <= 0));

  const [mppInput, setMppInput] = useState<string>("0.25");
  const [mppYInput, setMppYInput] = useState<string>("");
  const [mppSubmitting, setMppSubmitting] = useState<boolean>(false);
  const [mppError, setMppError] = useState<string | null>(null);

  const handleMppSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!slide?.id) return;
    const valX = parseFloat(mppInput);
    if (isNaN(valX) || valX <= 0) {
      setMppError("Please enter a valid positive number for MPP X.");
      return;
    }
    const valY = mppYInput ? parseFloat(mppYInput) : undefined;
    if (valY !== undefined && (isNaN(valY) || valY <= 0)) {
      setMppError("Please enter a valid positive number for MPP Y.");
      return;
    }
    setMppSubmitting(true);
    setMppError(null);
    try {
      await updateSlideMpp(caseId, slide.id, valX, valY);
      await loadData();
    } catch (err: any) {
      setMppError(err?.message || "Failed to update MPP");
    } finally {
      setMppSubmitting(false);
    }
  };

  const isPreprocessDone = preprocessStage?.status === "done" || preprocessStage?.status === "confirmed" || preprocessStage?.status === "awaiting_review";
  const isTriageDone = triageStage?.status === "done" || triageStage?.status === "confirmed" || triageStage?.status === "awaiting_review";
  const mitosisStage = caseDetail?.stages?.find((s) => s.stage === "mitosis");
  const isMitosisDone = mitosisStage?.status === "done" || mitosisStage?.status === "confirmed" || mitosisStage?.status === "awaiting_review";

  const handleRetryIngest = async () => {
    try {
      await retryStage(caseId, "ingest");
      await loadData();
    } catch (err) {
      console.error(err);
    }
  };

  const handleApprovePreprocess = async () => {
    setActionLoading(true);
    try {
      await approveStage(caseId, "preprocess");
      setHasUserNavigated(true);
      setActiveStage("triage");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleReprocessPreprocess = async () => {
    setActionLoading(true);
    try {
      await retryStage(caseId, "preprocess");
      setHasUserNavigated(true);
      setActiveStage("preprocess");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleReprocessTriage = async () => {
    setActionLoading(true);
    try {
      await retryStage(caseId, "triage");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleApproveTriage = async () => {
    setActionLoading(true);
    try {
      await approveStage(caseId, "triage");
      setHasUserNavigated(true);
      setActiveStage("mitosis");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleApproveMitosis = async () => {
    setActionLoading(true);
    try {
      await approveStage(caseId, "mitosis");
      setHasUserNavigated(true);
      setActiveStage("grading");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleReprocessMitosis = async () => {
    setActionLoading(true);
    try {
      await retryStage(caseId, "mitosis");
      await loadData();
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-900">
      {/* Top Workspace Bar */}
      <div className="bg-slate-900 border-b border-slate-800 text-white px-4 py-2 flex items-center justify-between z-20">
        <div className="flex items-center space-x-3">
          <Link
            href="/cases"
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition"
            title="Back to Cases"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div className="h-4 w-[1px] bg-slate-700" />
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-sm font-semibold tracking-tight">
                Case #{caseId.substring(0, 8)}
              </h1>
              <span className="text-[10px] bg-sky-900/60 border border-sky-700 text-sky-300 px-2 py-0.5 rounded font-mono font-medium">
                Nottingham Grading
              </span>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center space-x-3 mt-0.5">
              <span>MPP: {slide?.mpp_x ? `${slide.mpp_x} µm/px` : "Needs Calibration (Missing MPP)"}</span>
              <span>•</span>
              <span className="font-mono text-slate-300">Base Scan: {slide?.base_mag ? `${slide.base_mag}× Objective` : "Pending MPP"} {slide?.mpp_x ? `(400× Optical / ${slide.mpp_x} µm/px)` : ""}</span>
              <span>•</span>
              <span>Created: {formatISTDateTime(caseDetail?.created_at)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Pathologist Action Buttons for Step 2 (v4.1) */}
          {isPreprocessDone && activeStage === "preprocess" && (
            <div className="flex items-center space-x-2 border-r border-slate-700 pr-3 mr-1">
              <button
                onClick={handleReprocessPreprocess}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-amber-600/90 hover:bg-amber-600 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-amber-500/50"
                title="Re-run Macenko stain normalization & QC gate"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? "animate-spin" : ""}`} />
                <span>Re-Process Slide</span>
              </button>

              <button
                onClick={handleApprovePreprocess}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-emerald-400/50"
                title="Approve slide stain quality & proceed to Step 3 (v4.2 Hotspot Triage)"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Approve Slide & Proceed to Step 3</span>
              </button>
            </div>
          )}

          {/* Pathologist Action Buttons for Step 3 (v4.2) */}
          {isTriageDone && activeStage === "triage" && (
            <div className="flex items-center space-x-2 border-r border-slate-700 pr-3 mr-1">
              <button
                onClick={handleReprocessTriage}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-amber-600/90 hover:bg-amber-600 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-amber-500/50"
                title="Re-run Vertex AI Path Foundation screening and hotspot assessment"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? "animate-spin" : ""}`} />
                <span>Re-Assess Hotspots</span>
              </button>

              <button
                onClick={handleApproveTriage}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-emerald-400/50"
                title="Confirm hotspots & proceed to Step 4 (v4.3 Mitosis Counting)"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Confirm Hotspots & Proceed to Step 4</span>
              </button>
            </div>
          )}

          {/* Pathologist Action Buttons for Step 4 (v4.3) */}
          {isMitosisDone && activeStage === "mitosis" && (
            <div className="flex items-center space-x-2 border-r border-slate-700 pr-3 mr-1">
              <button
                onClick={handleReprocessMitosis}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-amber-600/90 hover:bg-amber-600 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-amber-500/50"
                title="Re-run mitosis detection and virtual HPF placement"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? "animate-spin" : ""}`} />
                <span>Re-Count Mitoses</span>
              </button>

              <button
                onClick={handleApproveMitosis}
                disabled={actionLoading}
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow border border-emerald-400/50"
                title="Confirm mitoses & proceed to Step 5 (v4.4 Nottingham Grading)"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Confirm Mitoses & Proceed to Step 5</span>
              </button>
            </div>
          )}

          <button
            onClick={() => setShowSlideDetails(!showSlideDetails)}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition text-xs flex items-center space-x-1.5 border border-slate-700"
          >
            <Info className="w-3.5 h-3.5 text-sky-400" />
            <span>Slide Details</span>
          </button>

          <button
            onClick={loadData}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition text-xs flex items-center space-x-1 border border-slate-700"
            title="Refresh Status"
          >
            <RefreshCcw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Stage Rail */}
        <StageRail
          caseId={caseId}
          stages={caseDetail?.stages || []}
          activeStage={activeStage}
          onSelectStage={(stage) => {
            setHasUserNavigated(true);
            setActiveStage(stage);
          }}
          onRefresh={loadData}
        />

        {/* Center Digital Slide Viewer / Stage View */}
        <div className="flex-1 relative overflow-hidden bg-slate-950">
          {loading ? (
            <div className="flex items-center justify-center h-full text-slate-400 text-sm">
              Loading slide workspace...
            </div>
          ) : isIngestRunning ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-300 space-y-4 bg-slate-950 p-8">
              <div className="relative w-16 h-16 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border-4 border-sky-500/20 border-t-sky-500 animate-spin" />
                <Microscope className="w-8 h-8 text-sky-400" />
              </div>
              <div className="text-center max-w-md">
                <h3 className="text-base font-bold text-white tracking-tight">Processing Whole-Slide Image</h3>
                <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
                  Extracting WSI metadata, streaming raw slide to Cloud Storage (<span className="font-mono text-sky-400">gs://oncogemma-dev-raw</span>), and generating multi-resolution pyramid tiles...
                </p>
                <div className="mt-4 inline-flex items-center space-x-2 text-[11px] font-mono text-sky-400 bg-sky-950/60 border border-sky-800/80 px-3 py-1.5 rounded-full">
                  <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
                  <span>Pipeline Worker Active</span>
                </div>
              </div>
            </div>
          ) : isIngestFailed ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-300 space-y-4 bg-slate-950 p-8">
              <AlertTriangle className="w-12 h-12 text-rose-500" />
              <div className="text-center max-w-md">
                <h3 className="text-base font-bold text-white">Slide Ingest Failed</h3>
                <p className="text-xs text-slate-400 mt-1">
                  {ingestStage?.error || "Failed to process slide file during pyramid tile generation."}
                </p>
                <button
                  onClick={handleRetryIngest}
                  className="mt-4 bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition shadow"
                >
                  Retry Ingest Stage
                </button>
              </div>
            </div>
          ) : isNeedsMpp ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-300 space-y-4 bg-slate-950 p-8">
              <div className="relative w-16 h-16 flex items-center justify-center bg-amber-500/10 rounded-full border border-amber-500/30">
                <AlertTriangle className="w-8 h-8 text-amber-400" />
              </div>
              <div className="text-center max-w-lg">
                <h3 className="text-base font-bold text-white tracking-tight">Slide Calibration Required (Missing MPP)</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  Per PRD 01-stage-v4.0 §2.3 step 4, the whole-slide scanner did not record micrometers-per-pixel (MPP).
                  Automatic guessing of 0.25 µm/px is strictly forbidden to prevent miscalculation of mitotic density and Nottingham Grade.
                  Please enter the calibrated scanner MPP to begin preprocessing.
                </p>
                <form onSubmit={handleMppSubmit} className="mt-5 bg-slate-900 border border-slate-800 rounded-xl p-4 text-left space-y-3">
                  <div>
                    <label className="block text-[11px] font-medium text-slate-300 mb-1">
                      MPP X (µm/pixel) <span className="text-rose-400">*</span>
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      min="0.001"
                      required
                      value={mppInput}
                      onChange={(e) => setMppInput(e.target.value)}
                      placeholder="e.g. 0.25 for 40× or 0.50 for 20×"
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-medium text-slate-300 mb-1">
                      MPP Y (µm/pixel) <span className="text-slate-500 text-[10px]">(optional, defaults to MPP X)</span>
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      min="0.001"
                      value={mppYInput}
                      onChange={(e) => setMppYInput(e.target.value)}
                      placeholder="Leave blank for square pixels"
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
                    />
                  </div>
                  {mppError && (
                    <p className="text-[11px] text-rose-400 font-medium">{mppError}</p>
                  )}
                  <button
                    type="submit"
                    disabled={mppSubmitting}
                    className="w-full bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-xs font-semibold py-2 px-4 rounded-lg transition shadow flex items-center justify-center space-x-2"
                  >
                    {mppSubmitting ? (
                      <span>Saving & Queuing Preprocess...</span>
                    ) : (
                      <span>Save Calibration & Queue Preprocess</span>
                    )}
                  </button>
                </form>
              </div>
            </div>
          ) : activeStage === "report" ? (
            <ReportWorkspace
              caseId={caseId}
              onRefreshCase={loadData}
            />
          ) : activeStage === "grading" ? (
            <GradingReviewWorkspace
              caseId={caseId}
              onAdvanceToReport={() => {
                setHasUserNavigated(true);
                setActiveStage("report");
                loadData();
              }}
              onReopenMitosis={() => {
                setHasUserNavigated(true);
                setActiveStage("mitosis");
              }}
            />
          ) : activeStage === "mitosis" ? (
            <MitosisViewer
              caseId={caseId}
              mppX={slide?.mpp_x || 0.25}
              mppY={slide?.mpp_y || slide?.mpp_x || 0.25}
              imageWidthPx={slide?.width_px || 2048}
              imageHeightPx={slide?.height_px || 2048}
              onRefreshCase={loadData}
              tileUrlTemplate={caseDetail?.tile_url_template}
            />
          ) : activeStage === "triage" ? (
            <TriageViewer
              caseId={caseId}
              mppX={slide?.mpp_x || 0.25}
              mppY={slide?.mpp_y || slide?.mpp_x || 0.25}
              imageWidthPx={slide?.width_px || 2048}
              imageHeightPx={slide?.height_px || 2048}
              onRefreshCase={loadData}
              onAdvanceToMitosis={() => {
                setHasUserNavigated(true);
                setActiveStage("mitosis");
                loadData();
              }}
              tileUrlTemplate={caseDetail?.tile_url_template}
            />
          ) : (
            <OpenSeadragonViewer
              caseId={caseId}
              mppX={slide?.mpp_x || 0.25}
              mppY={slide?.mpp_y || slide?.mpp_x || 0.25}
              imageWidthPx={slide?.width_px || 2048}
              imageHeightPx={slide?.height_px || 2048}
              tileUrlTemplate={caseDetail?.tile_url_template}
            />
          )}
        </div>

        {/* Slide Technical Details Popover/Modal */}
        {showSlideDetails && (
          <div className="absolute top-4 left-72 bg-slate-900/95 backdrop-blur border border-slate-700 text-white rounded-xl shadow-2xl p-4 w-96 z-30 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-1.5">
                <Info className="w-4 h-4" />
                <span>Technical Slide Details</span>
              </h3>
              <button
                onClick={() => setShowSlideDetails(false)}
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between border-b border-slate-800/50 py-1">
                <span className="text-slate-400">Slide ID:</span>
                <span className="font-mono text-slate-200">{slide?.id || "N/A"}</span>
              </div>
              <div className="flex justify-between border-b border-slate-800/50 py-1">
                <span className="text-slate-400">Original Format:</span>
                <span className="font-mono text-slate-200 uppercase">{slide?.format || "SVS"}</span>
              </div>
              <div className="flex justify-between border-b border-slate-800/50 py-1">
                <span className="text-slate-400">Dimensions:</span>
                <span className="font-mono text-slate-200">{slide?.width_px || "N/A"} x {slide?.height_px || "N/A"} px</span>
              </div>
              <div className="flex justify-between border-b border-slate-800/50 py-1">
                <span className="text-slate-400">Scanner Vendor:</span>
                <span className="font-mono text-slate-200 capitalize">{slide?.scanner || "Generic"}</span>
              </div>
              <div className="flex justify-between border-b border-slate-800/50 py-1">
                <span className="text-slate-400">SHA256 Checksum:</span>
                <span className="font-mono text-[10px] text-slate-300 truncate max-w-[180px]" title={slide?.checksum_sha256}>
                  {slide?.checksum_sha256 || "N/A"}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Label Stripped At:</span>
                <span className="font-mono text-slate-200">{formatISTDateTime(slide?.label_stripped_at)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
