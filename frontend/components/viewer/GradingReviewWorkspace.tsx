"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Edit3,
  Layers,
  ArrowRight,
  Eye,
  Microscope,
  HelpCircle,
  FileCheck,
  Check,
  X,
  ExternalLink,
  ChevronRight,
  Info,
  ShieldCheck,
  AlertCircle,
  Sliders,
  CheckCheck,
  Filter,
  Plus,
  Minus,
  MessageSquare
} from "lucide-react";
import {
  GradingStageData,
  GradingPatch,
  HpfGradingSite,
  fetchGradingStageData,
  reviewGradingPatches,
  reviewGradingHpfs,
  recomputeGradingPreview,
  confirmGradingStage,
  API_BASE
} from "@/lib/api";

interface GradingReviewWorkspaceProps {
  caseId: string;
  onAdvanceToReport?: () => void;
  onReopenMitosis?: () => void;
}

const HISTOLOGIC_TYPE_OPTIONS = [
  { id: "IDC-NST", label: "Invasive Breast Carcinoma of No Special Type (IDC-NST / Ductal)" },
  { id: "ILC", label: "Invasive Lobular Carcinoma (ILC)" },
  { id: "mucinous", label: "Mucinous Carcinoma" },
  { id: "tubular", label: "Tubular Carcinoma" },
  { id: "papillary", label: "Invasive Papillary Carcinoma" },
  { id: "metaplastic", label: "Metaplastic Carcinoma" },
  { id: "other", label: "Other / Special Variant Carcinoma" }
];

export function GradingReviewWorkspace({
  caseId,
  onAdvanceToReport,
  onReopenMitosis
}: GradingReviewWorkspaceProps) {
  const [data, setData] = useState<GradingStageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Global subscore overrides
  const [tubuleOverrideScore, setTubuleOverrideScore] = useState<number | null>(null);
  const [tubuleOverridePercent, setTubuleOverridePercent] = useState<number | null>(null);
  const [tubuleJustification, setTubuleJustification] = useState<string>("");
  const [isTubuleEditing, setIsTubuleEditing] = useState(false);

  const [pleoOverrideScore, setPleoOverrideScore] = useState<number | null>(null);
  const [pleoJustification, setPleoJustification] = useState<string>("");
  const [isPleoEditing, setIsPleoEditing] = useState(false);

  // Histologic type gate
  const [selectedHistologicType, setSelectedHistologicType] = useState<string>("IDC-NST");
  const [isTypeConfirmed, setIsTypeConfirmed] = useState<boolean>(false);

  // Filter & Inspection Modals
  const [patchFilter, setPatchFilter] = useState<"all" | "suggested" | "approved" | "modified">("all");
  const [selectedPatch, setSelectedPatch] = useState<GradingPatch | null>(null);
  const [editingPatch, setEditingPatch] = useState<GradingPatch | null>(null);
  const [patchEditTubule, setPatchEditTubule] = useState<number>(20);
  const [patchEditTumorPresent, setPatchEditTumorPresent] = useState<boolean>(true);
  const [patchEditPleo, setPatchEditPleo] = useState<1 | 2 | 3>(2);
  const [patchEditNotes, setPatchEditNotes] = useState<string>("");

  // HPF Edit Modal
  const [editingHpf, setEditingHpf] = useState<HpfGradingSite | null>(null);
  const [hpfEditCount, setHpfEditCount] = useState<number>(0);
  const [hpfEditNotes, setHpfEditNotes] = useState<string>("");

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetchGradingStageData(caseId);
      setData(res);

      if (res.histologic_type) {
        setSelectedHistologicType(res.histologic_type.confirmed_type || res.histologic_type.proposed_type || "IDC-NST");
        setIsTypeConfirmed(res.histologic_type.is_confirmed);
      }
      if (res.overrides?.tubule) {
        setTubuleOverrideScore(res.overrides.tubule.score);
        setTubuleOverridePercent(res.overrides.tubule.percent ?? null);
        setTubuleJustification(res.overrides.tubule.justification || "");
      }
      if (res.overrides?.pleo) {
        setPleoOverrideScore(res.overrides.pleo.score);
        setPleoJustification(res.overrides.pleo.justification || "");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load Stage 5 Nottingham grading data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [caseId]);

  // Patch Actions
  const handleApprovePatch = async (patch: GradingPatch) => {
    try {
      setActionLoading(true);
      const res = await reviewGradingPatches({
        case_id: caseId,
        action: "update",
        reviews: [
          {
            patch_id: patch.id,
            tubule_percent: patch.user_tubule_percent ?? patch.tubule.tubule_percent,
            tumor_present: patch.user_tumor_present ?? patch.tubule.tumor_present,
            pleomorphism_score: patch.user_pleo_score ?? patch.pleo.pleomorphism_score,
            status: patch.user_tubule_percent !== null && patch.user_tubule_percent !== undefined ? "modified" : "approved",
            notes: patch.user_notes || ""
          }
        ]
      });
      setData(res);
      if (selectedPatch && selectedPatch.id === patch.id) {
        setSelectedPatch(res.patches.find((p) => p.id === patch.id) || null);
      }
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to approve patch");
    } finally {
      setActionLoading(false);
    }
  };

  const handleApproveAllPatches = async () => {
    try {
      setActionLoading(true);
      const res = await reviewGradingPatches({
        case_id: caseId,
        action: "approve_all"
      });
      setData(res);
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to approve all patches");
    } finally {
      setActionLoading(false);
    }
  };

  const handleOpenEditPatch = (patch: GradingPatch) => {
    setEditingPatch(patch);
    setPatchEditTubule(patch.user_tubule_percent ?? patch.tubule.tubule_percent);
    setPatchEditTumorPresent(patch.user_tumor_present ?? patch.tubule.tumor_present);
    setPatchEditPleo(patch.user_pleo_score ?? patch.pleo.pleomorphism_score);
    setPatchEditNotes(patch.user_notes || "");
  };

  const handleSavePatchEdit = async () => {
    if (!editingPatch) return;
    try {
      setActionLoading(true);
      const isModified =
        patchEditTubule !== editingPatch.tubule.tubule_percent ||
        patchEditTumorPresent !== editingPatch.tubule.tumor_present ||
        patchEditPleo !== editingPatch.pleo.pleomorphism_score;

      const res = await reviewGradingPatches({
        case_id: caseId,
        action: "update",
        reviews: [
          {
            patch_id: editingPatch.id,
            tubule_percent: patchEditTubule,
            tumor_present: patchEditTumorPresent,
            pleomorphism_score: patchEditPleo,
            status: isModified ? "modified" : "approved",
            notes: patchEditNotes.trim()
          }
        ]
      });
      setData(res);
      setEditingPatch(null);
      if (selectedPatch && selectedPatch.id === editingPatch.id) {
        setSelectedPatch(res.patches.find((p) => p.id === editingPatch.id) || null);
      }
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to save patch modifications");
    } finally {
      setActionLoading(false);
    }
  };

  // HPF Actions
  const handleApproveHpf = async (hpf: HpfGradingSite) => {
    try {
      setActionLoading(true);
      const res = await reviewGradingHpfs({
        case_id: caseId,
        action: "update",
        reviews: [
          {
            seq: hpf.seq,
            mitotic_count: hpf.user_mitotic_count ?? hpf.mitotic_count,
            status: hpf.user_mitotic_count !== null && hpf.user_mitotic_count !== undefined ? "modified" : "approved",
            notes: hpf.user_notes || ""
          }
        ]
      });
      setData(res);
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to approve HPF");
    } finally {
      setActionLoading(false);
    }
  };

  const handleApproveAllHpfs = async () => {
    try {
      setActionLoading(true);
      const res = await reviewGradingHpfs({
        case_id: caseId,
        action: "approve_all"
      });
      setData(res);
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to approve all HPFs");
    } finally {
      setActionLoading(false);
    }
  };

  const handleOpenEditHpf = (hpf: HpfGradingSite) => {
    setEditingHpf(hpf);
    setHpfEditCount(hpf.user_mitotic_count ?? hpf.mitotic_count);
    setHpfEditNotes(hpf.user_notes || "");
  };

  const handleSaveHpfEdit = async () => {
    if (!editingHpf) return;
    try {
      setActionLoading(true);
      const isModified = hpfEditCount !== editingHpf.mitotic_count;
      const res = await reviewGradingHpfs({
        case_id: caseId,
        action: "update",
        reviews: [
          {
            seq: editingHpf.seq,
            mitotic_count: hpfEditCount,
            status: isModified ? "modified" : "approved",
            notes: hpfEditNotes.trim()
          }
        ]
      });
      setData(res);
      setEditingHpf(null);
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to save HPF modifications");
    } finally {
      setActionLoading(false);
    }
  };

  // Live Reactive Nottingham Sum & Grade Synthesis
  const activeTubuleScore = tubuleOverrideScore ?? (data?.current?.tubule_score || 2);
  const activePleoScore = pleoOverrideScore ?? (data?.current?.pleo_score || 2);
  const activeMitoticScore = data?.current?.mitotic_score || data?.mitotic_summary?.mitotic_score || 2;

  const activeSum = activeTubuleScore + activePleoScore + activeMitoticScore;
  const activeGrade = activeSum <= 5 ? 1 : activeSum <= 7 ? 2 : 3;

  const isTubuleOverridden = tubuleOverrideScore !== null && tubuleOverrideScore !== data?.machine?.tubule_score;
  const isPleoOverridden = pleoOverrideScore !== null && pleoOverrideScore !== data?.machine?.pleo_score;

  const isTubuleJustificationValid = !isTubuleOverridden || tubuleJustification.trim().length >= 10;
  const isPleoJustificationValid = !isPleoOverridden || pleoJustification.trim().length >= 10;

  const revSummary = data?.review_summary;
  const allPatchesApproved = revSummary?.all_patches_reviewed || false;
  const allHpfsApproved = revSummary?.all_hpfs_reviewed || false;

  const canConfirmStage =
    allPatchesApproved &&
    allHpfsApproved &&
    isTypeConfirmed &&
    isTubuleJustificationValid &&
    isPleoJustificationValid &&
    !isSubmitting;

  const handleApplyTubuleOverride = (score: number) => {
    setTubuleOverrideScore(score);
    setIsTubuleEditing(false);
  };

  const handleResetTubule = () => {
    setTubuleOverrideScore(null);
    setTubuleOverridePercent(null);
    setTubuleJustification("");
    setIsTubuleEditing(false);
  };

  const handleApplyPleoOverride = (score: number) => {
    setPleoOverrideScore(score);
    setIsPleoEditing(false);
  };

  const handleResetPleo = () => {
    setPleoOverrideScore(null);
    setPleoJustification("");
    setIsPleoEditing(false);
  };

  const handleConfirmFinalStage = async () => {
    if (!canConfirmStage) return;
    setIsSubmitting(true);
    setSubmitError(null);

    const overridesPayload: Record<string, any> = {};
    if (isTubuleOverridden && tubuleOverrideScore !== null) {
      overridesPayload.tubule = {
        score: tubuleOverrideScore,
        percent: tubuleOverridePercent,
        original_score: data?.machine?.tubule_score,
        justification: tubuleJustification.trim(),
        overridden_at: new Date().toISOString()
      };
    }
    if (isPleoOverridden && pleoOverrideScore !== null) {
      overridesPayload.pleo = {
        score: pleoOverrideScore,
        original_score: data?.machine?.pleo_score,
        justification: pleoJustification.trim(),
        overridden_at: new Date().toISOString()
      };
    }

    try {
      await confirmGradingStage({
        case_id: caseId,
        reviewed_by: "user_pathologist_001",
        histologic_type: selectedHistologicType,
        type_confirmed: true,
        overrides: overridesPayload,
        tubule_score: activeTubuleScore,
        tubule_percent: tubuleOverridePercent ?? data?.current?.tubule_percent ?? data?.machine?.tubule_percent,
        pleo_score: activePleoScore,
        mitotic_score: activeMitoticScore,
        nottingham_sum: activeSum,
        grade: activeGrade
      });

      if (onAdvanceToReport) {
        onAdvanceToReport();
      } else {
        await loadData();
      }
    } catch (err: any) {
      console.error(err);
      setSubmitError(err.message || "Failed to confirm Stage 5 grading");
    } finally {
      setIsSubmitting(false);
    }
  };

  const patches = data?.patches || [];
  const hpfs = data?.hpfs || [];
  const flags = data?.machine?.flags || [];

  const filteredPatches = useMemo(() => {
    if (patchFilter === "all") return patches;
    return patches.filter((p) => p.review_status === patchFilter);
  }, [patches, patchFilter]);

  if (loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 bg-slate-950 text-slate-200">
        <div className="w-10 h-10 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm font-medium">Evaluating Nottingham Parameters with MedGemma 1.5...</p>
        <p className="text-xs text-slate-400 mt-1">Processing 24 normalized 10× evidence patches</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 bg-slate-950 text-slate-200">
        <AlertTriangle className="w-12 h-12 text-rose-400 mb-3" />
        <h3 className="text-base font-semibold text-rose-300">Stage 5 Grading Error</h3>
        <p className="text-xs text-slate-400 mt-1 max-w-md text-center">{error}</p>
        <button
          onClick={loadData}
          className="mt-4 px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded text-xs font-semibold"
        >
          Retry Loading
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Top Header */}
      <header className="px-6 py-3.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-20 backdrop-blur shadow-md">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-sky-950 border border-sky-600 text-[11px] font-bold text-sky-300 uppercase tracking-wider">
              Stage 5: Nottingham Histological Grading
            </span>
            <span className="text-xs text-slate-400">• Dual-Level Pathologist Review</span>
          </div>
          <h1 className="text-lg font-bold text-white mt-0.5">
            Pathologist Confirmation & Architectural Grading Workspace
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            title="Refresh Stage 5 Data"
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-300 hover:text-white text-xs font-semibold transition"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </header>

      {/* Review Gate Status Banner */}
      <div className="mx-6 mt-4 p-4 bg-slate-900 border border-slate-800 rounded-xl shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-sky-400" />
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Mandatory Pathologist Sign-Off Gates
            </h2>
          </div>
          <span className="text-xs text-slate-400">
            Review and approve suggested findings before unlocking Stage 6 (CAP Report)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-3">
          {/* Gate 1: Patch Classification */}
          <div className={`p-3.5 rounded-lg border flex flex-col justify-between ${
            allPatchesApproved
              ? "bg-emerald-950/30 border-emerald-500/50"
              : "bg-amber-950/20 border-amber-500/40"
          }`}>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">1. Patch Classification Gate</span>
                {allPatchesApproved ? (
                  <span className="px-2 py-0.5 rounded bg-emerald-900/60 border border-emerald-500 text-[10px] font-bold text-emerald-300 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Approved
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded bg-amber-900/60 border border-amber-500 text-[10px] font-bold text-amber-300 flex items-center gap-1 animate-pulse">
                    <AlertCircle className="w-3 h-3" /> Review Pending
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 mt-1.5">
                {revSummary?.approved_patches || 0} of {revSummary?.total_patches || 24} patches confirmed
              </p>
            </div>
            <div className="mt-3">
              {!allPatchesApproved && (
                <button
                  onClick={handleApproveAllPatches}
                  disabled={actionLoading}
                  className="w-full py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-bold flex items-center justify-center gap-1.5 shadow transition"
                >
                  <CheckCheck className="w-3.5 h-3.5" /> Approve All 24 Patches
                </button>
              )}
            </div>
          </div>

          {/* Gate 2: HPF Field Review */}
          <div className={`p-3.5 rounded-lg border flex flex-col justify-between ${
            allHpfsApproved
              ? "bg-emerald-950/30 border-emerald-500/50"
              : "bg-amber-950/20 border-amber-500/40"
          }`}>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">2. HPF Field Gate (10 HPFs)</span>
                {allHpfsApproved ? (
                  <span className="px-2 py-0.5 rounded bg-emerald-900/60 border border-emerald-500 text-[10px] font-bold text-emerald-300 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Approved
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded bg-amber-900/60 border border-amber-500 text-[10px] font-bold text-amber-300 flex items-center gap-1 animate-pulse">
                    <AlertCircle className="w-3 h-3" /> Review Pending
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 mt-1.5">
                {revSummary?.approved_hpfs || 0} of {revSummary?.total_hpfs || 10} standard HPFs verified
              </p>
            </div>
            <div className="mt-3">
              {!allHpfsApproved && (
                <button
                  onClick={handleApproveAllHpfs}
                  disabled={actionLoading}
                  className="w-full py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-bold flex items-center justify-center gap-1.5 shadow transition"
                >
                  <CheckCheck className="w-3.5 h-3.5" /> Approve All 10 HPFs
                </button>
              )}
            </div>
          </div>

          {/* Gate 3: Histologic Subtype Confirmation */}
          <div className={`p-3.5 rounded-lg border flex flex-col justify-between ${
            isTypeConfirmed
              ? "bg-emerald-950/30 border-emerald-500/50"
              : "bg-amber-950/20 border-amber-500/40"
          }`}>
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">3. CAP Histologic Subtype</span>
                {isTypeConfirmed ? (
                  <span className="px-2 py-0.5 rounded bg-emerald-900/60 border border-emerald-500 text-[10px] font-bold text-emerald-300 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Confirmed
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded bg-amber-900/60 border border-amber-500 text-[10px] font-bold text-amber-300 flex items-center gap-1 animate-pulse">
                    <AlertCircle className="w-3 h-3" /> Unconfirmed
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 mt-1.5 font-mono">
                {selectedHistologicType}
              </p>
            </div>
            <div className="mt-3">
              {!isTypeConfirmed ? (
                <button
                  onClick={() => setIsTypeConfirmed(true)}
                  className="w-full py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-bold flex items-center justify-center gap-1.5 shadow transition"
                >
                  <ShieldCheck className="w-3.5 h-3.5" /> Confirm Histologic Subtype
                </button>
              ) : (
                <button
                  onClick={() => setIsTypeConfirmed(false)}
                  className="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs font-medium transition"
                >
                  Change Subtype Selection
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Quality Notice / Flags Banner */}
      {flags.length > 0 && (
        <div className="mx-6 mt-4 p-3 bg-amber-950/60 border border-amber-500/50 rounded-lg flex items-center gap-3 text-amber-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <div>
            <span className="font-semibold">Quality Agreement Notice: </span>
            {flags.includes("insufficient_tumor_patches") && "Low tumor patch density (<8 patches with tumor tissue). "}
            {flags.includes("pleo_high_variance") && "High nuclear pleomorphism variance across sampled areas (>30% off mode). "}
            Please inspect and verify the evidence patches.
          </div>
        </div>
      )}

      {/* Main Review Workspace Content */}
      <div className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
        {/* Top 3 Sub-scores Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* 1. Tubule Formation Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between relative shadow-lg">
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-indigo-950 border border-indigo-500 text-indigo-300 font-bold text-xs flex items-center justify-center">
                    T
                  </span>
                  <h2 className="text-sm font-bold text-white">Tubule Formation</h2>
                </div>
                {isTubuleOverridden && (
                  <span className="px-2 py-0.5 rounded bg-amber-950 border border-amber-500 text-[10px] font-bold text-amber-300 flex items-center gap-1">
                    <Edit3 className="w-3 h-3" /> Manually Assigned
                  </span>
                )}
              </div>

              <div className="mt-2 bg-slate-950 rounded-lg p-3 border border-slate-800">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-slate-400">Score & Classification:</span>
                  <span className="text-xl font-extrabold text-white">
                    Score {activeTubuleScore}{" "}
                    <span className="text-xs font-normal text-slate-400">
                      ({activeTubuleScore === 1 ? ">75%" : activeTubuleScore === 2 ? "10-75%" : "<10%"})
                    </span>
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-2 pt-2 border-t border-slate-800/80">
                  <span className="text-xs text-slate-400">Weighted Median:</span>
                  <span className="text-xs font-semibold text-sky-300">
                    {data.current?.tubule_percent ?? data.machine?.tubule_percent ?? 0.0}% glandular area
                  </span>
                </div>
              </div>

              {/* Patch Status Summary */}
              <div className="mt-3 text-xs text-slate-400 flex justify-between items-center">
                <span>Evaluated across {patches.length} evidence patches</span>
                <span className="text-emerald-400 font-medium">
                  {patches.filter((p) => p.review_status !== "suggested").length}/{patches.length} Reviewed
                </span>
              </div>
            </div>

            {/* Override Controls */}
            <div className="mt-4 pt-3 border-t border-slate-800">
              {!isTubuleEditing && !isTubuleOverridden ? (
                <button
                  onClick={() => setIsTubuleEditing(true)}
                  className="w-full py-1.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Edit3 className="w-3.5 h-3.5 text-sky-400" /> Manual Override Tubule Score
                </button>
              ) : (
                <div className="space-y-2.5">
                  <div className="flex items-center gap-1.5">
                    {[1, 2, 3].map((s) => (
                      <button
                        key={s}
                        onClick={() => handleApplyTubuleOverride(s)}
                        className={`flex-1 py-1 text-xs font-bold rounded border transition ${
                          activeTubuleScore === s
                            ? "bg-sky-600 border-sky-400 text-white"
                            : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        Score {s}
                      </button>
                    ))}
                  </div>

                  {isTubuleOverridden && (
                    <div>
                      <label className="text-[10px] text-slate-400 block mb-1">
                        Clinical Justification (min 10 chars):
                      </label>
                      <textarea
                        value={tubuleJustification}
                        onChange={(e) => setTubuleJustification(e.target.value)}
                        placeholder="State clinical reason for overriding tubule formation score..."
                        rows={2}
                        className={`w-full bg-slate-950 border rounded p-2 text-xs text-slate-200 focus:outline-none ${
                          tubuleJustification.trim().length >= 10
                            ? "border-emerald-600 focus:border-emerald-500"
                            : "border-amber-600 focus:border-amber-500"
                        }`}
                      />
                      <div className="flex justify-between items-center mt-1">
                        <span
                          className={`text-[10px] ${
                            tubuleJustification.trim().length >= 10 ? "text-emerald-400" : "text-amber-400"
                          }`}
                        >
                          {tubuleJustification.trim().length}/10 characters
                        </span>
                        <button
                          onClick={handleResetTubule}
                          className="text-[10px] text-slate-400 hover:text-rose-400 underline"
                        >
                          Reset to Patch Mode
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* 2. Nuclear Pleomorphism Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between relative shadow-lg">
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-purple-950 border border-purple-500 text-purple-300 font-bold text-xs flex items-center justify-center">
                    P
                  </span>
                  <h2 className="text-sm font-bold text-white">Nuclear Pleomorphism</h2>
                </div>
                {isPleoOverridden && (
                  <span className="px-2 py-0.5 rounded bg-amber-950 border border-amber-500 text-[10px] font-bold text-amber-300 flex items-center gap-1">
                    <Edit3 className="w-3 h-3" /> Manually Assigned
                  </span>
                )}
              </div>

              <div className="mt-2 bg-slate-950 rounded-lg p-3 border border-slate-800">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-slate-400">Score & Atypia:</span>
                  <span className="text-xl font-extrabold text-white">
                    Score {activePleoScore}{" "}
                    <span className="text-xs font-normal text-slate-400">
                      ({activePleoScore === 1 ? "Small/Uniform" : activePleoScore === 2 ? "Moderate" : "Marked/Vesicular"})
                    </span>
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-2 pt-2 border-t border-slate-800/80">
                  <span className="text-xs text-slate-400">Consensus Mode:</span>
                  <span className="text-xs font-semibold text-purple-300">
                    Worst-area weighted mode
                  </span>
                </div>
              </div>

              {/* Pleo Patch Rationale Sample */}
              <div className="mt-3">
                <div className="text-[11px] font-medium text-slate-400 mb-1.5">
                  Sample Patch Rationale:
                </div>
                <p className="text-xs text-slate-300 italic bg-slate-950/60 p-2.5 rounded border border-slate-800/60 truncate">
                  "{patches[0]?.pleo.rationale || "Moderate nuclear pleomorphism with open chromatin and conspicuous nucleoli."}"
                </p>
              </div>
            </div>

            {/* Override Controls */}
            <div className="mt-4 pt-3 border-t border-slate-800">
              {!isPleoEditing && !isPleoOverridden ? (
                <button
                  onClick={() => setIsPleoEditing(true)}
                  className="w-full py-1.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition"
                >
                  <Edit3 className="w-3.5 h-3.5 text-purple-400" /> Manual Override Pleo Score
                </button>
              ) : (
                <div className="space-y-2.5">
                  <div className="flex items-center gap-1.5">
                    {[1, 2, 3].map((s) => (
                      <button
                        key={s}
                        onClick={() => handleApplyPleoOverride(s)}
                        className={`flex-1 py-1 text-xs font-bold rounded border transition ${
                          activePleoScore === s
                            ? "bg-purple-600 border-purple-400 text-white"
                            : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        Score {s}
                      </button>
                    ))}
                  </div>

                  {isPleoOverridden && (
                    <div>
                      <label className="text-[10px] text-slate-400 block mb-1">
                        Clinical Justification (min 10 chars):
                      </label>
                      <textarea
                        value={pleoJustification}
                        onChange={(e) => setPleoJustification(e.target.value)}
                        placeholder="State clinical reason for overriding nuclear pleomorphism score..."
                        rows={2}
                        className={`w-full bg-slate-950 border rounded p-2 text-xs text-slate-200 focus:outline-none ${
                          pleoJustification.trim().length >= 10
                            ? "border-emerald-600 focus:border-emerald-500"
                            : "border-amber-600 focus:border-amber-500"
                        }`}
                      />
                      <div className="flex justify-between items-center mt-1">
                        <span
                          className={`text-[10px] ${
                            pleoJustification.trim().length >= 10 ? "text-emerald-400" : "text-amber-400"
                          }`}
                        >
                          {pleoJustification.trim().length}/10 characters
                        </span>
                        <button
                          onClick={handleResetPleo}
                          className="text-[10px] text-slate-400 hover:text-rose-400 underline"
                        >
                          Reset to Patch Mode
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* 3. Mitotic Score Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between relative shadow-lg">
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-emerald-950 border border-emerald-500 text-emerald-300 font-bold text-xs flex items-center justify-center">
                    M
                  </span>
                  <h2 className="text-sm font-bold text-white">Mitotic Count (10 HPFs)</h2>
                </div>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  allHpfsApproved ? "bg-emerald-950 border border-emerald-500 text-emerald-300" : "bg-amber-950 border border-amber-500 text-amber-300"
                }`}>
                  {allHpfsApproved ? "10/10 Approved" : "HPF Review Pending"}
                </span>
              </div>

              <div className="mt-2 bg-slate-950 rounded-lg p-3 border border-slate-800">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-slate-400">Score & Rate:</span>
                  <span className="text-xl font-extrabold text-white">
                    Score {activeMitoticScore}{" "}
                    <span className="text-xs font-normal text-slate-400">
                      ({activeMitoticScore === 1 ? "<8 mitoses" : activeMitoticScore === 2 ? "8-15 mitoses" : "≥16 mitoses"})
                    </span>
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-2 pt-2 border-t border-slate-800/80">
                  <span className="text-xs text-slate-400">Total Confirmed Mitoses:</span>
                  <span className="text-xs font-semibold text-emerald-300">
                    {data.mitotic_summary?.total_mitoses ?? 0} mitoses in 10 HPFs
                  </span>
                </div>
              </div>

              <div className="mt-3 space-y-1 text-xs text-slate-400">
                <div className="flex justify-between">
                  <span>Standard Area:</span>
                  <span className="text-slate-200 font-mono">2.157 mm²</span>
                </div>
                <div className="flex justify-between">
                  <span>Standardized Density:</span>
                  <span className="text-slate-200 font-mono">
                    {((data.mitotic_summary?.total_mitoses ?? 0) / 2.157).toFixed(1)} mitoses/mm²
                  </span>
                </div>
              </div>
            </div>

            {/* Reopen Stage 4 Link */}
            <div className="mt-4 pt-3 border-t border-slate-800">
              <button
                onClick={onReopenMitosis}
                className="w-full py-1.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition"
              >
                <RotateCcw className="w-3.5 h-3.5 text-emerald-400" /> Reopen Stage 4 Mitosis Canvas
              </button>
            </div>
          </div>
        </div>

        {/* SECTION 1: Patch-Level Morphological Classification Review */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <Microscope className="w-5 h-5 text-sky-400" />
                <h2 className="text-base font-bold text-white">
                  Patch-Level Morphological Classification Review (24 Patches)
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Inspect 10× normalized evidence patches. Confirm suggested Tubule % and Nuclear Pleomorphism findings or click Edit to customize.
              </p>
            </div>

            {/* Filter Tabs & Bulk Approve */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center bg-slate-950 rounded-lg p-1 border border-slate-800">
                {(["all", "suggested", "approved", "modified"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setPatchFilter(tab)}
                    className={`px-3 py-1 text-xs font-semibold rounded-md capitalize transition ${
                      patchFilter === tab
                        ? "bg-sky-600 text-white shadow"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {tab === "all" ? `All (${patches.length})` : `${tab} (${patches.filter((p) => p.review_status === tab).length})`}
                  </button>
                ))}
              </div>

              {!allPatchesApproved && (
                <button
                  onClick={handleApproveAllPatches}
                  disabled={actionLoading}
                  className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold flex items-center gap-1.5 shadow transition"
                >
                  <CheckCheck className="w-3.5 h-3.5" /> Approve All
                </button>
              )}
            </div>
          </div>

          {/* Patches Interactive Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mt-5">
            {filteredPatches.map((p) => {
              const isApproved = p.review_status === "approved";
              const isModified = p.review_status === "modified";
              const currentTubule = p.user_tubule_percent ?? p.tubule.tubule_percent;
              const currentPleo = p.user_pleo_score ?? p.pleo.pleomorphism_score;
              const currentTumor = p.user_tumor_present ?? p.tubule.tumor_present;

              return (
                <div
                  key={p.id}
                  className={`bg-slate-950 border rounded-xl p-3 flex flex-col justify-between transition group relative ${
                    isModified
                      ? "border-purple-500/70 shadow-[0_0_10px_rgba(168,85,247,0.15)]"
                      : isApproved
                      ? "border-emerald-600/60 bg-emerald-950/10"
                      : "border-slate-800 hover:border-sky-500/60"
                  }`}
                >
                  {/* Thumbnail Image */}
                  <div
                    onClick={() => setSelectedPatch(p)}
                    className="w-full aspect-square bg-slate-900 rounded-lg overflow-hidden relative cursor-pointer group"
                  >
                    <img
                      src={p.image_url?.startsWith("http") ? p.image_url : `${API_BASE}${p.image_url}`}
                      alt={`Patch ${p.id}`}
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                      loading="lazy"
                      onError={(e) => {
                        // Fallback placeholder if image load fails
                        const target = e.target as HTMLImageElement;
                        target.onerror = null;
                        target.src = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="%232e1a38"/><circle cx="256" cy="256" r="160" fill="%23e8d2e6" stroke="%238a3d7c" stroke-width="8"/><text x="50%" y="50%" font-size="28" text-anchor="middle" fill="%23ffffff" dy=".3em">10x Patch ${p.id}</text></svg>`;
                      }}
                    />
                    <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/80 text-[10px] font-mono font-bold text-white shadow">
                      #{p.index}
                    </span>

                    {p.hotspot_id && (
                      <span className="absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded bg-sky-950/90 border border-sky-600/70 text-[9px] font-mono font-bold text-sky-300 shadow">
                        {p.hotspot_id}
                      </span>
                    )}

                    {/* Status Badge on Image */}
                    <span className={`absolute bottom-1.5 right-1.5 px-1.5 py-0.5 rounded text-[9px] font-bold flex items-center gap-1 shadow ${
                      isModified
                        ? "bg-purple-950/90 text-purple-300 border border-purple-500"
                        : isApproved
                        ? "bg-emerald-950/90 text-emerald-300 border border-emerald-500"
                        : "bg-amber-950/90 text-amber-300 border border-amber-500"
                    }`}>
                      {isModified ? "Modified" : isApproved ? "Approved" : "Suggested"}
                    </span>
                  </div>

                  {/* Findings Breakdown */}
                  <div className="mt-3 space-y-1.5 text-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-[11px]">Tubule:</span>
                      <span className={`font-bold ${currentTumor ? "text-sky-300" : "text-slate-500 line-through"}`}>
                        {currentTumor ? `${currentTubule}%` : "No Tumor"}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 text-[11px]">Pleomorphism:</span>
                      <span className="font-bold text-purple-300">
                        Score {currentPleo}
                      </span>
                    </div>

                    {p.tissue_density !== undefined && p.tissue_density !== null && (
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-[10px]">Tissue Density:</span>
                        <span className="font-semibold text-emerald-400 text-[10px]">
                          {Math.round(p.tissue_density * 100)}%
                        </span>
                      </div>
                    )}

                    {p.user_notes && (
                      <p className="text-[10px] text-slate-400 italic bg-slate-900/90 p-1.5 rounded border border-slate-800 truncate">
                        "{p.user_notes}"
                      </p>
                    )}
                  </div>

                  {/* Patch Action Buttons */}
                  <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center gap-1.5">
                    {!isApproved && !isModified ? (
                      <button
                        onClick={() => handleApprovePatch(p)}
                        disabled={actionLoading}
                        className="flex-1 py-1 px-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[11px] font-bold flex items-center justify-center gap-1 shadow transition"
                        title="Confirm suggested findings"
                      >
                        <Check className="w-3 h-3" /> Approve
                      </button>
                    ) : (
                      <span className="flex-1 py-1 text-center text-[11px] font-semibold text-emerald-400 flex items-center justify-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Reviewed
                      </span>
                    )}

                    <button
                      onClick={() => handleOpenEditPatch(p)}
                      disabled={actionLoading}
                      className="p-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded border border-slate-700 transition"
                      title="Edit patch tubule or pleomorphism score"
                    >
                      <Edit3 className="w-3.5 h-3.5 text-sky-400" />
                    </button>

                    <button
                      onClick={() => setSelectedPatch(p)}
                      className="p-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded border border-slate-700 transition"
                      title="Inspect high-resolution patch"
                    >
                      <Eye className="w-3.5 h-3.5 text-slate-300" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SECTION 2: HPF-Level Field Review (10 Standardized Fields) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-emerald-400" />
                <h2 className="text-base font-bold text-white">
                  HPF-Level Field Review (10 Standardized Fields • 2.157 mm²)
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Verify mitotic counts across the 10 virtual high-power fields (HPFs) placed in highest density hotspot areas.
              </p>
            </div>

            {!allHpfsApproved && (
              <button
                onClick={handleApproveAllHpfs}
                disabled={actionLoading}
                className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold flex items-center gap-1.5 shadow transition"
              >
                <CheckCheck className="w-3.5 h-3.5" /> Approve All 10 HPFs
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5 mt-5">
            {hpfs.map((h) => {
              const isApproved = h.review_status === "approved" || h.review_status === "modified";
              const currentCount = h.user_mitotic_count ?? h.mitotic_count;

              return (
                <div
                  key={h.seq}
                  className={`bg-slate-950 border rounded-xl p-3 flex flex-col justify-between transition ${
                    h.review_status === "modified"
                      ? "border-purple-500/70"
                      : isApproved
                      ? "border-emerald-600/60 bg-emerald-950/10"
                      : "border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[10px] font-mono font-bold text-sky-300">
                        HPF #{h.seq}
                      </span>
                      <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                        isApproved ? "text-emerald-400 bg-emerald-950/60" : "text-amber-400 bg-amber-950/60"
                      }`}>
                        {h.review_status === "modified" ? "Modified" : isApproved ? "Approved" : "Suggested"}
                      </span>
                    </div>

                    <div className="text-center py-2">
                      <div className="text-2xl font-black text-white font-mono">
                        {currentCount}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        mitoses ({(currentCount / 0.2157).toFixed(1)}/mm²)
                      </div>
                    </div>
                  </div>

                  <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center gap-1.5">
                    {!isApproved ? (
                      <button
                        onClick={() => handleApproveHpf(h)}
                        disabled={actionLoading}
                        className="flex-1 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[11px] font-bold flex items-center justify-center gap-1 transition"
                      >
                        <Check className="w-3 h-3" /> Approve
                      </button>
                    ) : (
                      <span className="flex-1 py-1 text-center text-[10px] font-semibold text-emerald-400 flex items-center justify-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> Verified
                      </span>
                    )}
                    <button
                      onClick={() => handleOpenEditHpf(h)}
                      disabled={actionLoading}
                      className="p-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded border border-slate-700 transition"
                      title="Adjust mitotic count"
                    >
                      <Edit3 className="w-3.5 h-3.5 text-sky-400" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Histologic Subtype Classification Card (MANDATORY GATE) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white">CAP Histologic Subtype Classification</h2>
                {isTypeConfirmed ? (
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-950 border border-emerald-500 text-emerald-300 text-xs font-bold flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Confirmed
                  </span>
                ) : (
                  <span className="px-2.5 py-0.5 rounded-full bg-rose-950 border border-rose-500 text-rose-300 text-xs font-bold flex items-center gap-1 animate-pulse">
                    <AlertCircle className="w-3.5 h-3.5" /> Action Required Before Confirming Stage 5
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Multi-image consensus across top 8 tumor patches. Pathologist confirmation is strictly required.
              </p>
            </div>

            <div>
              {!isTypeConfirmed ? (
                <button
                  onClick={() => setIsTypeConfirmed(true)}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold flex items-center gap-2 shadow-lg shadow-emerald-950 transition"
                >
                  <ShieldCheck className="w-4 h-4" /> Confirm Histologic Subtype
                </button>
              ) : (
                <button
                  onClick={() => setIsTypeConfirmed(false)}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs font-medium transition"
                >
                  Edit Subtype Selection
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-5">
            <div className="lg:col-span-1 space-y-3">
              <label className="text-xs font-semibold text-slate-300 block">
                Primary Histologic Subtype:
              </label>
              <select
                value={selectedHistologicType}
                onChange={(e) => {
                  setSelectedHistologicType(e.target.value);
                  setIsTypeConfirmed(false);
                }}
                disabled={isTypeConfirmed}
                className={`w-full bg-slate-950 border rounded-lg p-2.5 text-xs text-white focus:outline-none ${
                  isTypeConfirmed
                    ? "border-emerald-600/70 bg-emerald-950/20"
                    : "border-slate-700 focus:border-sky-500"
                }`}
              >
                {HISTOLOGIC_TYPE_OPTIONS.map((opt) => (
                  <option key={opt.id} value={opt.id}>
                    {opt.label}
                  </option>
                ))}
              </select>

              {data.histologic_type?.differential && data.histologic_type.differential.length > 0 && (
                <div className="mt-3">
                  <span className="text-[11px] text-slate-400 block mb-1">Differential Diagnoses Considered:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {data.histologic_type.differential.map((d, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-300"
                      >
                        {d}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-2 bg-slate-950/80 border border-slate-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-sky-400" />
                <span className="text-xs font-bold text-slate-300">MedGemma Morphological Rationale:</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                  Confidence: {data.histologic_type?.confidence || "High"}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {data.histologic_type?.rationale ||
                  "Invasive ductal carcinoma characterized by cohesive malignant cell cords and irregular tubular formations infiltrating fibrous desmoplastic stroma."}
              </p>
            </div>
          </div>
        </div>

        {/* Live Nottingham Histological Grade Synthesis Card */}
        <div className="bg-gradient-to-r from-slate-900 via-sky-950/40 to-slate-900 border border-sky-800/40 rounded-xl p-6 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <span className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                Overall Histological Synthesis
              </span>
              <h2 className="text-2xl font-black text-white mt-0.5 flex items-center gap-3">
                Nottingham Histological Grade {activeGrade}
                <span className="text-sm font-normal text-slate-300">
                  ({activeGrade === 1 ? "Well Differentiated" : activeGrade === 2 ? "Moderately Differentiated" : "Poorly Differentiated"})
                </span>
              </h2>
            </div>

            <div className="flex items-center gap-3">
              <div className="px-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-right">
                <div className="text-[10px] text-slate-400">Nottingham Sum (T + P + M)</div>
                <div className="text-xl font-mono font-bold text-sky-300">{activeSum} / 9</div>
              </div>
            </div>
          </div>

          {/* Formula Display */}
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-3 bg-slate-950/70 p-4 rounded-lg border border-slate-800 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Tubule Score (T):</span>
              <span className="font-mono font-bold text-white">
                {activeTubuleScore} {isTubuleOverridden && <span className="text-amber-400">*</span>}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Pleomorphism Score (P):</span>
              <span className="font-mono font-bold text-white">
                {activePleoScore} {isPleoOverridden && <span className="text-amber-400">*</span>}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Mitotic Score (M):</span>
              <span className="font-mono font-bold text-white">{activeMitoticScore}</span>
            </div>
            <div className="flex items-center justify-between border-t md:border-t-0 md:border-l md:pl-3 border-slate-800">
              <span className="text-sky-400 font-semibold">Sum = {activeSum} →</span>
              <span className="font-bold text-sky-300">Grade {activeGrade}</span>
            </div>
          </div>

          {/* Diagnostic Summary Narrative */}
          <div className="mt-5">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Diagnostic Summary Narrative:
            </h4>
            <p className="text-xs text-slate-200 leading-relaxed bg-slate-950/90 p-4 rounded-lg border border-slate-800/80">
              {data.narrative ||
                `Invasive breast carcinoma (${selectedHistologicType}), Nottingham Histological Grade ${activeGrade} (Total Score ${activeSum}/9). Tubule formation is evaluated across 24 evidence patches (${data.current?.tubule_percent ?? data.machine?.tubule_percent ?? 22}%, Score ${activeTubuleScore}). Nuclear pleomorphism demonstrates atypia (Score ${activePleoScore}). Mitotic activity is evaluated across 10 standardized high-power fields (Score ${activeMitoticScore}).`}
            </p>
          </div>

          {/* Final Confirmation Bar */}
          <div className="mt-6 pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div className="text-xs text-slate-400">
              {!allPatchesApproved ? (
                <span className="text-amber-400 flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4" /> Please approve all 24 image patches above (Gate 1).
                </span>
              ) : !allHpfsApproved ? (
                <span className="text-amber-400 flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4" /> Please approve all 10 High-Power Fields above (Gate 2).
                </span>
              ) : !isTypeConfirmed ? (
                <span className="text-amber-400 flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4" /> Please click "Confirm Histologic Subtype" above (Gate 3).
                </span>
              ) : !isTubuleJustificationValid || !isPleoJustificationValid ? (
                <span className="text-amber-400 flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4" /> Please provide at least 10 characters justification for manual score overrides.
                </span>
              ) : (
                <span className="text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" /> All dual-level review gates satisfied. Ready to proceed to CAP Report Generation.
                </span>
              )}
            </div>

            <button
              onClick={handleConfirmFinalStage}
              disabled={!canConfirmStage}
              className={`px-6 py-3 rounded-lg text-xs font-bold flex items-center gap-2 shadow-lg transition ${
                canConfirmStage
                  ? "bg-sky-600 hover:bg-sky-500 text-white shadow-sky-950 cursor-pointer"
                  : "bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed"
              }`}
            >
              {isSubmitting ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Finalizing Nottingham Grade...
                </>
              ) : (
                <>
                  Confirm Nottingham Grade & Advance to CAP Report (Stage 6) <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>

          {submitError && (
            <div className="mt-3 text-xs text-rose-400 bg-rose-950/60 p-2.5 rounded border border-rose-800">
              {submitError}
            </div>
          )}
        </div>
      </div>

      {/* Patch Edit Modal */}
      {editingPatch && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg flex flex-col shadow-2xl overflow-hidden">
            <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Edit3 className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-bold text-white">
                  Pathologist Review: Patch #{editingPatch.index} ({editingPatch.id})
                </h3>
              </div>
              <button
                onClick={() => setEditingPatch(null)}
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="w-32 h-32 bg-black rounded-lg overflow-hidden border border-slate-800 mx-auto">
                <img
                  src={editingPatch.image_url?.startsWith("http") ? editingPatch.image_url : `${API_BASE}${editingPatch.image_url}`}
                  alt={`Patch ${editingPatch.id}`}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.onerror = null;
                    target.src = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><rect width="512" height="512" fill="%232e1a38"/><circle cx="256" cy="256" r="160" fill="%23e8d2e6" stroke="%238a3d7c" stroke-width="8"/><text x="50%" y="50%" font-size="28" text-anchor="middle" fill="%23ffffff" dy=".3em">10x Patch ${editingPatch.id}</text></svg>`;
                  }}
                />
              </div>

              {/* Tubule Formation Percentage Slider */}
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-semibold text-slate-300">Tubule Formation (%)</label>
                  <span className="text-sm font-mono font-bold text-sky-400">{patchEditTubule}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={patchEditTubule}
                  onChange={(e) => setPatchEditTubule(Number(e.target.value))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Score 3 (&lt;10%)</span>
                  <span>Score 2 (10-75%)</span>
                  <span>Score 1 (&gt;75%)</span>
                </div>
              </div>

              {/* Tumor Present Toggle */}
              <div className="flex items-center justify-between bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                <div>
                  <span className="text-xs font-semibold text-slate-200 block">Invasive Tumor Present</span>
                  <span className="text-[10px] text-slate-400">Include this patch in glandular area calculations</span>
                </div>
                <button
                  type="button"
                  onClick={() => setPatchEditTumorPresent(!patchEditTumorPresent)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    patchEditTumorPresent ? "bg-sky-600" : "bg-slate-800"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      patchEditTumorPresent ? "translate-x-4" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>

              {/* Nuclear Pleomorphism Score (1, 2, 3) */}
              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-2">
                <label className="text-xs font-semibold text-slate-300 block">Nuclear Pleomorphism Score</label>
                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3].map((score) => (
                    <button
                      key={score}
                      type="button"
                      onClick={() => setPatchEditPleo(score as 1 | 2 | 3)}
                      className={`py-2 px-3 rounded-lg border text-xs font-bold transition flex flex-col items-center gap-0.5 ${
                        patchEditPleo === score
                          ? "bg-purple-900/60 border-purple-500 text-purple-200 shadow-md shadow-purple-950/40"
                          : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <span>Score {score}</span>
                      <span className="text-[10px] font-normal text-slate-500">
                        {score === 1 ? "Small/Uniform" : score === 2 ? "Moderate" : "Marked"}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Pathologist Observation Notes */}
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">
                  Morphological Notes (Optional)
                </label>
                <textarea
                  value={patchEditNotes}
                  onChange={(e) => setPatchEditNotes(e.target.value)}
                  placeholder="Record morphological rationale for this patch..."
                  rows={2}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>

            <div className="p-4 bg-slate-950 border-t border-slate-800 flex justify-end gap-3">
              <button
                onClick={() => setEditingPatch(null)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition"
              >
                Cancel
              </button>
              <button
                onClick={handleSavePatchEdit}
                disabled={actionLoading}
                className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-bold shadow-lg shadow-sky-950 transition flex items-center gap-1.5"
              >
                <Check className="w-4 h-4" /> Save Patch Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* HPF Edit Modal */}
      {editingHpf && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md flex flex-col shadow-2xl overflow-hidden">
            <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-bold text-white">
                  Adjust HPF #{editingHpf.seq} Mitotic Count
                </h3>
              </div>
              <button
                onClick={() => setEditingHpf(null)}
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-center space-y-3">
                <label className="text-xs font-semibold text-slate-300 block">
                  Confirmed Mitotic Figures in HPF #{editingHpf.seq}
                </label>
                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={() => setHpfEditCount(Math.max(0, hpfEditCount - 1))}
                    className="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white flex items-center justify-center font-bold text-lg"
                  >
                    <Minus className="w-4 h-4" />
                  </button>
                  <span className="text-3xl font-black font-mono text-emerald-400 min-w-[3rem]">
                    {hpfEditCount}
                  </span>
                  <button
                    onClick={() => setHpfEditCount(hpfEditCount + 1)}
                    className="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white flex items-center justify-center font-bold text-lg"
                  >
                    <Plus className="w-4 h-4" />
                  </button>
                </div>
                <div className="text-[11px] text-slate-400">
                  Standardized Field Density: {(hpfEditCount / 0.2157).toFixed(1)} mitoses/mm²
                </div>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800 space-y-1.5">
                <label className="text-xs font-semibold text-slate-300 block">
                  Pathologist Note
                </label>
                <input
                  type="text"
                  value={hpfEditNotes}
                  onChange={(e) => setHpfEditNotes(e.target.value)}
                  placeholder="e.g. Verified prophase and metaphase figures..."
                  className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setEditingHpf(null)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveHpfEdit}
                  disabled={actionLoading}
                  className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold shadow transition"
                >
                  Save HPF Review
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Single Patch Detailed Inspection Modal */}
      {selectedPatch && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl flex flex-col shadow-2xl overflow-hidden">
            <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Microscope className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-bold text-white">
                  Evidence Patch #{selectedPatch.index} ({selectedPatch.id}) • 10× Magnification
                </h3>
              </div>
              <button
                onClick={() => setSelectedPatch(null)}
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="w-full aspect-square max-h-72 bg-black rounded-lg overflow-hidden border border-slate-800 mx-auto">
                <img
                  src={`${API_BASE}${selectedPatch.image_url}`}
                  alt={`Patch ${selectedPatch.id}`}
                  className="w-full h-full object-contain"
                />
              </div>

              {/* Hotspot & Tissue Density Metadata Strip */}
              <div className="flex items-center justify-between bg-slate-950 px-3 py-2 rounded-lg border border-slate-800 text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">Hotspot Origin:</span>
                  <span className="font-mono font-bold text-sky-300 bg-sky-950 px-1.5 py-0.5 rounded border border-sky-800">
                    {selectedPatch.hotspot_id || "Direct Sampling"}
                  </span>
                </div>
                {selectedPatch.tissue_density !== undefined && selectedPatch.tissue_density !== null && (
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-slate-400">Tissue Density:</span>
                    <span className="font-mono font-bold text-emerald-400">
                      {Math.round(selectedPatch.tissue_density * 100)}%
                    </span>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Tubule Formation</span>
                  <span className="text-lg font-bold text-sky-400">
                    {selectedPatch.user_tubule_percent ?? selectedPatch.tubule.tubule_percent}%
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-1">
                    Confidence: {selectedPatch.tubule.confidence}
                  </span>
                </div>

                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Nuclear Pleomorphism</span>
                  <span className="text-lg font-bold text-purple-400">
                    Score {selectedPatch.user_pleo_score ?? selectedPatch.pleo.pleomorphism_score}
                  </span>
                  <span className="text-[10px] text-slate-500 block mt-1">
                    Confidence: {selectedPatch.pleo.confidence}
                  </span>
                </div>
              </div>

              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs">
                <span className="text-[10px] text-slate-400 block mb-1">Pleomorphism Rationale:</span>
                <p className="text-slate-300 italic">"{selectedPatch.pleo.rationale}"</p>
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                <button
                  onClick={() => {
                    handleOpenEditPatch(selectedPatch);
                    setSelectedPatch(null);
                  }}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-sky-300 rounded text-xs font-semibold flex items-center gap-1.5 transition"
                >
                  <Edit3 className="w-3.5 h-3.5" /> Modify Findings
                </button>

                {selectedPatch.review_status === "suggested" ? (
                  <button
                    onClick={() => {
                      handleApprovePatch(selectedPatch);
                    }}
                    disabled={actionLoading}
                    className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-bold flex items-center gap-1.5 shadow transition"
                  >
                    <Check className="w-3.5 h-3.5" /> Approve Patch
                  </button>
                ) : (
                  <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-4 h-4" /> Pathologist Approved
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

