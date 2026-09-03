"use client";

import React, { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { 
  Microscope, 
  CheckCircle2, 
  XCircle, 
  Plus, 
  RotateCcw, 
  ArrowRight, 
  ArrowLeft,
  Layers, 
  Sliders, 
  Activity, 
  ShieldAlert, 
  Crosshair, 
  Loader2, 
  Sparkles,
  Info,
  Check,
  RefreshCw,
  Eye,
  EyeOff,
  AlertTriangle,
  ChevronRight,
  ChevronLeft,
  Maximize2,
  FileCheck2,
  MapPin,
  Compass
} from "lucide-react";
import { 
  MitosisStageData, 
  MitosisCandidate, 
  VirtualHpfSite, 
  MitoticScoreSummary, 
  fetchMitosisStageData, 
  recomputeMitosis, 
  addPathologistMitosis, 
  bulkRejectUnreviewedMitosis, 
  replaceMitosisHpfs, 
  confirmMitosisStage,
  API_BASE 
} from "@/lib/api";
import { OpenSeadragonViewer, ViewerDetectionMarker, ViewerHotspot } from "./OpenSeadragonViewer";
import { MitosisGallery } from "./MitosisGallery";

type WorkflowPhase = "overview" | "field_review" | "completion_summary";

interface MitosisViewerProps {
  caseId: string;
  mppX?: number;
  mppY?: number;
  imageWidthPx?: number;
  imageHeightPx?: number;
  onRefreshCase?: () => void;
  tileUrlTemplate?: string | null;
}

export function MitosisViewer({
  caseId,
  mppX = 0.25,
  mppY = 0.25,
  imageWidthPx = 20000,
  imageHeightPx = 20000,
  onRefreshCase,
  tileUrlTemplate = null
}: MitosisViewerProps) {
  const [data, setData] = useState<MitosisStageData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [candidates, setCandidates] = useState<MitosisCandidate[]>([]);
  const [hpfs, setHpfs] = useState<VirtualHpfSite[]>([]);
  const [summary, setSummary] = useState<MitoticScoreSummary>({
    count_total: 0,
    n_hpf: 10,
    area_mm2: 2.157,
    per_mm2: 0.0,
    classic_per_10hpf: 0.0,
    mitotic_score: 1
  });

  // 3-Phase Clinical Workflow State: (a) overview -> (b) field_review -> (c) completion_summary
  const [workflowPhase, setWorkflowPhase] = useState<WorkflowPhase>("overview");
  const [activeHpfSeq, setActiveHpfSeq] = useState<number>(1);
  const [approvedFields, setApprovedFields] = useState<Record<number, boolean>>({});
  const [showCalculationDetails, setShowCalculationDetails] = useState<boolean>(false);

  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [stainMode, setStainMode] = useState<"norm" | "orig">("norm");
  const [filterMode, setFilterMode] = useState<"all" | "unreviewed" | "mitosis" | "not_mitosis">("all");
  const [isPinningMode, setIsPinningMode] = useState<boolean>(false);
  const [showHpfCircles, setShowHpfCircles] = useState<boolean>(true);
  const [showCandidateMarkers, setShowCandidateMarkers] = useState<boolean>(true);

  // Debounce ref for live score recomputation
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch initial data
  const loadStageData = async () => {
    try {
      setLoading(true);
      setError(null);
      const stageData = await fetchMitosisStageData(caseId);
      setData(stageData);
      setCandidates(stageData.candidates || []);
      setHpfs(stageData.hpfs || []);
      setSummary(stageData.summary || {
        count_total: 0,
        n_hpf: 10,
        area_mm2: 2.157,
        per_mm2: 0.0,
        classic_per_10hpf: 0.0,
        mitotic_score: 1
      });
      if (stageData.candidates && stageData.candidates.length > 0) {
        setSelectedCandidateId(stageData.candidates[0].id);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load Mitosis Stage data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStageData();
  }, [caseId]);

  // Auto-poll while mitosis stage is queued or running
  useEffect(() => {
    if (!data || data.status === "running" || data.status === "queued") {
      const timer = setInterval(() => {
        fetchMitosisStageData(caseId).then((stageData) => {
          if (stageData && stageData.status !== "queued") {
            setData(stageData);
            setCandidates(stageData.candidates || []);
            setHpfs(stageData.hpfs || []);
            if (stageData.summary) setSummary(stageData.summary);
            if (stageData.candidates && stageData.candidates.length > 0 && !selectedCandidateId) {
              setSelectedCandidateId(stageData.candidates[0].id);
            }
          }
        }).catch(() => {});
      }, 3000);
      return () => clearInterval(timer);
    }
  }, [caseId, data?.status, selectedCandidateId]);

  // Client-side optimistic Nottingham Score recalculation
  const computeClientScore = useCallback((currentCandidates: MitosisCandidate[], currentHpfs: VirtualHpfSite[]) => {
    const mitoses = currentCandidates.filter(c => c.label === "mitosis");
    let totalInHpfs = 0;
    const updatedHpfs = currentHpfs.map(hpf => {
      const cx = hpf.center_um[0];
      const cy = hpf.center_um[1];
      const r = hpf.radius_um || 262.0;
      const count = mitoses.filter(m => {
        const dx = m.centroid_um[0] - cx;
        const dy = m.centroid_um[1] - cy;
        return (dx * dx + dy * dy) <= (r * r);
      }).length;
      return { ...hpf, count };
    });

    // Unique mitoses inside any HPF
    const uniqueContained = new Set<string>();
    mitoses.forEach(m => {
      for (const hpf of currentHpfs) {
        const dx = m.centroid_um[0] - hpf.center_um[0];
        const dy = m.centroid_um[1] - hpf.center_um[1];
        if ((dx * dx + dy * dy) <= (hpf.radius_um * hpf.radius_um)) {
          uniqueContained.add(m.id);
          break;
        }
      }
    });

    totalInHpfs = uniqueContained.size;
    const nHpf = currentHpfs.length || 10;
    const singleHpfArea = Math.PI * Math.pow(262.0 / 1000.0, 2); // 0.21565 mm2
    const totalArea = Math.max(0.001, nHpf * singleHpfArea);
    const density = totalInHpfs / totalArea;
    const classicEquivalent = density * 2.74;

    let score = 1;
    if (density >= 7.30) score = 3;
    else if (density >= 3.65) score = 2;

    return {
      updatedHpfs,
      newSummary: {
        count_total: totalInHpfs,
        n_hpf: nHpf,
        area_mm2: Number(totalArea.toFixed(3)),
        per_mm2: Number(density.toFixed(2)),
        classic_per_10hpf: Number(classicEquivalent.toFixed(1)),
        mitotic_score: score
      }
    };
  }, []);

  // Server debounced sync
  const syncWithServer = useCallback((
    updatedCandidates: MitosisCandidate[], 
    updatedHpfs: VirtualHpfSite[], 
    auditToggle?: { id: string; from: string; to: string }
  ) => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(async () => {
      try {
        const labelsMap: Record<string, string> = {};
        updatedCandidates.forEach(c => { labelsMap[c.id] = c.label; });
        const res = await recomputeMitosis({
          case_id: caseId,
          candidate_labels: labelsMap,
          hpfs: updatedHpfs,
          audit_toggle: auditToggle
        });
        if (res && res.summary) {
          setSummary(res.summary);
          setHpfs(res.hpfs);
        }
      } catch (err) {
        console.error("Debounced recompute sync error:", err);
      }
    }, 300);
  }, [caseId]);

  // Toggle candidate label
  const handleToggleCandidate = (id: string, newLabel: "mitosis" | "not_mitosis" | "unreviewed") => {
    const cand = candidates.find(c => c.id === id);
    if (!cand) return;
    const oldLabel = cand.label;

    const updated = candidates.map(c => {
      if (c.id === id) {
        return { ...c, label: newLabel, label_source: "pathologist" as const };
      }
      return c;
    });

    setCandidates(updated);

    // Optimistic UI score update
    const { updatedHpfs, newSummary } = computeClientScore(updated, hpfs);
    setHpfs(updatedHpfs);
    setSummary(newSummary);

    // Sync debounced to server with audit event
    syncWithServer(updated, updatedHpfs, { id, from: oldLabel, to: newLabel });
  };

  const handleAddCandidateFromClick = async (x_um: number, y_um: number) => {
    try {
      const res = await addPathologistMitosis(caseId, [x_um, y_um], "mitosis");
      if (res && res.candidate) {
        const updated = [...candidates, res.candidate];
        setCandidates(updated);
        const { updatedHpfs, newSummary } = computeClientScore(updated, hpfs);
        setHpfs(updatedHpfs);
        setSummary(newSummary);
        setIsPinningMode(false);
      }
    } catch (err) {
      console.error("Add candidate error:", err);
    }
  };

  // Resolve Active HPF
  const activeHpf = useMemo(() => {
    return hpfs.find(h => h.seq === activeHpfSeq) || hpfs[0] || null;
  }, [hpfs, activeHpfSeq]);

  // Candidates filtered strictly to the active HPF (with slight boundary margin)
  const activeFieldCandidates = useMemo(() => {
    if (!activeHpf) return candidates;
    const [cx, cy] = activeHpf.center_um;
    const r = (activeHpf.radius_um || 262.0) * 1.15; // 15% margin
    return candidates.filter(cand => {
      const dx = cand.centroid_um[0] - cx;
      const dy = cand.centroid_um[1] - cy;
      return (dx * dx + dy * dy) <= (r * r);
    });
  }, [candidates, activeHpf]);

  // Step to Next Field or Finish with Smart Auto-Reject for unreviewed candidates in active field
  const handleApproveFieldAndNext = () => {
    // Auto-reject any unreviewed candidates remaining in the current active field
    const unreviewedInField = activeFieldCandidates.filter(c => c.label === "unreviewed");
    let currentCandidates = candidates;
    if (unreviewedInField.length > 0) {
      const unreviewedIds = new Set(unreviewedInField.map(c => c.id));
      currentCandidates = candidates.map(c => {
        if (unreviewedIds.has(c.id)) {
          return { ...c, label: "not_mitosis" as const, label_source: "pathologist" as const };
        }
        return c;
      });
      setCandidates(currentCandidates);
      const { updatedHpfs, newSummary } = computeClientScore(currentCandidates, hpfs);
      setHpfs(updatedHpfs);
      setSummary(newSummary);
      syncWithServer(currentCandidates, updatedHpfs);
    }

    setApprovedFields(prev => ({ ...prev, [activeHpfSeq]: true }));
    if (activeHpfSeq < (hpfs.length || 10)) {
      setActiveHpfSeq(activeHpfSeq + 1);
    } else {
      // All 10 fields approved -> transition to Phase (c)
      setWorkflowPhase("completion_summary");
    }
  };

  // Start Guided Review
  const handleStartGuidedReview = (startSeq: number = 1) => {
    setActiveHpfSeq(startSeq);
    setWorkflowPhase("field_review");
  };

  // Convert HPFs to ViewerHotspots for Whole-Slide OpenSeadragon
  const hpfHotspots: ViewerHotspot[] = useMemo(() => {
    if (!showHpfCircles) return [];
    return hpfs.map((hpf) => {
      const [cx, cy] = hpf.center_um;
      const r = hpf.radius_um || 262.0;
      const poly: number[][] = [];
      const numPts = 32;
      for (let i = 0; i <= numPts; i++) {
        const theta = (i * 2 * Math.PI) / numPts;
        poly.push([cx + r * Math.cos(theta), cy + r * Math.sin(theta)]);
      }
      return {
        id: `hpf_${hpf.seq}`,
        polygon_um: poly,
        area_mm2: 0.216,
        prob_mean: hpf.count > 0 ? 0.95 : 0.4,
        source: `HPF #${hpf.seq} (${hpf.count} mitoses)`
      };
    });
  }, [hpfs, showHpfCircles]);

  // Convert candidate detections to ViewerDetectionMarkers
  const detectionMarkers: ViewerDetectionMarker[] = useMemo(() => {
    if (!showCandidateMarkers) return [];
    return candidates.map((cand) => {
      let inHpf = false;
      if (activeHpf) {
        const dx = cand.centroid_um[0] - activeHpf.center_um[0];
        const dy = cand.centroid_um[1] - activeHpf.center_um[1];
        if (dx * dx + dy * dy <= (activeHpf.radius_um || 262.0) * (activeHpf.radius_um || 262.0)) {
          inHpf = true;
        }
      }
      return {
        id: cand.id,
        x_um: cand.centroid_um[0],
        y_um: cand.centroid_um[1],
        label: cand.label,
        conf: cand.ver_conf || cand.det_conf,
        in_hpf: inHpf
      };
    });
  }, [candidates, activeHpf, showCandidateMarkers]);

  // Jump viewer to candidate
  const handleJumpToCandidate = (candidate: MitosisCandidate) => {
    setSelectedCandidateId(candidate.id);
  };

  // Keyboard Navigation: j, k (cards), m (mitosis), x (reject), Enter (approve field)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)) return;

      if (workflowPhase === "field_review") {
        const filtered = activeFieldCandidates.filter(c => {
          if (filterMode === "all") return true;
          return c.label === filterMode;
        });

        const currentIndex = filtered.findIndex(c => c.id === selectedCandidateId);

        if (e.key === "j" || e.key === "ArrowDown") {
          e.preventDefault();
          if (filtered.length > 0) {
            const nextIdx = (currentIndex + 1) % filtered.length;
            setSelectedCandidateId(filtered[nextIdx].id);
          }
        } else if (e.key === "k" || e.key === "ArrowUp") {
          e.preventDefault();
          if (filtered.length > 0) {
            const prevIdx = (currentIndex - 1 + filtered.length) % filtered.length;
            setSelectedCandidateId(filtered[prevIdx].id);
          }
        } else if (e.key === "m") {
          e.preventDefault();
          if (selectedCandidateId) {
            const target = candidates.find(c => c.id === selectedCandidateId);
            if (target) {
              handleToggleCandidate(target.id, target.label === "mitosis" ? "unreviewed" : "mitosis");
            }
          }
        } else if (e.key === "x") {
          e.preventDefault();
          if (selectedCandidateId) {
            const target = candidates.find(c => c.id === selectedCandidateId);
            if (target) {
              handleToggleCandidate(target.id, target.label === "not_mitosis" ? "unreviewed" : "not_mitosis");
            }
          }
        } else if (e.key === "a" || e.key === "A") {
          e.preventDefault();
          setShowCandidateMarkers(prev => !prev);
        } else if (e.key === "Enter") {
          e.preventDefault();
          handleApproveFieldAndNext();
        }
      } else {
        if (e.key === "a" || e.key === "A") {
          e.preventDefault();
          setShowCandidateMarkers(prev => !prev);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [workflowPhase, activeFieldCandidates, selectedCandidateId, filterMode, candidates, activeHpfSeq, hpfs.length]);

  // Bulk reject remaining unreviewed candidates
  const handleBulkReject = async () => {
    try {
      setLoading(true);
      const res = await bulkRejectUnreviewedMitosis(caseId);
      setData(res);
      setCandidates(res.candidates || []);
      setHpfs(res.hpfs || []);
      setSummary(res.summary || summary);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to bulk reject unreviewed candidates.");
    } finally {
      setLoading(false);
    }
  };

  // Confirm Stage 4 Safety Gate & Advance
  const handleConfirmStage = async () => {
    try {
      setSubmitting(true);
      setError(null);
      await confirmMitosisStage(caseId);
      if (onRefreshCase) onRefreshCase();
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to confirm Mitosis Stage. Ensure high-confidence candidates are reviewed.");
    } finally {
      setSubmitting(false);
    }
  };

  // Calculate unreviewed count for candidates >= 0.50 conf
  const unreviewedHighConf = candidates.filter(
    c => c.label === "unreviewed" && ((c.ver_conf || c.det_conf || 0) >= 0.50)
  ).length;

  if (loading && !data) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-slate-950 text-slate-200">
        <Loader2 className="w-8 h-8 animate-spin text-emerald-400 mb-3" />
        <span className="text-sm font-medium">Loading Mitosis Detection & Virtual HPFs...</span>
      </div>
    );
  }

  const opticalPatchUrl = `${API_BASE}/api/v1/stages/mitosis/${caseId}/hpfs/${activeHpf?.seq || 1}/thumbnail?mag=40x&stain=${stainMode}&v=${data?.stage_execution_id || 'v3'}`;
  const wholeSlideThumbnailUrl = `${API_BASE}/api/v1/cases/${caseId}/thumbnail`;

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* TOP HEADER: Clean Navigation & Mitotic Score Summary */}
      <header className="px-4 py-2 bg-slate-900/95 border-b border-slate-800 shrink-0 flex flex-col gap-2 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Microscope className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-sm text-slate-100 tracking-tight">
                  Stage v4.3: Mitosis Scoring (40× Objective / 400× Optical)
                </h1>
                {workflowPhase === "overview" && (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-semibold">
                    Whole-Slide Overview
                  </span>
                )}
                {workflowPhase === "field_review" && (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800/60 font-semibold">
                    Field #{activeHpfSeq} of {hpfs.length || 10}
                  </span>
                )}
                {workflowPhase === "completion_summary" && (
                  <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/60 font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Review Complete
                  </span>
                )}
              </div>
              <span className="text-[11px] text-slate-400">
                10-HPF Systematic Review & Nottingham Mitotic Indexing (WHO 5th Ed)
              </span>
            </div>
          </div>

          {/* Clean Clinical Score Summary */}
          <div className="flex items-center gap-3">
            {/* Total Mitoses Pill */}
            <div className="bg-slate-950 px-3 py-1 rounded-lg border border-slate-800 text-xs flex items-center gap-2">
              <span className="text-slate-400">Total:</span>
              <span className="font-bold text-emerald-400 font-mono text-sm">
                {summary.count_total}
              </span>
              <span className="text-slate-500 text-[11px]">in 10 HPFs</span>
            </div>

            {/* Mitotic Score Badge */}
            <div
              className={`px-3 py-1 rounded-lg font-bold text-xs shadow flex items-center gap-1.5 ${
                summary.mitotic_score === 3
                  ? "bg-rose-950/80 text-rose-300 border border-rose-600/70"
                  : summary.mitotic_score === 2
                  ? "bg-amber-950/80 text-amber-300 border border-amber-600/70"
                  : "bg-emerald-950/80 text-emerald-300 border border-emerald-600/70"
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              Nottingham Mitotic Score: {summary.mitotic_score} ({summary.mitotic_score === 3 ? "High ≥20" : summary.mitotic_score === 2 ? "Mod 10-19" : "Low 0-9"})
            </div>

            {/* Refresh Data Button */}
            <button
              onClick={loadStageData}
              disabled={loading}
              className="p-1 rounded-lg bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700 transition"
              title="Refresh Mitosis Stage Data"
            >
              <RotateCcw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>

            {/* Calculation Details Popover Button */}
            <button
              onClick={() => setShowCalculationDetails(!showCalculationDetails)}
              className="p-1 rounded-lg bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700 transition"
              title="View Standardized Density & Area Math"
            >
              <Info className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Calculation Details Dropdown (Hidden by default) */}
        {showCalculationDetails && (
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-xs text-slate-300 flex items-center justify-between animate-fadeIn">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">10-HPF Standard Area</span>
                <span className="font-bold font-mono text-slate-200">{summary.area_mm2.toFixed(3)} mm²</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">Standardized Density</span>
                <span className="font-bold font-mono text-sky-400">{summary.per_mm2.toFixed(1)} mitoses/mm²</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">Classic 10 HPF Equiv (2.74 mm²)</span>
                <span className="font-bold font-mono text-slate-200">{summary.classic_per_10hpf.toFixed(0)} mitoses</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px] uppercase font-semibold">WHO Scoring Rules</span>
                <span className="text-slate-400">&lt;3.65/mm² = 1 | 3.65-7.30 = 2 | ≥7.30 = 3</span>
              </div>
            </div>
            <button
              onClick={() => setShowCalculationDetails(false)}
              className="text-xs text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded"
            >
              Close
            </button>
          </div>
        )}

        {/* STEPPER BAR: 10 Field Navigation Pills & Toolbar */}
        <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
          <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
            <button
              onClick={() => setWorkflowPhase("overview")}
              className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all border ${
                workflowPhase === "overview"
                  ? "bg-emerald-600 text-white border-emerald-400 shadow"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700 hover:text-white"
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>Slide Overview</span>
            </button>

            <div className="h-4 w-px bg-slate-800 mx-1" />

            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mr-1 shrink-0">
              Fields:
            </span>
            {hpfs.map((hpf) => {
              const isActive = workflowPhase === "field_review" && hpf.seq === activeHpfSeq;
              const isApproved = approvedFields[hpf.seq];

              return (
                <button
                  key={`hpf-step-${hpf.seq}`}
                  onClick={() => handleStartGuidedReview(hpf.seq)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all shrink-0 border ${
                    isActive
                      ? "bg-sky-600 text-white border-sky-400 shadow-md font-bold ring-2 ring-sky-400/30"
                      : isApproved
                      ? "bg-emerald-950/40 text-emerald-300 border-emerald-700/50 hover:bg-emerald-900/40"
                      : "bg-slate-800 text-slate-300 border-slate-700/70 hover:bg-slate-700 hover:text-white"
                  }`}
                >
                  <span>Field {hpf.seq}</span>
                  {isApproved ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <span className={`text-[10px] px-1 rounded ${isActive ? "bg-sky-800 text-sky-100" : "bg-slate-900 text-slate-400 font-mono"}`}>
                      {hpf.count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Stepper Controls & Toolbar */}
          <div className="flex items-center gap-2 shrink-0">
            {workflowPhase === "field_review" && (
              <div className="flex items-center bg-slate-800 rounded-lg p-0.5 border border-slate-700">
                <button
                  onClick={() => setActiveHpfSeq(Math.max(1, activeHpfSeq - 1))}
                  disabled={activeHpfSeq === 1}
                  className="p-1 rounded text-slate-300 hover:text-white hover:bg-slate-700 disabled:opacity-40 disabled:hover:bg-transparent"
                  title="Previous Field"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="px-2 text-xs font-mono font-semibold text-slate-200">
                  {activeHpfSeq} / {hpfs.length || 10}
                </span>
                <button
                  onClick={() => setActiveHpfSeq(Math.min(hpfs.length || 10, activeHpfSeq + 1))}
                  disabled={activeHpfSeq === (hpfs.length || 10)}
                  className="p-1 rounded text-slate-300 hover:text-white hover:bg-slate-700 disabled:opacity-40 disabled:hover:bg-transparent"
                  title="Next Field"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Annotation Mask Toggle Button */}
            <button
              onClick={() => setShowCandidateMarkers(!showCandidateMarkers)}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold flex items-center gap-1.5 transition-all border ${
                showCandidateMarkers
                  ? "bg-emerald-950/90 text-emerald-300 border-emerald-600/80 hover:bg-emerald-900/60 shadow-sm"
                  : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-slate-200"
              }`}
              title="Toggle Mitotic Figure Annotations & Green Dots (Hotkey: A)"
            >
              {showCandidateMarkers ? (
                <Eye className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <EyeOff className="w-3.5 h-3.5 text-slate-500" />
              )}
              <span>{showCandidateMarkers ? "Mitosis Dots: ON" : "Mitosis Dots: OFF"}</span>
              <kbd className="text-[9px] font-mono px-1 py-0.2 bg-slate-900/80 rounded border border-slate-700 text-slate-400">
                A
              </kbd>
            </button>

            {/* Stain Switcher */}
            <div className="flex items-center bg-slate-800 p-0.5 rounded border border-slate-700">
              <button
                onClick={() => setStainMode("norm")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 transition-all ${
                  stainMode === "norm" ? "bg-emerald-700 text-white font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Sparkles className="w-3 h-3 text-amber-300" /> Norm H&E
              </button>
              <button
                onClick={() => setStainMode("orig")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all ${
                  stainMode === "orig" ? "bg-slate-700 text-white font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Orig
              </button>
            </div>

            {/* Pin Mitosis Mode Button */}
            <button
              onClick={() => setIsPinningMode(!isPinningMode)}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold flex items-center gap-1 transition-all border ${
                isPinningMode
                  ? "bg-amber-600 text-white border-amber-400 shadow-md animate-pulse"
                  : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700 hover:text-white"
              }`}
            >
              <Crosshair className="w-3.5 h-3.5" />
              {isPinningMode ? "Click Slide to Pin" : "+ Pin (40×)"}
            </button>
          </div>
        </div>
      </header>

      {/* WORKSPACE VIEWS */}
      <div className="flex-1 flex overflow-hidden">
        {/* ============================================================ */}
        {/* PHASE (a): WHOLE-SLIDE MACRO OVERVIEW & AUTOMATED ANALYSIS */}
        {/* ============================================================ */}
        {workflowPhase === "overview" && (
          <div className="flex-1 flex overflow-hidden">
            {/* Left/Center: Macro Whole Slide Viewer with 10 HPFs */}
            <div className="flex-1 relative bg-black flex flex-col overflow-hidden">
              <OpenSeadragonViewer
                caseId={caseId}
                tileUrlTemplate={tileUrlTemplate}
                imageWidthPx={imageWidthPx}
                imageHeightPx={imageHeightPx}
                mppX={mppX}
                mppY={mppY}
                layer={stainMode}
                hotspots={hpfHotspots}
                showHotspotMask={showHpfCircles}
                onSelectHotspot={(id) => {
                  const seq = parseInt(id.replace("hpf_", ""), 10);
                  if (!isNaN(seq)) handleStartGuidedReview(seq);
                }}
                detectionMarkers={detectionMarkers}
                showCandidateMarkers={showCandidateMarkers}
                isAddingRoiMode={isPinningMode}
                onAddRoiClick={handleAddCandidateFromClick}
                className="w-full h-full"
              />
            </div>

            {/* Right Drawer: Automated Analysis Findings & Start Review Action */}
            <div className="w-96 shrink-0 h-full bg-slate-900 border-l border-slate-800 p-4 flex flex-col justify-between overflow-y-auto">
              <div className="space-y-4">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <FileCheck2 className="w-5 h-5 text-emerald-400" />
                  <div>
                    <h2 className="font-bold text-sm text-slate-100">Automated Triage Complete</h2>
                    <span className="text-xs text-slate-400">10 Standardized HPF Sites Mapped</span>
                  </div>
                </div>

                {/* Score Summary Box */}
                <div className="bg-slate-950 rounded-xl p-3.5 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Automated Mitosis Count:</span>
                    <span className="font-mono font-bold text-emerald-400 text-sm">{summary.count_total} mitoses</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Total Evaluated Area:</span>
                    <span className="font-mono font-bold text-slate-200">2.157 mm² (10 HPFs)</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Calculated Proliferation:</span>
                    <span className="font-mono font-bold text-sky-400">{summary.per_mm2.toFixed(1)} /mm²</span>
                  </div>
                  <div className="border-t border-slate-800 pt-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300">Initial Mitotic Score:</span>
                    <span className={`px-2 py-0.5 rounded font-bold text-xs ${
                      summary.mitotic_score === 3 ? "bg-rose-950 text-rose-300 border border-rose-700" : "bg-emerald-950 text-emerald-300 border border-emerald-700"
                    }`}>
                      Score {summary.mitotic_score} ({summary.mitotic_score === 3 ? "High ≥20" : "Low/Mod"})
                    </span>
                  </div>
                </div>

                {/* 10-HPF List */}
                <div className="space-y-1.5">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                    HPF Fields Mapped:
                  </span>
                  <div className="space-y-1 max-h-56 overflow-y-auto pr-1">
                    {hpfs.map((h) => (
                      <div
                        key={h.seq}
                        onClick={() => handleStartGuidedReview(h.seq)}
                        className="flex items-center justify-between p-2 rounded-lg bg-slate-950/60 hover:bg-slate-800 border border-slate-800/80 cursor-pointer transition text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className="w-5 h-5 rounded-full bg-slate-800 flex items-center justify-center font-mono font-bold text-[11px] text-slate-300">
                            {h.seq}
                          </span>
                          <span className="text-slate-300 font-medium">Field #{h.seq}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-emerald-400 font-mono font-bold">{h.count} mitoses</span>
                          <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Start Guided Review CTA */}
              <div className="pt-4 border-t border-slate-800">
                <button
                  onClick={() => handleStartGuidedReview(1)}
                  className="w-full py-3 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl shadow-lg flex items-center justify-center gap-2 transition active:scale-[0.98]"
                >
                  <Microscope className="w-4 h-4" />
                  <span>Start 10-HPF Guided Review (Field 1)</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* PHASE (b): DEDICATED HIGH-RESOLUTION 40X HPF FIELD INSPECTION */}
        {/* ============================================================ */}
        {workflowPhase === "field_review" && (
          <div className="flex-1 flex overflow-hidden">
            {/* Left / Center: High-Resolution 40x Optical Patch + Picture-in-Picture Minimap */}
            <div className="flex-1 relative bg-slate-950 flex flex-col items-center justify-center overflow-hidden select-none">
              {/* Field Microscope Stage Canvas */}
              <div className="relative w-full h-full flex items-center justify-center p-6">
                {/* 40x High-Res Optical Patch Container */}
                <div className="relative w-[520px] h-[520px] rounded-2xl overflow-hidden shadow-2xl border-2 border-slate-700 bg-slate-900 flex items-center justify-center">
                  <img
                    key={`${caseId}-${activeHpf?.seq || 1}-${stainMode}`}
                    src={opticalPatchUrl}
                    alt={`HPF ${activeHpfSeq} 40x View`}
                    className="w-full h-full object-cover select-none pointer-events-none"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="100%" height="100%" fill="%231e1b4b"/><text x="50%" y="50%" fill="%23a855f7" text-anchor="middle" font-size="16">Field ${activeHpfSeq} 40x Optical Patch</text></svg>`;
                    }}
                  />

                  {/* SVG Microscopic Reticle Circle & Candidate Mitosis Markers Overlay */}
                  <svg className="absolute inset-0 w-full h-full pointer-events-none">
                    {/* Standardized HPF Boundary (524 µm / 0.2157 mm2) with comfortable margin */}
                    <circle
                      cx="260"
                      cy="260"
                      r="236"
                      fill="none"
                      stroke="#10b981"
                      strokeWidth="2"
                      strokeDasharray="8 4"
                      className="drop-shadow-md"
                    />
                    
                    {/* Crosshairs inside reticle */}
                    <line x1="260" y1="24" x2="260" y2="496" stroke="rgba(16, 185, 129, 0.25)" strokeWidth="1" />
                    <line x1="24" y1="260" x2="496" y2="260" stroke="rgba(16, 185, 129, 0.25)" strokeWidth="1" />

                    {/* Reticle Central Dot */}
                    <circle cx="260" cy="260" r="2.5" fill="#10b981" />

                    {/* Candidate Mitosis Pins on 40x Optical Patch (Toggleable) */}
                    {showCandidateMarkers && activeFieldCandidates.map((cand) => {
                      if (!activeHpf) return null;
                      const [cx, cy] = activeHpf.center_um;
                      const dx_um = cand.centroid_um[0] - cx;
                      const dy_um = cand.centroid_um[1] - cy;
                      
                      // Precise physical radius to pixel reticle mapping (262 µm -> 236 px)
                      const reticleRadiusPx = 236.0;
                      const hpfRadiusUm = activeHpf.radius_um || 262.0;
                      const pxX = 260 + (dx_um / hpfRadiusUm) * reticleRadiusPx;
                      const pxY = 260 + (dy_um / hpfRadiusUm) * reticleRadiusPx;

                      const isSelected = cand.id === selectedCandidateId;
                      const color = cand.label === "mitosis" ? "#10b981" : (cand.label === "not_mitosis" ? "#64748b" : "#f59e0b");

                      return (
                        <g
                          key={`cand-patch-${cand.id}`}
                          className="pointer-events-auto cursor-pointer"
                          onClick={() => setSelectedCandidateId(cand.id)}
                        >
                          <circle
                            cx={pxX}
                            cy={pxY}
                            r={isSelected ? 7.5 : 5}
                            fill={color}
                            stroke={isSelected ? "#38bdf8" : "#0f172a"}
                            strokeWidth={isSelected ? 2.5 : 1.5}
                            className={isSelected ? "filter drop-shadow-[0_0_6px_rgba(56,189,248,0.9)]" : "hover:stroke-sky-300 hover:stroke-[2] transition-colors"}
                          />
                          {isSelected && (
                            <circle
                              cx={pxX}
                              cy={pxY}
                              r={12}
                              fill="none"
                              stroke="#38bdf8"
                              strokeWidth="1.5"
                              strokeDasharray="3 3"
                            />
                          )}
                        </g>
                      );
                    })}
                  </svg>

                  {/* On-Stage Floating Controls */}
                  <div className="absolute top-3 left-3 bg-slate-900/90 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-700 text-[11px] font-mono text-slate-200 shadow">
                    HPF #{activeHpfSeq} • 40× Objective (400× Optical / 0.25 µm/px)
                  </div>
                  <div className="absolute top-3 right-3 bg-slate-900/90 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-700 text-[11px] font-mono text-emerald-400 font-bold shadow flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    {activeHpf?.count || 0} Mitoses
                  </div>

                  {/* Stage Quick Mask Toggle */}
                  <button
                    onClick={() => setShowCandidateMarkers(!showCandidateMarkers)}
                    className="absolute bottom-3 right-3 bg-slate-900/90 hover:bg-slate-800 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-700 text-[11px] font-medium text-slate-300 hover:text-white shadow flex items-center gap-1.5 transition-all"
                    title="Toggle annotations on/off (Key: A)"
                  >
                    {showCandidateMarkers ? <Eye className="w-3.5 h-3.5 text-emerald-400" /> : <EyeOff className="w-3.5 h-3.5 text-slate-500" />}
                    <span>{showCandidateMarkers ? "Hide Annotations" : "Show Annotations"}</span>
                  </button>
                </div>

                {/* Picture-in-Picture Macro Biopsy Minimap (Never lose position sense) */}
                <div className="absolute bottom-6 left-6 bg-slate-900/95 backdrop-blur-md rounded-xl p-2.5 border border-slate-800 shadow-2xl flex flex-col gap-1.5 w-44 select-none z-10">
                  <div className="flex items-center justify-between text-[10px] font-bold text-slate-300 uppercase tracking-wider">
                    <span className="flex items-center gap-1.5 text-sky-400">
                      <MapPin className="w-3.5 h-3.5" /> Biopsy Location
                    </span>
                    <span className="text-slate-500 font-mono text-[9px]">Field #{activeHpfSeq}</span>
                  </div>
                  <div className="relative w-full h-44 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center p-1">
                    {(() => {
                      const slideW = data?.slide?.width_px || imageWidthPx || 20000;
                      const slideH = data?.slide?.height_px || imageHeightPx || 20000;
                      const mppXVal = data?.slide?.mpp_x || mppX || 0.25;
                      const mppYVal = data?.slide?.mpp_y || mppY || 0.25;
                      const totalSlideW_um = slideW * mppXVal;
                      const totalSlideH_um = slideH * mppYVal;
                      const slideAspect = slideW / slideH;

                      const beaconLeftPct = activeHpf
                        ? Math.min(96, Math.max(4, (activeHpf.center_um[0] / totalSlideW_um) * 100))
                        : 50;
                      const beaconTopPct = activeHpf
                        ? Math.min(96, Math.max(4, (activeHpf.center_um[1] / totalSlideH_um) * 100))
                        : 50;

                      return (
                        <div
                          className="relative h-full max-w-full flex items-center justify-center"
                          style={{ aspectRatio: `${slideAspect}` }}
                        >
                          <img
                            src={wholeSlideThumbnailUrl}
                            alt="Biopsy overview"
                            className="w-full h-full object-fill rounded pointer-events-none"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160"><rect width="100%" height="100%" fill="%230f172a"/><text x="50%" y="50%" fill="%2394a3b8" text-anchor="middle" font-size="10">Biopsy Core</text></svg>`;
                            }}
                          />
                          {/* Active HPF Beacon on Minimap */}
                          {activeHpf && (
                            <div
                              className="absolute transform -translate-x-1/2 -translate-y-1/2 pointer-events-none z-10"
                              style={{
                                left: `${beaconLeftPct}%`,
                                top: `${beaconTopPct}%`
                              }}
                            >
                              <div className="w-4 h-4 rounded-full bg-emerald-400 border-2 border-white shadow-[0_0_12px_#10b981] animate-pulse flex items-center justify-center">
                                <div className="w-1.5 h-1.5 rounded-full bg-slate-950" />
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Candidate Gallery Scoped to Active HPF */}
            <div className="w-96 shrink-0 h-full">
              <MitosisGallery
                caseId={caseId}
                candidates={activeFieldCandidates}
                selectedCandidateId={selectedCandidateId}
                onSelectCandidate={(cand) => {
                  setSelectedCandidateId(cand.id);
                }}
                onToggleCandidate={handleToggleCandidate}
                onJumpToCandidate={handleJumpToCandidate}
                stainMode={stainMode}
                filterMode={filterMode}
                onSetFilterMode={setFilterMode}
                fieldSeq={activeHpfSeq}
                totalFields={hpfs.length || 10}
                onApproveFieldAndNext={handleApproveFieldAndNext}
              />
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* PHASE (c): REVIEW COMPLETION & VERIFIED SCORE SUMMARY */}
        {/* ============================================================ */}
        {workflowPhase === "completion_summary" && (
          <div className="flex-1 flex flex-col items-center justify-center p-8 bg-slate-950 overflow-y-auto">
            <div className="max-w-2xl w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
              {/* Header */}
              <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
                <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                  <CheckCircle2 className="w-7 h-7" />
                </div>
                <div>
                  <h2 className="font-bold text-lg text-slate-100">10-HPF Systematic Review Complete</h2>
                  <span className="text-xs text-slate-400">All 10 High-Power Fields verified by pathologist</span>
                </div>
              </div>

              {/* Final Score Card */}
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col items-center justify-center">
                  <span className="text-xs text-slate-400 uppercase font-semibold">Total Verified Mitoses</span>
                  <span className="font-bold font-mono text-2xl text-emerald-400 mt-1">
                    {summary.count_total}
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5">Across 10 HPFs</span>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col items-center justify-center">
                  <span className="text-xs text-slate-400 uppercase font-semibold">Standard Density</span>
                  <span className="font-bold font-mono text-2xl text-sky-400 mt-1">
                    {summary.per_mm2.toFixed(1)}
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5">mitoses / mm²</span>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col items-center justify-center">
                  <span className="text-xs text-slate-400 uppercase font-semibold">Nottingham Mitotic Score</span>
                  <span className={`font-bold text-2xl mt-1 ${
                    summary.mitotic_score === 3 ? "text-rose-400" : summary.mitotic_score === 2 ? "text-amber-400" : "text-emerald-400"
                  }`}>
                    Score {summary.mitotic_score}
                  </span>
                  <span className="text-[11px] text-slate-400 mt-0.5">
                    {summary.mitotic_score === 3 ? "High Proliferation (≥20)" : summary.mitotic_score === 2 ? "Moderate (10-19)" : "Low (0-9)"}
                  </span>
                </div>
              </div>

              {/* 10-Field Mitotic Distribution Grid */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                  Field-by-Field Breakdown:
                </span>
                <div className="grid grid-cols-5 gap-2">
                  {hpfs.map((h) => (
                    <div key={h.seq} className="bg-slate-950 p-2 rounded-lg border border-slate-800 flex items-center justify-between text-xs">
                      <span className="text-slate-400">Field #{h.seq}</span>
                      <span className="font-mono font-bold text-emerald-400">{h.count}m</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-4 border-t border-slate-800">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setWorkflowPhase("field_review")}
                    className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition flex items-center gap-1.5"
                  >
                    <RotateCcw className="w-3.5 h-3.5" /> Review Fields Again
                  </button>
                  <button
                    onClick={() => setWorkflowPhase("overview")}
                    className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition flex items-center gap-1.5"
                  >
                    <Compass className="w-3.5 h-3.5" /> Slide Overview
                  </button>
                </div>

                <button
                  onClick={handleConfirmStage}
                  disabled={submitting}
                  className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl shadow-lg text-sm flex items-center gap-2 transition active:scale-[0.98]"
                >
                  {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                  <span>Confirm Stage 4 & Proceed</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
