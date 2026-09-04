"use client";

import React, { useEffect, useState } from "react";
import { 
  Flame, 
  CheckCircle2, 
  XCircle, 
  Plus, 
  Sliders, 
  ShieldAlert, 
  ArrowRight, 
  Info,
  Loader2,
  Trash2,
  Crosshair,
  ZoomIn,
  Eye,
  X,
  Layers, 
  Activity,
  RotateCcw,
  MapPin
} from "lucide-react";
import { API_BASE, retryStage } from "@/lib/api";
import { OpenSeadragonViewer } from "./OpenSeadragonViewer";

interface HotspotItem {
  id: string;
  polygon_um: number[][];
  area_mm2: number;
  prob_mean: number;
  prob_max: number;
  source: string;
  excluded: boolean;
  exclude_reason?: string | null;
  thumbnail_url?: string | null;
}

interface TriageData {
  case_id: string;
  stage_execution_id: string;
  status: string;
  heatmap_png_uri: string | null;
  heatmap_direct_url?: string | null;
  prob_grid_uri: string | null;
  grid: {
    origin_um: number[];
    stride_um: number;
    nx: number;
    ny: number;
  };
  machine_hotspots: HotspotItem[];
  effective_hotspots: HotspotItem[];
  review_edits: any[];
  model_versions?: Record<string, string>;
}

interface TriageViewerProps {
  caseId: string;
  mppX?: number;
  mppY?: number;
  imageWidthPx?: number;
  imageHeightPx?: number;
  onRefreshCase?: () => void;
  tileUrlTemplate?: string | null;
  onAdvanceToMitosis?: () => void;
}

export function TriageViewer({
  caseId,
  mppX = 0.25,
  mppY = 0.25,
  imageWidthPx = 2048,
  imageHeightPx = 2048,
  onRefreshCase,
  tileUrlTemplate = null,
  onAdvanceToMitosis
}: TriageViewerProps) {
  const [data, setData] = useState<TriageData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [reprocessing, setReprocessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.6);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [showHotspotMask, setShowHotspotMask] = useState<boolean>(true);
  const [hotspotsList, setHotspotsList] = useState<HotspotItem[]>([]);
  const [selectedHotspotId, setSelectedHotspotId] = useState<string | null>(null);
  const [previewHotspot, setPreviewHotspot] = useState<HotspotItem | null>(null);
  const [modalMag, setModalMag] = useState<"10x" | "20x" | "40x">("10x");
  const [stainMode, setStainMode] = useState<"norm" | "orig">("norm");
  const [isAddingRoiMode, setIsAddingRoiMode] = useState<boolean>(false);
  const [noInvasiveTumor, setNoInvasiveTumor] = useState<boolean>(false);
  const [excludeReasonInput, setExcludeReasonInput] = useState<{ [id: string]: string }>({});

  const fetchTriageData = async (silent: boolean = false) => {
    try {
      if (!silent) setLoading(true);
      const res = await fetch(`${API_BASE}/api/v1/stages/triage/${caseId}?_t=${Date.now()}`, {
        headers: { "X-User-Role": "pathologist" }
      });
      if (!res.ok) {
        throw new Error(`Failed to fetch triage data (Status: ${res.status})`);
      }
      const json = await res.json();
      setData(json);
      setHotspotsList(json.effective_hotspots || []);
    } catch (err: any) {
      if (!silent) setError(err.message || "Failed to load triage data");
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const handleReprocessTriage = async () => {
    try {
      setReprocessing(true);
      await retryStage(caseId, "triage");
      if (onRefreshCase) onRefreshCase();
      await fetchTriageData();
    } catch (err: any) {
      console.error(err);
      setError(`Failed to re-process triage: ${err.message}`);
    } finally {
      setReprocessing(false);
    }
  };

  useEffect(() => {
    fetchTriageData();
  }, [caseId]);

  // Auto-poll while triage is running or queued
  useEffect(() => {
    if (!data || data.status === "running" || data.status === "queued") {
      const timer = setInterval(() => {
        fetchTriageData(true);
      }, 3000);
      return () => clearInterval(timer);
    }
  }, [caseId, data?.status]);

  const handleExcludeHotspot = (id: string) => {
    const reason = excludeReasonInput[id] || "Pathologist excluded";
    setHotspotsList((prev) =>
      prev.map((h) => (h.id === id ? { ...h, excluded: true, exclude_reason: reason } : h))
    );
  };

  const handleRestoreHotspot = (id: string) => {
    setHotspotsList((prev) =>
      prev.map((h) => (h.id === id ? { ...h, excluded: false, exclude_reason: null } : h))
    );
  };

  const handleDeleteHotspot = (id: string) => {
    setHotspotsList((prev) => prev.filter((h) => h.id !== id));
  };

  const handleAddRoiFromClick = (x_um: number, y_um: number) => {
    const half_um = 300.0;
    const polygon = [
      [Number((x_um - half_um).toFixed(2)), Number((y_um - half_um).toFixed(2))],
      [Number((x_um + half_um).toFixed(2)), Number((y_um - half_um).toFixed(2))],
      [Number((x_um + half_um).toFixed(2)), Number((y_um + half_um).toFixed(2))],
      [Number((x_um - half_um).toFixed(2)), Number((y_um + half_um).toFixed(2))],
      [Number((x_um - half_um).toFixed(2)), Number((y_um - half_um).toFixed(2))]
    ];

    const userCount = hotspotsList.filter((h) => h.id.startsWith("user_")).length;
    const newId = `user_${(userCount + 1).toString().padStart(2, "0")}`;
    const newHs: HotspotItem = {
      id: newId,
      polygon_um: polygon,
      area_mm2: 0.36,
      prob_mean: 0.88,
      prob_max: 0.95,
      source: "pathologist_added",
      excluded: false
    };

    setHotspotsList((prev) => [...prev, newHs]);
    setIsAddingRoiMode(false);
    setSelectedHotspotId(newId);
    setPreviewHotspot(newHs);
  };

  const handleSaveDraftEdits = async () => {
    try {
      setSubmitting(true);
      const edits = hotspotsList.map((h) => {
        if (h.excluded) {
          return { op: "exclude", id: h.id, reason: h.exclude_reason };
        } else if (h.source === "pathologist_added") {
          return { op: "add", id: h.id, polygon_um: h.polygon_um, area_mm2: h.area_mm2 };
        }
        return { op: "modify", id: h.id, polygon_um: h.polygon_um };
      });

      const res = await fetch(`${API_BASE}/api/v1/stages/triage/edits`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Role": "pathologist"
        },
        body: JSON.stringify({ case_id: caseId, edits })
      });

      if (!res.ok) {
        throw new Error("Failed to save draft edits");
      }
    } catch (err: any) {
      alert(`Error saving edits: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleConfirmStage = async () => {
    try {
      setSubmitting(true);
      await handleSaveDraftEdits();

      const res = await fetch(`${API_BASE}/api/v1/stages/triage/confirm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Role": "pathologist"
        },
        body: JSON.stringify({
          case_id: caseId,
          no_invasive_tumor: noInvasiveTumor,
          reviewed_by: "pathologist_01"
        })
      });

      if (!res.ok) {
        throw new Error("Failed to confirm stage execution");
      }

      const json = await res.json();
      if (onAdvanceToMitosis) {
        onAdvanceToMitosis();
      } else if (onRefreshCase) {
        onRefreshCase();
      }
    } catch (err: any) {
      console.error(err);
      setError(`Error confirming triage: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const activeHotspotsCount = hotspotsList.filter((h) => !h.excluded).length;
  const totalAreaMm2 = hotspotsList
    .filter((h) => !h.excluded)
    .reduce((sum, h) => sum + (h.area_mm2 || 0), 0);

  const isHeatmapAvailable = Boolean(
    data?.heatmap_png_uri ||
    data?.status === "awaiting_review" ||
    data?.status === "done" ||
    data?.status === "confirmed"
  );
  const heatmapOverlayUri = isHeatmapAvailable
    ? `${API_BASE}/api/v1/stages/triage/${caseId}/heatmap?v=${data?.stage_execution_id || ''}`
    : null;

  if (loading) {
    return (
      <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center text-slate-400">
        <Loader2 className="w-8 h-8 animate-spin text-sky-500 mb-2" />
        <p className="text-sm font-medium">Extracting 10× Path Foundation Tumor Hotspots...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center text-rose-400">
        <ShieldAlert className="w-10 h-10 mb-2" />
        <p className="text-sm font-semibold">{error}</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex bg-slate-950 overflow-hidden">
      {/* Left Main Viewport */}
      <div className="flex-1 relative">
        <OpenSeadragonViewer
          caseId={caseId}
          mppX={mppX}
          mppY={mppY || mppX}
          imageWidthPx={imageWidthPx}
          imageHeightPx={imageHeightPx}
          overlayImageUri={heatmapOverlayUri}
          overlayOpacity={heatmapOpacity}
          showOverlay={showHeatmap}
          showHotspotMask={showHotspotMask}
          hotspots={hotspotsList}
          selectedHotspotId={selectedHotspotId}
          onSelectHotspot={setSelectedHotspotId}
          isAddingRoiMode={isAddingRoiMode}
          onAddRoiClick={handleAddRoiFromClick}
          tileUrlTemplate={tileUrlTemplate}
        />

        {/* Interactive Click-to-Add ROI Floating Banner */}
        {isAddingRoiMode && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 bg-sky-950/95 border-2 border-sky-400 rounded-full px-5 py-2.5 shadow-2xl flex items-center space-x-3 backdrop-blur animate-pulse">
            <Crosshair className="w-4 h-4 text-sky-400 animate-spin" />
            <span className="text-xs font-bold text-sky-100">
              Click anywhere on the Whole Slide Image to add a custom 10× HPF candidate ROI
            </span>
            <button
              onClick={() => setIsAddingRoiMode(false)}
              className="px-2.5 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-full text-xs font-bold transition border border-slate-700"
            >
              Cancel
            </button>
          </div>
        )}

        {/* Heatmap & Hotspot Locations Mask Floating Toolbar */}
        <div className="absolute top-4 left-4 z-20 bg-slate-900/95 backdrop-blur border border-slate-800 rounded-lg p-3 shadow-xl flex flex-col space-y-2.5">
          {/* Row 1: Tumor Heatmap Toggle */}
          <div className="flex items-center space-x-3 justify-between">
            <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
              <Flame className="w-4 h-4 text-amber-400" />
              <span>Tumor Heatmap</span>
            </div>

            <div className="flex items-center space-x-2">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-8 h-4 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-sky-600"></div>
              </label>

              {showHeatmap && (
                <div className="flex items-center space-x-1.5 border-l border-slate-800 pl-2">
                  <Sliders className="w-3.5 h-3.5 text-slate-400" />
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                    className="w-14 accent-sky-500 cursor-pointer"
                  />
                  <span className="text-[10px] font-mono text-slate-400">
                    {Math.round(heatmapOpacity * 100)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Row 2: Hotspot Locations Mask Toggle */}
          <div className="flex items-center space-x-3 justify-between pt-2 border-t border-slate-800/80">
            <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
              <MapPin className="w-4 h-4 text-sky-400" />
              <span>Hotspot Locations Mask</span>
            </div>

            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={showHotspotMask}
                onChange={(e) => setShowHotspotMask(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-8 h-4 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-sky-600"></div>
            </label>
          </div>

          {/* Colormap Legend */}
          {showHeatmap && (
            <div className="pt-2 border-t border-slate-800/80 flex items-center space-x-2 text-[10px] text-slate-400">
              <span className="font-semibold text-slate-500">Scale:</span>
              <div className="flex items-center space-x-1">
                <div className="w-2.5 h-2.5 rounded-sm bg-[#440154]" />
                <span>Stroma</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2.5 h-2.5 rounded-sm bg-[#21918c]" />
                <span>Moderate</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-2.5 h-2.5 rounded-sm bg-[#fde725]" />
                <span className="text-amber-300 font-semibold">Hotspot (&gt;75%)</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Pathologist Review Sidebar Rail */}
      <div className="w-96 border-l border-slate-800 bg-slate-900 flex flex-col h-full shadow-2xl z-20">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center space-x-2">
              <Flame className="w-4 h-4 text-amber-500" />
              <span>Stage 3: Hotspot Triage</span>
            </h2>
            <p className="text-[11px] text-slate-400 mt-0.5">Path Foundation 10× Tumor Front Screening</p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => fetchTriageData()}
              className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs font-semibold flex items-center space-x-1 transition shadow-sm"
              title="Refresh Triage Data"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <button
              onClick={handleReprocessTriage}
              disabled={reprocessing}
              className="p-1.5 bg-amber-600/20 hover:bg-amber-600/40 text-amber-400 border border-amber-500/30 rounded-lg text-xs font-semibold flex items-center space-x-1 transition shadow-sm"
              title="Re-run Vertex AI Path Foundation screening and hotspot assessment"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${reprocessing ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Re-Assess</span>
            </button>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
              data?.status === "confirmed" ? "bg-emerald-950 text-emerald-400 border border-emerald-800" : "bg-amber-950 text-amber-400 border border-amber-800"
            }`}>
              {data?.status}
            </span>
          </div>
        </div>

        {/* Stats Summary Panel */}
        <div className="p-4 bg-slate-950/60 border-b border-slate-800 grid grid-cols-2 gap-3">
          <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
            <div className="text-[10px] font-semibold uppercase text-slate-400">Active Hotspots</div>
            <div className="text-lg font-bold font-mono text-sky-400">{activeHotspotsCount}</div>
          </div>
          <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
            <div className="text-[10px] font-semibold uppercase text-slate-400">Total Tumor Area</div>
            <div className="text-lg font-bold font-mono text-amber-400">{totalAreaMm2.toFixed(2)} mm²</div>
          </div>
        </div>

        {/* Hotspots List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Proposed Tumor ROIs</span>
            <button
              onClick={() => setIsAddingRoiMode(!isAddingRoiMode)}
              className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center space-x-1.5 transition ${
                isAddingRoiMode
                  ? "bg-sky-600 text-white ring-2 ring-sky-400 shadow-md shadow-sky-600/30"
                  : "bg-sky-600/20 hover:bg-sky-600/40 text-sky-400 border border-sky-600/40"
              }`}
              title="Click on the Whole Slide Image to select a custom tumor ROI"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>{isAddingRoiMode ? "Cancel Pinning" : "+ Pin ROI on Slide"}</span>
            </button>
          </div>

          {hotspotsList.length === 0 ? (
            <div className="p-4 border border-dashed border-slate-800 rounded-lg text-center text-xs text-slate-500">
              No tumor hotspots extracted.
            </div>
          ) : (
            hotspotsList.map((hs) => {
              const isSelected = selectedHotspotId === hs.id;
              return (
                <div
                  key={hs.id}
                  className={`p-3 rounded-lg border transition ${
                    isSelected
                      ? "bg-slate-900 border-sky-500 ring-1 ring-sky-500/50 shadow-lg"
                      : hs.excluded
                      ? "bg-slate-950/40 border-slate-800/60 opacity-60"
                      : "bg-slate-900/90 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-xs font-bold text-sky-400">{hs.id}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                        {hs.source}
                      </span>
                    </div>

                    <div className="flex items-center space-x-1.5">
                      {!hs.excluded && (
                        <button
                          onClick={() => {
                            setSelectedHotspotId(null);
                            setTimeout(() => setSelectedHotspotId(hs.id), 50);
                          }}
                          className={`px-2 py-0.5 rounded text-[11px] font-semibold flex items-center space-x-1 transition ${
                            isSelected
                              ? "bg-sky-600 text-white shadow-md shadow-sky-600/30"
                              : "bg-slate-800 hover:bg-sky-600/30 text-sky-400 border border-slate-700"
                          }`}
                          title="Highlight hotspot location on slide with crosshair reticle"
                        >
                          <Crosshair className="w-3 h-3" />
                          <span>Locate</span>
                        </button>
                      )}

                      {hs.excluded ? (
                        <button
                          onClick={() => handleRestoreHotspot(hs.id)}
                          className="text-xs text-emerald-400 hover:underline font-semibold"
                        >
                          Restore
                        </button>
                      ) : (
                        <button
                          onClick={() => handleDeleteHotspot(hs.id)}
                          className="p-1 hover:bg-slate-800 text-slate-500 hover:text-rose-400 rounded"
                          title="Delete Hotspot"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* 10x Microscopic Patch Preview */}
                  {!hs.excluded && (
                    <div
                      className="relative group/thumb cursor-pointer overflow-hidden rounded border border-slate-800 bg-slate-950 h-28 mb-2 flex items-center justify-center shadow-inner"
                      onClick={() => setPreviewHotspot(hs)}
                      title="Click to inspect microscopic morphology"
                    >
                      {(() => {
                        const poly = hs.polygon_um || [];
                        const cx = poly.length > 0 ? Math.round(poly.reduce((sum, p) => sum + p[0], 0) / poly.length) : 0;
                        const cy = poly.length > 0 ? Math.round(poly.reduce((sum, p) => sum + p[1], 0) / poly.length) : 0;
                        const thumbSrc = hs.thumbnail_url && !hs.thumbnail_url.includes("storage.googleapis.com")
                          ? (hs.thumbnail_url.startsWith("http") ? hs.thumbnail_url : `${API_BASE}${hs.thumbnail_url}`)
                          : `${API_BASE}/api/v1/stages/triage/${caseId}/hotspots/${hs.id}/thumbnail?mag=10x&cx=${cx}&cy=${cy}`;
                        return (
                          <img
                            src={thumbSrc}
                            alt={`10x patch ${hs.id}`}
                            className="w-full h-full object-cover group-hover/thumb:scale-105 transition-transform duration-200"
                          />
                        );
                      })()}
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex items-end justify-between p-2 opacity-90 group-hover/thumb:opacity-100 transition">
                        <span className="text-[10px] text-sky-300 font-semibold flex items-center space-x-1">
                          <ZoomIn className="w-3 h-3" />
                          <span>10× Patch View</span>
                        </span>
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-900/90 text-amber-300 border border-amber-500/30">
                          {((hs.prob_mean || 0.7) * 100).toFixed(0)}% Tumor
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-3 gap-1 text-[11px] font-mono text-slate-400 mb-2">
                    <div>Area: <span className="text-slate-200">{hs.area_mm2} mm²</span></div>
                    <div>Mean: <span className="text-slate-200">{hs.prob_mean}</span></div>
                    <div>Max: <span className="text-slate-200">{hs.prob_max}</span></div>
                  </div>

                  {!hs.excluded && (
                    <div className="flex items-center space-x-2 pt-2 border-t border-slate-800/80">
                      <input
                        type="text"
                        placeholder="Reason for exclusion..."
                        value={excludeReasonInput[hs.id] || ""}
                        onChange={(e) => setExcludeReasonInput({ ...excludeReasonInput, [hs.id]: e.target.value })}
                        className="flex-1 bg-slate-950 border border-slate-800 text-xs px-2 py-1 rounded text-slate-300 placeholder-slate-600 focus:outline-none focus:border-slate-700"
                      />
                      <button
                        onClick={() => handleExcludeHotspot(hs.id)}
                        className="px-2 py-1 bg-rose-950/60 hover:bg-rose-900 text-rose-300 border border-rose-800/60 rounded text-xs font-semibold transition"
                      >
                        Exclude
                      </button>
                    </div>
                  )}

                  {hs.excluded && hs.exclude_reason && (
                    <div className="text-[11px] text-amber-400 italic mt-1">
                      Excluded: {hs.exclude_reason}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer Confirmation Gate */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/90 space-y-3">
          <label className="flex items-start space-x-2 cursor-pointer bg-slate-900/60 p-2 rounded-lg border border-slate-800">
            <input
              type="checkbox"
              checked={noInvasiveTumor}
              onChange={(e) => setNoInvasiveTumor(e.target.checked)}
              className="mt-0.5 accent-rose-500 rounded cursor-pointer"
            />
            <span className="text-xs text-slate-300">
              No invasive tumor identified (route directly to benign report queue)
            </span>
          </label>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleReprocessTriage}
              disabled={reprocessing}
              className="px-3 py-2.5 bg-amber-600/20 hover:bg-amber-600/30 text-amber-300 border border-amber-500/40 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1.5 transition shadow-sm"
              title="Re-run assessment of hotspots"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${reprocessing ? "animate-spin" : ""}`} />
              <span>Re-Assess</span>
            </button>

            <button
              onClick={handleConfirmStage}
              disabled={submitting || (activeHotspotsCount === 0 && !noInvasiveTumor)}
              className={`flex-1 py-2.5 rounded-lg text-xs font-bold flex items-center justify-center space-x-2 shadow-lg transition ${
                activeHotspotsCount > 0 || noInvasiveTumor
                  ? "bg-sky-600 hover:bg-sky-500 text-white shadow-sky-600/20"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed"
              }`}
              title="Confirm 10 High-Power Fields and advance to Stage 4 (Mitosis Counting)"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <>
                  <span>Confirm & Move to Stage 4</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Microscopic Patch Morphology Inspector Modal */}
      {previewHotspot && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-lg w-full max-h-[92vh] overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="p-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-950/80 shrink-0">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-bold text-slate-100">
                  Microscopic Morphology — <span className="font-mono text-sky-400">{previewHotspot.id}</span>
                </h3>
              </div>
              <button
                onClick={() => setPreviewHotspot(null)}
                className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body with Scroll */}
            <div className="p-4 flex-1 overflow-y-auto flex flex-col items-center space-y-3">
              {/* Controls Bar: Magnification + Stain Normalization Toggle */}
              <div className="flex items-center space-x-2 w-full justify-between max-w-sm">
                {/* Magnification Selector Tabs */}
                <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 space-x-1 flex-1 justify-center">
                  {(["10x", "20x", "40x"] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setModalMag(m)}
                      className={`flex-1 py-1 px-2 rounded text-xs font-semibold font-mono transition ${
                        modalMag === m
                          ? "bg-sky-600 text-white shadow-sm"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                      }`}
                    >
                      {m === "10x" ? "10×" : m === "20x" ? "20×" : "40×"}
                    </button>
                  ))}
                </div>

                {/* Stain Normalization Mode Switcher */}
                <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 space-x-1">
                  <button
                    onClick={() => setStainMode("norm")}
                    className={`py-1 px-2.5 rounded text-xs font-semibold flex items-center space-x-1 transition ${
                      stainMode === "norm"
                        ? "bg-emerald-600 text-white shadow-sm"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                    }`}
                    title="Macenko Standardized Stain Normalization"
                  >
                    <span>Norm H&E</span>
                  </button>
                  <button
                    onClick={() => setStainMode("orig")}
                    className={`py-1 px-2.5 rounded text-xs font-semibold flex items-center space-x-1 transition ${
                      stainMode === "orig"
                        ? "bg-amber-600 text-white shadow-sm"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                    }`}
                    title="Original Scanner H&E Colors"
                  >
                    <span>Orig H&E</span>
                  </button>
                </div>
              </div>

              {/* High-Resolution Microscopic Patch Display */}
              <div className="relative w-64 h-64 sm:w-72 sm:h-72 rounded-lg overflow-hidden border border-slate-700 bg-slate-950 shadow-2xl shrink-0 flex items-center justify-center">
                {(() => {
                  const poly = previewHotspot.polygon_um || [];
                  const cx = poly.length > 0 ? Math.round(poly.reduce((sum, p) => sum + p[0], 0) / poly.length) : 0;
                  const cy = poly.length > 0 ? Math.round(poly.reduce((sum, p) => sum + p[1], 0) / poly.length) : 0;
                  return (
                    <img
                      key={`${previewHotspot.id}-${modalMag}-${stainMode}`}
                      src={`${API_BASE}/api/v1/stages/triage/${caseId}/hotspots/${previewHotspot.id}/thumbnail?mag=${modalMag}&stain=${stainMode}&cx=${cx}&cy=${cy}`}
                      alt={`Microscopic morphology for ${previewHotspot.id} at ${modalMag} (${stainMode})`}
                      className="w-full h-full object-cover transition-opacity duration-200"
                    />
                  );
                })()}
                <div className="absolute top-2 right-2 px-2 py-0.5 bg-slate-900/90 border border-slate-700 rounded text-[10px] font-mono text-sky-300 font-semibold shadow">
                  {modalMag.toUpperCase()} • {stainMode === "norm" ? "Norm" : "Orig"}
                </div>
              </div>

              {/* Morphologic Metrics */}
              <div className="w-full grid grid-cols-3 gap-2">
                <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800 text-center">
                  <div className="text-[9px] text-slate-400 font-semibold uppercase">Cluster Area</div>
                  <div className="text-sm font-bold font-mono text-slate-100 mt-0.5">{previewHotspot.area_mm2} mm²</div>
                </div>
                <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800 text-center">
                  <div className="text-[9px] text-slate-400 font-semibold uppercase">Mean Tumor Prob</div>
                  <div className="text-sm font-bold font-mono text-sky-400 mt-0.5">{(previewHotspot.prob_mean * 100).toFixed(0)}%</div>
                </div>
                <div className="bg-slate-950/80 p-2 rounded-lg border border-slate-800 text-center">
                  <div className="text-[9px] text-slate-400 font-semibold uppercase">Peak Tumor Prob</div>
                  <div className="text-sm font-bold font-mono text-amber-400 mt-0.5">{(previewHotspot.prob_max * 100).toFixed(0)}%</div>
                </div>
              </div>

              <div className="w-full text-[11px] text-slate-400 bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/80 flex items-start space-x-2">
                <Info className="w-3.5 h-3.5 text-sky-400 shrink-0 mt-0.5" />
                <p>
                  Screened via Vertex AI Path Foundation model. This ROI will be transferred to <strong>Stage 4 (Mitosis Counting)</strong> for high-power mitotic figure enumeration.
                </p>
              </div>
            </div>

            {/* Pinned Modal Footer */}
            <div className="p-3 border-t border-slate-800 bg-slate-950/80 flex items-center justify-end shrink-0">
              <button
                onClick={() => setPreviewHotspot(null)}
                className="px-5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold transition border border-slate-700"
              >
                Close Morphology Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
