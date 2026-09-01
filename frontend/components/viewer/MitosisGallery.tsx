"use client";

import React, { useEffect, useRef } from "react";
import { 
  CheckCircle2, 
  XCircle, 
  HelpCircle, 
  Sparkles, 
  Layers, 
  ExternalLink,
  Filter,
  Eye
} from "lucide-react";
import { MitosisCandidate, API_BASE } from "@/lib/api";

interface MitosisGalleryProps {
  caseId: string;
  candidates: MitosisCandidate[];
  selectedCandidateId: string | null;
  onSelectCandidate: (candidate: MitosisCandidate) => void;
  onToggleCandidate: (id: string, newLabel: "mitosis" | "not_mitosis" | "unreviewed") => void;
  onJumpToCandidate: (candidate: MitosisCandidate) => void;
  stainMode: "norm" | "orig";
  filterMode: "all" | "unreviewed" | "mitosis" | "not_mitosis";
  onSetFilterMode: (mode: "all" | "unreviewed" | "mitosis" | "not_mitosis") => void;
  fieldSeq?: number;
  totalFields?: number;
  onApproveFieldAndNext?: () => void;
}

export function MitosisGallery({
  caseId,
  candidates,
  selectedCandidateId,
  onSelectCandidate,
  onToggleCandidate,
  onJumpToCandidate,
  stainMode,
  filterMode,
  onSetFilterMode,
  fieldSeq = 1,
  totalFields = 10,
  onApproveFieldAndNext
}: MitosisGalleryProps) {
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Filter candidates
  const filteredCandidates = candidates.filter((c) => {
    if (filterMode === "all") return true;
    return c.label === filterMode;
  });

  const unreviewedCount = candidates.filter((c) => c.label === "unreviewed").length;
  const mitosisCount = candidates.filter((c) => c.label === "mitosis").length;
  const rejectedCount = candidates.filter((c) => c.label === "not_mitosis").length;

  // Auto-scroll selected card into view
  useEffect(() => {
    if (selectedCandidateId && cardRefs.current[selectedCandidateId]) {
      cardRefs.current[selectedCandidateId]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
      });
    }
  }, [selectedCandidateId]);

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-800 text-slate-100 select-none">
      {/* Header & Filter Tabs */}
      <div className="p-3 border-b border-slate-800 bg-slate-950/60 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span className="font-semibold text-xs tracking-wider uppercase text-slate-300">
              Field #{fieldSeq} Candidates
            </span>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
            {filteredCandidates.length} / {candidates.length}
          </span>
        </div>

        {/* Filter Pills */}
        <div className="grid grid-cols-4 gap-1 p-1 bg-slate-900/90 rounded-lg text-[11px] font-medium">
          <button
            onClick={() => onSetFilterMode("all")}
            className={`py-1 rounded px-1.5 transition-all text-center ${
              filterMode === "all"
                ? "bg-slate-700 text-white font-semibold shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            All ({candidates.length})
          </button>
          <button
            onClick={() => onSetFilterMode("unreviewed")}
            className={`py-1 rounded px-1.5 transition-all text-center ${
              filterMode === "unreviewed"
                ? "bg-amber-600/40 text-amber-300 font-semibold border border-amber-500/30"
                : "text-slate-400 hover:text-amber-300"
            }`}
          >
            Unrev ({unreviewedCount})
          </button>
          <button
            onClick={() => onSetFilterMode("mitosis")}
            className={`py-1 rounded px-1.5 transition-all text-center ${
              filterMode === "mitosis"
                ? "bg-emerald-600/40 text-emerald-300 font-semibold border border-emerald-500/30"
                : "text-slate-400 hover:text-emerald-300"
            }`}
          >
            Mitosis ({mitosisCount})
          </button>
          <button
            onClick={() => onSetFilterMode("not_mitosis")}
            className={`py-1 rounded px-1.5 transition-all text-center ${
              filterMode === "not_mitosis"
                ? "bg-slate-700/40 text-slate-300 font-semibold border border-slate-600/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Rej ({rejectedCount})
          </button>
        </div>

        {/* Keyboard Shortcut Tips */}
        <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400 bg-slate-900/50 px-2 py-1 rounded border border-slate-800/80">
          <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">j</kbd>/<kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">k</kbd> Nav</span>
          <span><kbd className="px-1 py-0.5 bg-emerald-950 text-emerald-300 border border-emerald-700/50 rounded font-mono">m</kbd> Mitosis</span>
          <span><kbd className="px-1 py-0.5 bg-rose-950 text-rose-300 border border-rose-700/50 rounded font-mono">x</kbd> Reject</span>
          <span><kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">Space</kbd> 40× Focus</span>
        </div>
      </div>

      {/* Candidates Scroll List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {filteredCandidates.length === 0 ? (
          <div className="h-48 flex flex-col items-center justify-center text-slate-500 text-xs">
            <Filter className="w-6 h-6 mb-2 stroke-[1.5] text-slate-600" />
            No candidates in this filter tab
          </div>
        ) : (
          filteredCandidates.map((cand, idx) => {
            const isSelected = cand.id === selectedCandidateId;
            const isMitosis = cand.label === "mitosis";
            const isRejected = cand.label === "not_mitosis";
            const isUnreviewed = cand.label === "unreviewed";

            const cropUrl = `${API_BASE}/api/v1/stages/mitosis/${caseId}/candidates/${cand.id}/crop?stain=${stainMode}&v=1.2`;

            return (
              <div
                key={cand.id}
                ref={(el) => { cardRefs.current[cand.id] = el; }}
                onClick={() => onSelectCandidate(cand)}
                className={`relative rounded-lg p-2 transition-all cursor-pointer border ${
                  isSelected
                    ? "bg-slate-800 border-sky-500 shadow-md ring-1 ring-sky-500/50"
                    : isMitosis
                    ? "bg-emerald-950/20 border-emerald-800/40 hover:bg-emerald-950/30"
                    : isRejected
                    ? "bg-slate-900/50 border-slate-800/50 opacity-60 hover:opacity-90"
                    : "bg-slate-800/40 border-amber-700/30 hover:bg-slate-800/70"
                }`}
              >
                <div className="flex items-center gap-3">
                  {/* 128x128 Crop Thumbnail */}
                  <div className="relative w-16 h-16 rounded overflow-hidden bg-black shrink-0 border border-slate-700/80 group">
                    <img
                      src={cropUrl}
                      alt={cand.id}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                    {/* Reticle Overlay on Hover */}
                    <div className="absolute inset-0 bg-sky-500/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Eye className="w-4 h-4 text-sky-300 drop-shadow" />
                    </div>
                  </div>

                  {/* Metadata & Confidence */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-semibold text-slate-200">
                        {cand.id}
                      </span>
                      {/* State Badge */}
                      {isMitosis && (
                        <span className="flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-700/40">
                          <CheckCircle2 className="w-3 h-3" /> Mitosis
                        </span>
                      )}
                      {isRejected && (
                        <span className="flex items-center gap-1 text-[10px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700/40">
                          <XCircle className="w-3 h-3 text-slate-500" /> Rejected
                        </span>
                      )}
                      {isUnreviewed && (
                        <span className="flex items-center gap-1 text-[10px] text-amber-400 bg-amber-950/80 px-1.5 py-0.5 rounded border border-amber-700/40">
                          <HelpCircle className="w-3 h-3" /> Review
                        </span>
                      )}
                    </div>

                    <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                      <span>Det: <strong className="text-slate-300">{((cand.det_conf || 0) * 100).toFixed(0)}%</strong></span>
                      {cand.ver_conf !== null && cand.ver_conf !== undefined && (
                        <span>Ver: <strong className="text-slate-300">{((cand.ver_conf || 0) * 100).toFixed(0)}%</strong></span>
                      )}
                      {cand.label_source && cand.label_source !== "model" && (
                        <span className="text-[9px] px-1 bg-sky-950 text-sky-300 rounded border border-sky-800/40">
                          Manual
                        </span>
                      )}
                    </div>

                    {/* Quick Action Toggle Buttons */}
                    <div className="mt-2 flex items-center gap-1.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggleCandidate(cand.id, isMitosis ? "unreviewed" : "mitosis");
                        }}
                        className={`flex-1 py-1 px-2 rounded text-[11px] font-semibold flex items-center justify-center gap-1 transition-all ${
                          isMitosis
                            ? "bg-emerald-600 text-white shadow-sm"
                            : "bg-slate-800 text-slate-300 hover:bg-emerald-950 hover:text-emerald-300 border border-slate-700/60"
                        }`}
                      >
                        <CheckCircle2 className="w-3 h-3" /> Mitosis
                      </button>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggleCandidate(cand.id, isRejected ? "unreviewed" : "not_mitosis");
                        }}
                        className={`flex-1 py-1 px-2 rounded text-[11px] font-semibold flex items-center justify-center gap-1 transition-all ${
                          isRejected
                            ? "bg-rose-900/80 text-rose-200 border border-rose-700"
                            : "bg-slate-800 text-slate-400 hover:bg-rose-950/40 hover:text-rose-300 border border-slate-700/60"
                        }`}
                      >
                        <XCircle className="w-3 h-3" /> Reject
                      </button>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onJumpToCandidate(cand);
                        }}
                        title="Focus in 40x Viewer"
                        className="p-1 rounded bg-slate-800 text-slate-400 hover:text-sky-300 hover:bg-slate-700 border border-slate-700/60 transition-colors"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Sticky Bottom Fast-Forward Action */}
      {onApproveFieldAndNext && (
        <div className="p-3 border-t border-slate-800 bg-slate-950/80 shrink-0">
          <button
            onClick={onApproveFieldAndNext}
            className="w-full py-2 px-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow-lg text-xs flex items-center justify-center gap-1.5 transition-all active:scale-[0.98]"
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>
              {fieldSeq < totalFields ? `Approve Field #${fieldSeq} & Next (${fieldSeq + 1}/${totalFields})` : "Approve Field #10 (Complete)"}
            </span>
            <kbd className="ml-1 px-1 py-0.5 bg-emerald-700/80 rounded text-[10px] font-mono">↵ Enter</kbd>
          </button>
        </div>
      )}
    </div>
  );
}
