"use client";

import React, { useState } from "react";
import { CheckCircle2, Clock, XCircle, Play, PanelLeftClose, PanelLeft, RotateCcw, UserCheck } from "lucide-react";
import { retryStage } from "@/lib/api";

export interface StageInfo {
  id: string;
  stage: string;
  attempt: number;
  status: string;
  error?: string;
  started_at?: string;
  completed_at?: string;
}

interface StageRailProps {
  caseId: string;
  stages: StageInfo[];
  activeStage: string;
  onSelectStage: (stageName: string) => void;
  onRefresh?: () => void;
}

const STAGE_ORDER = [
  { name: "ingest", label: "v4.0 WSI Ingest" },
  { name: "preprocess", label: "v4.1 Stain & QC Gate" },
  { name: "triage", label: "v4.2 Hotspot Triage" },
  { name: "mitosis", label: "v4.3 Mitosis Counting" },
  { name: "grading", label: "v4.4 Nottingham Grade" },
  { name: "report", label: "v4.5 CAP Report" },
];

export function StageRail({ caseId, stages, activeStage, onSelectStage, onRefresh }: StageRailProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [retryingStage, setRetryingStage] = useState<string | null>(null);

  const getStageInfo = (stageName: string) => {
    const sorted = stages.filter((s) => s.stage === stageName).sort((a, b) => b.attempt - a.attempt);
    return sorted[0];
  };

  const handleRetry = async (e: React.MouseEvent, stageName: string) => {
    e.stopPropagation();
    setRetryingStage(stageName);
    try {
      await retryStage(caseId, stageName);
      if (onRefresh) onRefresh();
    } catch (err) {
      console.error(err);
      alert("Failed to retry stage execution");
    } finally {
      setRetryingStage(null);
    }
  };

  const formatStatusLabel = (status: string) => {
    switch (status) {
      case "awaiting_review":
        return "Ready for Review";
      case "confirmed":
        return "Confirmed";
      case "done":
        return "Completed";
      case "running":
        return "Processing...";
      case "queued":
        return "In Queue";
      case "failed":
        return "Failed";
      default:
        return status;
    }
  };

  const renderStatusBadge = (stageInfo?: StageInfo) => {
    const status = stageInfo?.status || "pending";
    switch (status) {
      case "done":
      case "confirmed":
        return <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />;
      case "running":
        return <Play className="w-4 h-4 text-blue-500 animate-pulse shrink-0" />;
      case "awaiting_review":
        return <UserCheck className="w-4 h-4 text-sky-500 shrink-0" />;
      case "queued":
        return <Clock className="w-4 h-4 text-sky-500 shrink-0" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-rose-500 shrink-0" />;
      default:
        return <div className="w-2 h-2 rounded-full bg-slate-300 shrink-0" />;
    }
  };

  return (
    <aside
      className={`bg-white border-r border-slate-200 flex flex-col h-full select-none transition-all duration-300 relative ${
        collapsed ? "w-14" : "w-64"
      }`}
    >
      {/* Header */}
      <div className="p-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
        {!collapsed && (
          <div>
            <h2 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Workflow Pipeline
            </h2>
            <p className="text-[10px] text-slate-500">Nottingham Grade Diagnostics</p>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 hover:bg-slate-200 text-slate-500 rounded transition mx-auto"
          title={collapsed ? "Expand Pipeline Sidebar" : "Collapse Sidebar"}
        >
          {collapsed ? <PanelLeft className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </button>
      </div>

      {/* Stage Navigation List */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {STAGE_ORDER.map((st, idx) => {
          const stageInfo = getStageInfo(st.name);
          const status = stageInfo?.status || "pending";
          const isActive = activeStage === st.name;
          const isFailed = status === "failed";

          return (
            <div key={st.name} className="flex flex-col space-y-1">
              <button
                onClick={() => onSelectStage(st.name)}
                title={collapsed ? `${st.label} (${status})` : undefined}
                className={`w-full text-left rounded-lg border transition-all flex items-center ${
                  collapsed ? "p-2.5 justify-center" : "p-3 justify-between"
                } ${
                  isActive
                    ? "bg-sky-50 border-sky-300 text-sky-900 shadow-sm"
                    : isFailed
                    ? "bg-rose-50/50 border-rose-200 text-rose-900"
                    : "bg-white border-slate-100 hover:bg-slate-50 text-slate-700"
                }`}
              >
                <div className="flex items-center space-x-2.5 min-w-0">
                  {!collapsed && (
                    <span className="text-xs font-medium text-slate-400 w-3.5 shrink-0">
                      {idx + 1}.
                    </span>
                  )}
                  {!collapsed && (
                    <div className="truncate">
                      <div className="text-xs font-semibold truncate">{st.label}</div>
                      <div className="text-[10px] text-slate-400 capitalize truncate">
                        {formatStatusLabel(status)} {stageInfo?.attempt && stageInfo.attempt > 1 ? `(Attempt ${stageInfo.attempt})` : ""}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-1.5">
                  {!collapsed && isFailed && (
                    <button
                      onClick={(e) => handleRetry(e, st.name)}
                      disabled={retryingStage === st.name}
                      className="p-1 hover:bg-rose-100 text-rose-600 rounded transition"
                      title="Retry Stage Execution"
                    >
                      <RotateCcw className={`w-3.5 h-3.5 ${retryingStage === st.name ? "animate-spin" : ""}`} />
                    </button>
                  )}
                  {renderStatusBadge(stageInfo)}
                </div>
              </button>

              {/* Show error snippet if failed */}
              {!collapsed && isFailed && stageInfo?.error && (
                <div className="mx-1 px-2.5 py-1.5 bg-rose-50 border border-rose-200 rounded text-[10px] text-rose-700 font-mono truncate" title={stageInfo.error}>
                  Error: {stageInfo.error.split("\n").filter(Boolean).pop() || "Stage failed"}
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="p-3 border-t border-slate-100 bg-slate-50 text-[11px] text-slate-400 truncate">
          OncoGemma v4.0 • Pathologist Verified
        </div>
      )}
    </aside>
  );
}
