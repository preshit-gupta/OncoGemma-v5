"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Upload, Plus, FileText, ArrowRight, CheckCircle2, Trash2 } from "lucide-react";
import { fetchCases, createCase, uploadSlideFile, deleteCase, clearAllCases, Case } from "@/lib/api";
import { formatISTDateTime } from "@/lib/utils";

export default function CasesPage() {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatusText, setUploadStatusText] = useState("Uploading Slide File...");

  useEffect(() => {
    loadCases();
  }, []);

  const loadCases = async () => {
    try {
      const data = await fetchCases();
      setCases(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAndUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(5);
    setUploadStatusText(`Creating case for ${file.name}...`);

    try {
      // 1. Create case
      const newCase = await createCase();
      setUploadProgress(10);
      setUploadStatusText(`Uploading ${file.name} to Cloud Storage (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`);

      // 2. Upload actual slide file bytes with live progress
      await uploadSlideFile(newCase.id, file, (pct) => {
        setUploadProgress(10 + Math.round(pct * 0.85));
        if (pct >= 100) {
          setUploadStatusText("Finalizing ingest pipeline...");
        }
      });
      
      setUploadProgress(100);
      await loadCases();
    } catch (err: any) {
      console.error(err);
      alert(`Upload Error: ${err.message || "Failed to upload slide file"}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteCase = async (e: React.MouseEvent, caseId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this case?")) return;
    try {
      await deleteCase(caseId);
      await loadCases();
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to delete case");
    }
  };

  const handleClearAll = async () => {
    if (!confirm("Are you sure you want to clear all diagnostic cases from the dashboard?")) return;
    try {
      await clearAllCases();
      await loadCases();
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to clear cases");
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-8 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Diagnostic Cases
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Upload H&E whole-slide images to initiate automated Nottingham grading pipeline.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {cases.length > 0 && (
            <button
              onClick={handleClearAll}
              className="bg-slate-100 hover:bg-rose-50 text-slate-600 hover:text-rose-600 border border-slate-200 hover:border-rose-200 font-medium px-3.5 py-2.5 rounded-lg flex items-center space-x-1.5 shadow-sm transition text-xs"
              title="Clear all cases"
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear All</span>
            </button>
          )}

          {/* Upload dropzone button */}
          <label className="cursor-pointer bg-sky-600 hover:bg-sky-700 text-white font-medium px-4 py-2.5 rounded-lg flex items-center space-x-2 shadow-sm transition text-xs">
            <Plus className="w-4 h-4" />
            <span>New Case & Upload WSI</span>
            <input
              type="file"
              accept=".svs,.ndpi,.mrxs,.tif,.tiff,.jpg,.jpeg,.png"
              className="hidden"
              onChange={handleCreateAndUpload}
              disabled={uploading}
            />
          </label>
        </div>
      </div>

      {/* Upload progress banner */}
      {uploading && (
        <div className="mb-6 p-4 bg-sky-50 border border-sky-200 rounded-lg flex flex-col space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-sky-800">
            <span>{uploadStatusText}</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-sky-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-sky-600 h-full transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Cases list */}
      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading cases...</div>
      ) : cases.length === 0 ? (
        <div className="border-2 border-dashed border-slate-200 rounded-xl p-12 text-center bg-white">
          <Upload className="w-10 h-10 text-slate-300 mx-auto mb-3" />
          <h3 className="text-base font-semibold text-slate-700">No active cases</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
            Upload your first H&E breast carcinoma WSI slide (.svs, .ndpi, .tif, .jpg, .png) to get started with OncoGemma.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {cases.map((c) => (
            <Link
              key={c.id}
              href={`/cases/${c.id}`}
              className="bg-white border border-slate-200 hover:border-sky-300 hover:shadow-md transition-all rounded-xl p-5 flex flex-col justify-between group relative"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-medium">
                    {c.id.substring(0, 8)}...
                  </span>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded font-medium flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3" />
                      <span className="capitalize">{c.status}</span>
                    </span>
                    <button
                      onClick={(e) => handleDeleteCase(e, c.id)}
                      className="p-1 hover:bg-rose-50 text-slate-400 hover:text-rose-600 rounded transition"
                      title="Delete Case"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                <div className="flex items-center space-x-2 text-slate-800 font-semibold text-sm">
                  <FileText className="w-4 h-4 text-sky-600" />
                  <span>Case #{c.id.substring(0, 8)}</span>
                </div>
                <div className="text-xs text-slate-500 mt-2 font-medium">
                  Created: {formatISTDateTime(c.created_at)}
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-end text-xs font-semibold text-sky-600 space-x-1">
                <span>Open Workspace</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
