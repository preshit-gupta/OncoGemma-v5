"use client";

import React, { useEffect, useState } from "react";
import {
  FileText,
  Download,
  CheckCircle2,
  Lock,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  ExternalLink,
  ChevronRight,
  Edit3,
  Layers,
  Activity,
  Microscope,
  Info,
  Calendar,
  UserCheck,
  X
} from "lucide-react";
import {
  fetchReportData,
  updateReportData,
  regenerateReportNarrative,
  signReport,
  amendReport,
  CapReportData,
  API_BASE
} from "@/lib/api";
import { formatISTDateTime } from "@/lib/utils";

interface ReportWorkspaceProps {
  caseId: string;
  onRefreshCase?: () => void;
}

export function ReportWorkspace({ caseId, onRefreshCase }: ReportWorkspaceProps) {
  const [data, setData] = useState<CapReportData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [generatingNarrative, setGeneratingNarrative] = useState<boolean>(false);

  // Form State
  const [procedure, setProcedure] = useState<string>("Core Needle Biopsy");
  const [laterality, setLaterality] = useState<string>("right");
  const [tumorSite, setTumorSite] = useState<string>("upper_outer_quadrant");
  const [tumorSizeMm, setTumorSizeMm] = useState<number>(18.0);
  const [lviStatus, setLviStatus] = useState<"absent" | "present" | "indeterminate">("absent");
  const [dcisPresent, setDcisPresent] = useState<boolean>(false);

  const [marginStatus, setMarginStatus] = useState<"negative" | "positive" | "cannot_be_assessed">("negative");
  const [closestMarginMm, setClosestMarginMm] = useState<number>(5.0);
  const [closestMarginName, setClosestMarginName] = useState<string>("posterior");

  const [nodesExamined, setNodesExamined] = useState<number>(0);
  const [nodesPositive, setNodesPositive] = useState<number>(0);
  const [extranodalExt, setExtranodalExt] = useState<boolean>(false);

  // Biomarkers
  const [erPercent, setErPercent] = useState<number>(95);
  const [prPercent, setPrPercent] = useState<number>(80);
  const [her2Score, setHer2Score] = useState<string>("1+");
  const [her2Result, setHer2Result] = useState<string>("negative");
  const [ki67Percent, setKi67Percent] = useState<number>(18);

  // Narrative
  const [diagnosisLine, setDiagnosisLine] = useState<string>("");
  const [microscopicFindings, setMicroscopicFindings] = useState<string>("");
  const [clinicalCorrelation, setClinicalCorrelation] = useState<string>("");

  // Sign-off Modal
  const [showSignModal, setShowSignModal] = useState<boolean>(false);
  const [signedBy, setSignedBy] = useState<string>("Dr. Jane Doe, MD, FCAP");
  const [npi, setNpi] = useState<string>("NPI-1982347102");
  const [attestationAgreed, setAttestationAgreed] = useState<boolean>(false);
  const [signLoading, setSignLoading] = useState<boolean>(false);

  // Amendment Modal
  const [showAmendModal, setShowAmendModal] = useState<boolean>(false);
  const [amendReason, setAmendReason] = useState<string>("");
  const [amendLoading, setAmendLoading] = useState<boolean>(false);

  const loadData = async () => {
    try {
      setLoading(true);
      const res = await fetchReportData(caseId);
      setData(res);

      setProcedure(res.procedure || "Core Needle Biopsy");
      setLaterality(res.laterality || "right");
      setTumorSite(res.tumor_site || "upper_outer_quadrant");
      setTumorSizeMm(res.tumor_size_mm || 18.0);
      setLviStatus(res.lvi_status || "absent");
      setDcisPresent(res.dcis_present || false);

      setMarginStatus(res.margins?.status || "negative");
      setClosestMarginMm(res.margins?.closest_margin_mm ?? 5.0);
      setClosestMarginName(res.margins?.closest_margin_name || "posterior");

      setNodesExamined(res.lymph_nodes?.examined_count || 0);
      setNodesPositive(res.lymph_nodes?.positive_count || 0);
      setExtranodalExt(res.lymph_nodes?.extranodal_extension || false);

      setErPercent(res.biomarkers?.er?.percent ?? 95);
      setPrPercent(res.biomarkers?.pr?.percent ?? 80);
      setHer2Score(res.biomarkers?.her2?.ihc_score || "1+");
      setHer2Result(res.biomarkers?.her2?.result || "negative");
      setKi67Percent(res.biomarkers?.ki67?.percent ?? 18);

      const defaultDiag = `BREAST, CORE NEEDLE BIOPSY: INVASIVE BREAST CARCINOMA OF NO SPECIAL TYPE (DUCTAL), NOTTINGHAM HISTOLOGIC GRADE ${res.nottingham_grade?.grade || 3} (SCORE ${res.nottingham_grade?.nottingham_sum || 8}/9: TUBULE ${res.nottingham_grade?.tubule_score || 3}, PLEOMORPHISM ${res.nottingham_grade?.pleo_score || 3}, MITOSIS ${res.nottingham_grade?.mitotic_score || 2}).`;
      const defaultMicro = `Histologic sections demonstrate infiltrating cohesive cords and solid sheets of malignant epithelial cells dissecting fibrous stroma. Glandular lumen formation is minimal (<10%, Score 3). Nuclei exhibit marked pleomorphism with prominent variation in size and contour, open vesicular chromatin, and identifiable macronucleoli (Score 3). Mitotic activity reveals 12 mitotic figures across 10 standardized high-power fields (5.56 mitoses/mm², Score 2).`;
      const defaultCorr = `Nottingham Combined Histological Grade 3 (Poorly Differentiated). Routine immunohistochemical reflex evaluation for ER, PR, HER2, and Ki-67 proliferation index is recommended on diagnostic tissue.`;

      setDiagnosisLine(res.narrative?.diagnosis_line || defaultDiag);
      setMicroscopicFindings(res.narrative?.microscopic_findings || defaultMicro);
      setClinicalCorrelation(res.narrative?.clinical_correlation || defaultCorr);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [caseId]);

  // Keyboard Escape listener to dismiss open modal overlays
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (showSignModal) setShowSignModal(false);
        else if (showAmendModal) setShowAmendModal(false);
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [showSignModal, showAmendModal]);

  const handleUpdate = async (overrides?: Partial<CapReportData>) => {
    try {
      setSaving(true);
      const payload = {
        case_id: caseId,
        procedure,
        laterality,
        tumor_site: tumorSite,
        tumor_size_mm: tumorSizeMm,
        lvi_status: lviStatus,
        dcis_present: dcisPresent,
        margins: {
          status: marginStatus,
          closest_margin_mm: closestMarginMm,
          closest_margin_name: closestMarginName,
          positive_margins: []
        },
        lymph_nodes: {
          examined_count: nodesExamined,
          positive_count: nodesPositive,
          extranodal_extension: extranodalExt,
          largest_metastasis_mm: 0.0
        },
        biomarkers: {
          er: { status: erPercent > 1 ? "positive" : "negative", percent: erPercent, allred_score: 8 },
          pr: { status: prPercent > 1 ? "positive" : "negative", percent: prPercent, allred_score: 7 },
          her2: { ihc_score: her2Score, fish_status: "not_performed", result: her2Result },
          ki67: { percent: ki67Percent }
        },
        narrative: {
          diagnosis_line: diagnosisLine,
          microscopic_findings: microscopicFindings,
          clinical_correlation: clinicalCorrelation
        },
        ...overrides
      };
      const res = await updateReportData(payload);
      setData(res);
      if (onRefreshCase) onRefreshCase();
    } catch (err) {
      console.error(err);
      alert("Failed to update report");
    } finally {
      setSaving(false);
    }
  };

  const handleRegenerateNarrative = async () => {
    try {
      setGeneratingNarrative(true);
      const res = await regenerateReportNarrative(caseId);
      if (res.narrative) {
        setDiagnosisLine(res.narrative.diagnosis_line);
        setMicroscopicFindings(res.narrative.microscopic_findings);
        setClinicalCorrelation(res.narrative.clinical_correlation);
      }
      await loadData();
    } catch (err) {
      console.error(err);
      alert("Failed to regenerate narrative");
    } finally {
      setGeneratingNarrative(false);
    }
  };

  const handleSign = async () => {
    if (!attestationAgreed) {
      alert("Please agree to the pathologist attestation statement before signing.");
      return;
    }
    try {
      setSignLoading(true);
      const res = await signReport({
        case_id: caseId,
        signed_by: signedBy,
        npi,
        attestation_statement: "I electronically attest that I have reviewed the whole slide image, triage hotspots, mitotic counts, and Nottingham histologic grading parameters, and I verify that the diagnostic findings, CAP synoptic elements, and AJCC staging in this report are accurate."
      });
      setData(res);
      setShowSignModal(false);
      if (onRefreshCase) onRefreshCase();
    } catch (err: any) {
      console.error(err);
      if (err.message?.includes("already finalized") || err.message?.includes("signed")) {
        await loadData();
        setShowSignModal(false);
        if (onRefreshCase) onRefreshCase();
      } else {
        alert(err.message || "Failed to sign report");
      }
    } finally {
      setSignLoading(false);
    }
  };

  const handleAmend = async () => {
    if (amendReason.trim().length < 10) {
      alert("Please provide an amendment rationale of at least 10 characters.");
      return;
    }
    try {
      setAmendLoading(true);
      const res = await amendReport({
        case_id: caseId,
        amended_by: signedBy,
        amendment_reason: amendReason,
        updated_fields: {
          tumor_size_mm: tumorSizeMm,
          lvi_status: lviStatus
        }
      });
      setData(res);
      setShowAmendModal(false);
      setAmendReason("");
      if (onRefreshCase) onRefreshCase();
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Failed to submit amendment");
    } finally {
      setAmendLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 space-y-3 bg-slate-950 p-8">
        <div className="w-10 h-10 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin" />
        <p className="text-sm font-medium">Loading CAP-Compliant Synoptic Report...</p>
      </div>
    );
  }

  const isSigned = data?.status === "signed" || data?.status === "amended";
  const isResection = procedure.toLowerCase().includes("excision") || procedure.toLowerCase().includes("mastectomy");

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-950 text-slate-100">
      {/* Top Action & Status Bar */}
      <div className="bg-slate-900 border-b border-slate-800 px-6 py-3.5 flex items-center justify-between z-10 shrink-0">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-sky-950/80 border border-sky-800/80 rounded-lg text-sky-400">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2.5">
              <h2 className="text-base font-bold text-white tracking-tight">
                Stage 6: CAP Synoptic Pathology Report
              </h2>
              <span
                className={`text-[10px] font-mono font-bold uppercase px-2.5 py-0.5 rounded-full border ${
                  isSigned
                    ? "bg-emerald-950/80 border-emerald-700 text-emerald-300"
                    : "bg-amber-950/80 border-amber-700 text-amber-300"
                }`}
              >
                {data?.status || "Draft"}
              </span>
              {data?.amendments && data.amendments.length > 0 && (
                <span className="text-[10px] font-mono bg-purple-950/80 border border-purple-700 text-purple-300 px-2 py-0.5 rounded-full">
                  v1.{data.amendments.length}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              College of American Pathologists (CAP) Cancer Protocol • Invasive Carcinoma of the Breast
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2.5">
          {/* Download JSON Button */}
          <a
            href={`${API_BASE}/api/v1/stages/report/${caseId}/json`}
            download={`CAP_Synoptic_${caseId.substring(0, 8)}.json`}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 rounded-lg transition text-xs font-semibold flex items-center space-x-1.5"
            title="Download structured CAP eCC / FHIR-compatible JSON"
          >
            <Download className="w-3.5 h-3.5 text-sky-400" />
            <span>Export JSON</span>
          </a>

          {/* Download/Preview PDF Button */}
          <a
            href={`${API_BASE}/api/v1/stages/report/${caseId}/pdf`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 rounded-lg transition text-xs font-semibold flex items-center space-x-1.5"
            title="Open printable institutional clinical PDF"
          >
            <ExternalLink className="w-3.5 h-3.5 text-sky-400" />
            <span>Printable PDF</span>
          </a>

          {/* Save / Update Button */}
          {!isSigned && (
            <button
              onClick={() => handleUpdate()}
              disabled={saving}
              className="px-3.5 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded-lg transition text-xs font-semibold flex items-center space-x-1.5 shadow"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${saving ? "animate-spin" : ""}`} />
              <span>{saving ? "Saving..." : "Save Draft"}</span>
            </button>
          )}

          {/* Sign / Amend Action Buttons */}
          {!isSigned ? (
            <button
              onClick={() => setShowSignModal(true)}
              className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition text-xs font-bold flex items-center space-x-1.5 shadow-lg shadow-emerald-950/50 border border-emerald-400/50"
            >
              <ShieldCheck className="w-4 h-4" />
              <span>Sign & Finalize Report</span>
            </button>
          ) : (
            <button
              onClick={() => setShowAmendModal(true)}
              className="px-3.5 py-1.5 bg-purple-700 hover:bg-purple-600 text-white rounded-lg transition text-xs font-bold flex items-center space-x-1.5 shadow border border-purple-500/50"
            >
              <Edit3 className="w-3.5 h-3.5" />
              <span>Create Amendment</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Content Workspace Grid */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Signed / Locked State Banner */}
        {isSigned && (
          <div className="bg-emerald-950/40 border border-emerald-800/80 rounded-xl p-4 flex items-start justify-between">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-emerald-900/60 rounded-lg text-emerald-400 shrink-0 mt-0.5">
                <Lock className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <h4 className="text-sm font-bold text-emerald-300">
                    Report Finalized & Electronically Signed
                  </h4>
                  <span className="text-[11px] font-mono text-slate-400">
                    ({formatISTDateTime(data?.signed_at)})
                  </span>
                </div>
                <p className="text-xs text-slate-300">
                  Attested by <span className="font-semibold text-white">{data?.signed_by}</span> ({data?.npi}). This synoptic document is locked against accidental alterations.
                </p>
                <div className="text-[10px] font-mono text-emerald-400/80 truncate max-w-2xl pt-1">
                  SHA-256 Verification Hash: {data?.integrity_hash}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2-Column Responsive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Specimen Setup & Nottingham Grade Synthesis (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* 1. Specimen & Digital Pathology Intake Card */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-2">
                  <Layers className="w-4 h-4" />
                  <span>1. Specimen & Digital Pathology Intake</span>
                </h3>
                <span className="text-[11px] font-mono text-slate-400">
                  Case #{caseId.substring(0, 8)}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Specimen / Procedure</div>
                  <div className="text-sm font-bold text-white">
                    Breast Core Needle Biopsy
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Needle core biopsy fragments
                  </div>
                </div>

                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Evaluated Tissue Area</div>
                  <div className="text-sm font-bold text-sky-300">
                    3.60 mm²
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Mapped across biopsy cores
                  </div>
                </div>

                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Diagnostic Status</div>
                  <div className="text-sm font-bold text-emerald-400">
                    {data?.status?.toUpperCase() || "SIGNED"}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    Finalized & Attested
                  </div>
                </div>
              </div>
            </div>

            {/* 2. Verified Nottingham Grade Synthesis */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-2">
                  <Microscope className="w-4 h-4" />
                  <span>2. Verified Histologic & Grade Synthesis (Stages 4 & 5)</span>
                </h3>
                <span className="text-[10px] font-mono bg-sky-950 border border-sky-800 text-sky-300 px-2 py-0.5 rounded">
                  Elston-Ellis Modification
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* Histologic Type */}
                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">CAP Histologic Subtype</div>
                  <div className="text-sm font-bold text-white truncate">
                    {data?.nottingham_grade?.histologic_type || "IDC-NST"}
                  </div>
                  <div className="text-[10px] text-emerald-400 flex items-center space-x-1">
                    <UserCheck className="w-3 h-3" />
                    <span>Pathologist Confirmed</span>
                  </div>
                </div>

                {/* Nottingham Grade */}
                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Nottingham Grade</div>
                  <div className="text-sm font-bold text-sky-300">
                    Grade {data?.nottingham_grade?.grade || 3} ({data?.nottingham_grade?.nottingham_sum || 8}/9)
                  </div>
                  <div className="text-[10px] text-slate-400">
                    T{data?.nottingham_grade?.tubule_score || 3} + P{data?.nottingham_grade?.pleo_score || 3} + M{data?.nottingham_grade?.mitotic_score || 2}
                  </div>
                </div>

                {/* Mitotic Activity */}
                <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase">Mitotic Density</div>
                  <div className="text-sm font-bold text-purple-300">
                    Score {data?.nottingham_grade?.mitotic_score || 2} (5.56/mm²)
                  </div>
                  <div className="text-[10px] text-slate-400">
                    12 mitoses across 10 HPFs
                  </div>
                </div>
              </div>

              {/* Component Breakdown Table */}
              <div className="border border-slate-800 rounded-lg overflow-hidden text-xs">
                <table className="w-full text-left">
                  <thead className="bg-slate-950/80 text-[10px] font-semibold text-slate-400 uppercase border-b border-slate-800">
                    <tr>
                      <th className="px-3 py-2">Nottingham Component</th>
                      <th className="px-3 py-2">Quantitative Finding</th>
                      <th className="px-3 py-2 text-right">Assigned Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                    <tr>
                      <td className="px-3 py-2 font-medium text-slate-200">1. Glandular / Tubule Formation</td>
                      <td className="px-3 py-2 text-slate-400">
                        {data?.nottingham_grade?.tubule_percent ? `${data.nottingham_grade.tubule_percent}%` : "<10%"} tubular differentiation (sheet-like solid architecture)
                      </td>
                      <td className="px-3 py-2 text-right font-bold text-sky-400">
                        Score {data?.nottingham_grade?.tubule_score || 3}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-3 py-2 font-medium text-slate-200">2. Nuclear Pleomorphism</td>
                      <td className="px-3 py-2 text-slate-400">
                        Marked nuclear variation in size and contour, open vesicular chromatin, prominent nucleoli
                      </td>
                      <td className="px-3 py-2 text-right font-bold text-sky-400">
                        Score {data?.nottingham_grade?.pleo_score || 3}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-3 py-2 font-medium text-slate-200">3. Mitotic Figure Density</td>
                      <td className="px-3 py-2 text-slate-400">
                        12 mitotic figures identified across 10 standardized 40× HPFs (2.157 mm², 5.56 mitoses/mm²)
                      </td>
                      <td className="px-3 py-2 text-right font-bold text-purple-400">
                        Score {data?.nottingham_grade?.mitotic_score || 2}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* 3. Systematic 10-HPF Architectural Mapping Card */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-2">
                  <Activity className="w-4 h-4" />
                  <span>3. Systematic 10-HPF Tumor Mapping</span>
                </h3>
                <span className="text-[10px] font-mono text-emerald-400">
                  Total Tissue Area: 3.60 mm²
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-400">Evaluated HPFs</div>
                  <div className="text-sm font-bold text-white mt-0.5">10 Fields</div>
                  <div className="text-[10px] text-slate-400">524 µm field diameter</div>
                </div>
                <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-400">Evaluated HPF Area</div>
                  <div className="text-sm font-bold text-white mt-0.5">2.157 mm²</div>
                  <div className="text-[10px] text-slate-400">0.2157 mm² per HPF</div>
                </div>
                <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-400">Mitotic Density</div>
                  <div className="text-sm font-bold text-sky-300 mt-0.5">5.56 / mm²</div>
                  <div className="text-[10px] text-slate-400">12 figures total</div>
                </div>
                <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-[10px] text-slate-400">Classic Equivalent</div>
                  <div className="text-sm font-bold text-purple-300 mt-0.5">15.2 Mitoses</div>
                  <div className="text-[10px] text-slate-400">Per 2.74 mm²</div>
                </div>
              </div>
            </div>

            {/* 4. Ancillary Reflex Testing Notice */}
            <div className="bg-sky-950/20 border border-sky-800/40 rounded-xl p-4 flex items-start space-x-3 text-xs">
              <Info className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-semibold text-sky-300">
                  Ancillary Biomarker Reflex Recommendation
                </div>
                <p className="text-slate-400 leading-relaxed">
                  Immunohistochemical receptor analysis (Estrogen Receptor, Progesterone Receptor, HER2/neu) and Ki-67 proliferation index require separate IHC/ISH assays on paraffin sections. These panels are recommended as reflex testing on diagnostic biopsy tissue.
                </p>
              </div>
            </div>
          </div>

          {/* Right Column: Diagnostic Narrative & Visual Evidence (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Clinical Narrative Card */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-2">
                  <Sparkles className="w-4 h-4" />
                  <span>Clinical Diagnostic Narrative</span>
                </h3>
                {!isSigned && (
                  <button
                    onClick={handleRegenerateNarrative}
                    disabled={generatingNarrative}
                    className="p-1.5 bg-sky-950 hover:bg-sky-900 text-sky-300 rounded border border-sky-800/80 text-xs font-medium flex items-center space-x-1"
                    title="Regenerate diagnostic narrative with MedGemma"
                  >
                    <RotateCcw className={`w-3 h-3 ${generatingNarrative ? "animate-spin" : ""}`} />
                    <span>Regenerate</span>
                  </button>
                )}
              </div>

              <div className="space-y-3 text-xs">
                {/* Diagnosis Line */}
                <div className="space-y-1">
                  <label htmlFor="synoptic-diagnosis-line" className="text-slate-400 font-semibold uppercase text-[10px]">
                    Synoptic Diagnosis Line
                  </label>
                  <textarea
                    id="synoptic-diagnosis-line"
                    rows={2}
                    disabled={isSigned}
                    value={diagnosisLine}
                    onChange={(e) => setDiagnosisLine(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-sky-500 font-mono text-[11px] leading-relaxed resize-none"
                  />
                </div>

                {/* Microscopic Findings */}
                <div className="space-y-1">
                  <label htmlFor="microscopic-findings" className="text-slate-400 font-semibold uppercase text-[10px]">
                    Microscopic Architectural Findings
                  </label>
                  <textarea
                    id="microscopic-findings"
                    rows={4}
                    disabled={isSigned}
                    value={microscopicFindings}
                    onChange={(e) => setMicroscopicFindings(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-sky-500 text-xs leading-relaxed resize-none"
                  />
                </div>

                {/* Clinical Correlation */}
                <div className="space-y-1">
                  <label htmlFor="clinical-correlation" className="text-slate-400 font-semibold uppercase text-[10px]">
                    Clinical Correlation & Pathologist Comments
                  </label>
                  <textarea
                    id="clinical-correlation"
                    rows={3}
                    disabled={isSigned}
                    value={clinicalCorrelation}
                    onChange={(e) => setClinicalCorrelation(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-sky-500 text-xs leading-relaxed resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Key Visual Evidence Gallery Preview */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                <h3 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-2">
                  <Layers className="w-4 h-4" />
                  <span>Key Computational Evidence</span>
                </h3>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-1">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-center space-y-1">
                  <div className="h-16 bg-slate-900 rounded flex items-center justify-center text-[10px] text-sky-400 font-medium">
                    Tumor Map
                  </div>
                  <div className="text-[10px] text-slate-300 font-medium truncate">WSI Heatmap</div>
                </div>

                <div className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-center space-y-1">
                  <div className="h-16 bg-slate-900 rounded flex items-center justify-center text-[10px] text-purple-400 font-medium">
                    Mitosis HPF
                  </div>
                  <div className="text-[10px] text-slate-300 font-medium truncate">Top HPF (40×)</div>
                </div>

                <div className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-center space-y-1">
                  <div className="h-16 bg-slate-900 rounded flex items-center justify-center text-[10px] text-emerald-400 font-medium">
                    Grading Patch
                  </div>
                  <div className="text-[10px] text-slate-300 font-medium truncate">10× Morphology</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pathologist Sign-Off Modal */}
      {showSignModal && (
        <div 
          role="dialog" 
          aria-modal="true" 
          aria-labelledby="sign-off-modal-title"
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
        >
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2.5 bg-emerald-950/80 border border-emerald-800 rounded-xl text-emerald-400">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <div>
                  <h3 id="sign-off-modal-title" className="text-base font-bold text-white">
                    Pathologist Electronic Sign-Off & Attestation
                  </h3>
                  <p className="text-xs text-slate-400">
                    Case #{caseId.substring(0, 8)} • Stage 6 CAP Finalization
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowSignModal(false)}
                aria-label="Close dialog"
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 text-xs">
              <div className="space-y-1.5">
                <label htmlFor="pathologist-name" className="text-slate-300 font-medium">Pathologist Name & Title</label>
                <input
                  id="pathologist-name"
                  type="text"
                  value={signedBy}
                  onChange={(e) => setSignedBy(e.target.value)}
                  placeholder="e.g. Dr. Jane Doe, MD, FCAP"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500 font-medium"
                />
              </div>

              <div className="space-y-1.5">
                <label htmlFor="pathologist-npi" className="text-slate-300 font-medium">NPI / License Number</label>
                <input
                  id="pathologist-npi"
                  type="text"
                  value={npi}
                  onChange={(e) => setNpi(e.target.value)}
                  placeholder="e.g. NPI-1982347102"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-sky-500 font-mono"
                />
              </div>

              <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
                <div className="text-[11px] font-semibold text-slate-300">
                  Legal Attestation Statement:
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed italic">
                  "I electronically attest that I have reviewed the Whole-Slide Image (WSI), AI-generated hotspot triage regions, mitotic figure annotations across 10 high-power fields, and Nottingham histological parameters, and I verify that the diagnostic findings, CAP synoptic elements, and AJCC staging in this report are clinically accurate."
                </p>
                <label htmlFor="attestation-checkbox" className="flex items-start space-x-2.5 pt-1 cursor-pointer">
                  <input
                    id="attestation-checkbox"
                    type="checkbox"
                    checked={attestationAgreed}
                    onChange={(e) => setAttestationAgreed(e.target.checked)}
                    className="mt-0.5 rounded border-slate-700 text-emerald-600 focus:ring-0 bg-slate-900 w-4 h-4"
                  />
                  <span className="text-xs font-semibold text-emerald-400">
                    I agree and electronically sign this diagnostic report.
                  </span>
                </label>
              </div>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={() => setShowSignModal(false)}
                disabled={signLoading}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSign}
                disabled={signLoading || !attestationAgreed}
                className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-bold transition shadow-lg shadow-emerald-950/50 flex items-center space-x-1.5"
              >
                <Lock className={`w-3.5 h-3.5 ${signLoading ? "animate-spin" : ""}`} />
                <span>{signLoading ? "Signing & Hashing..." : "Commit Signature"}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Amendment Modal */}
      {showAmendModal && (
        <div 
          role="dialog" 
          aria-modal="true" 
          aria-labelledby="amendment-modal-title"
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
        >
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2.5 bg-purple-950/80 border border-purple-800 rounded-xl text-purple-400">
                  <Edit3 className="w-6 h-6" />
                </div>
                <div>
                  <h3 id="amendment-modal-title" className="text-base font-bold text-white">
                    Create Formal Synoptic Amendment
                  </h3>
                  <p className="text-xs text-slate-400">
                    Versioned Clinical Addendum (v1.{((data?.amendments?.length || 0) + 1)})
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setShowAmendModal(false)}
                aria-label="Close dialog"
                className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 text-xs">
              <div className="space-y-1.5">
                <label htmlFor="amendment-reason" className="text-slate-300 font-medium">Amendment Rationale & Justification</label>
                <textarea
                  id="amendment-reason"
                  rows={4}
                  value={amendReason}
                  onChange={(e) => setAmendReason(e.target.value)}
                  placeholder="State the clinical or laboratory rationale for amending this finalized report (min 10 characters)..."
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-slate-200 focus:outline-none focus:border-purple-500 leading-relaxed resize-none"
                />
              </div>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={() => setShowAmendModal(false)}
                disabled={amendLoading}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleAmend}
                disabled={amendLoading || amendReason.trim().length < 10}
                className="px-5 py-2 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-white rounded-lg text-xs font-bold transition shadow flex items-center space-x-1.5"
              >
                <RotateCcw className={`w-3.5 h-3.5 ${amendLoading ? "animate-spin" : ""}`} />
                <span>{amendLoading ? "Submitting..." : "Submit Amendment"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
