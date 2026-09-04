"use client";

import React, { useEffect, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";
import { ZoomIn, ZoomOut, Maximize, ChevronDown, Check, Layers, Image as ImageIcon, Info } from "lucide-react";
import { API_BASE } from "@/lib/api";

export interface ViewerHotspot {
  id: string;
  polygon_um: number[][];
  area_mm2?: number;
  prob_mean?: number;
  prob_max?: number;
  source?: string;
  excluded?: boolean;
}

export interface ViewerDetectionMarker {
  id: string;
  x_um: number;
  y_um: number;
  label: "mitosis" | "not_mitosis" | "unreviewed";
  conf?: number | null;
  in_hpf?: boolean;
}

interface OpenSeadragonViewerProps {
  caseId?: string;
  mppX?: number;
  mppY?: number;
  imageWidthPx?: number;
  imageHeightPx?: number;
  layer?: "orig" | "norm";
  overlayImageUri?: string | null;
  overlayOpacity?: number;
  showOverlay?: boolean;
  showHotspotMask?: boolean;
  hotspots?: ViewerHotspot[];
  selectedHotspotId?: string | null;
  onSelectHotspot?: (id: string) => void;
  detectionMarkers?: ViewerDetectionMarker[];
  showCandidateMarkers?: boolean;
  selectedCandidateId?: string | null;
  onSelectCandidate?: (id: string) => void;
  tileUrlTemplate?: string | null;
  focusPointUm?: [number, number] | null;
  focusMag?: number;
  isAddingRoiMode?: boolean;
  onAddRoiClick?: (x_um: number, y_um: number) => void;
  className?: string;
}

const ZOOM_PRESETS = [2.5, 5, 10, 20, 40];

export function OpenSeadragonViewer({
  caseId = "",
  mppX = 0.25,
  mppY = 0.25,
  imageWidthPx = 2048,
  imageHeightPx = 2048,
  layer = "orig",
  overlayImageUri = null,
  overlayOpacity = 0.6,
  showOverlay = true,
  showHotspotMask = true,
  hotspots = [],
  selectedHotspotId = null,
  onSelectHotspot,
  detectionMarkers = [],
  showCandidateMarkers = true,
  selectedCandidateId = null,
  onSelectCandidate,
  isAddingRoiMode = false,
  onAddRoiClick,
  tileUrlTemplate = null,
  focusPointUm = null,
  focusMag = 20.0,
  className
}: OpenSeadragonViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [scaleLengthUm, setScaleLengthUm] = useState<number>(100);
  const [scalebarWidthPx, setScalebarWidthPx] = useState<number>(120);
  const [currentMag, setCurrentMag] = useState<number>(1.0);
  const [isEditingZoom, setIsEditingZoom] = useState<boolean>(false);
  const [customZoomInput, setCustomZoomInput] = useState<string>("1.0");
  const [showDropdown, setShowDropdown] = useState<boolean>(false);
  const [activeLayer, setActiveLayer] = useState<"orig" | "norm">(layer);
  const [svgPolygons, setSvgPolygons] = useState<
    Array<{ id: string; points: string; center: { x: number; y: number }; excluded?: boolean }>
  >([]);
  const [svgMarkers, setSvgMarkers] = useState<
    Array<{ id: string; x: number; y: number; label: string; conf?: number | null; in_hpf?: boolean }>
  >([]);
  const [focusedHotspotId, setFocusedHotspotId] = useState<string | null>(null);

  useEffect(() => {
    if (layer && (layer === "orig" || layer === "norm")) {
      setActiveLayer(layer);
    }
  }, [layer]);

  const isAddingRoiModeRef = useRef(isAddingRoiMode);
  useEffect(() => {
    isAddingRoiModeRef.current = isAddingRoiMode;
  }, [isAddingRoiMode]);

  const onAddRoiClickRef = useRef(onAddRoiClick);
  useEffect(() => {
    onAddRoiClickRef.current = onAddRoiClick;
  }, [onAddRoiClick]);

  const hotspotsRef = useRef(hotspots);
  useEffect(() => {
    hotspotsRef.current = hotspots;
  }, [hotspots]);

  const detectionMarkersRef = useRef(detectionMarkers);
  useEffect(() => {
    detectionMarkersRef.current = detectionMarkers;
    updateMarkers(detectionMarkers);
  }, [detectionMarkers]);

  const overlayItemRef = useRef<any>(null);
  const currentOverlayUriRef = useRef<string | null>(null);
  const isAddingOverlayRef = useRef<boolean>(false);

  // Programmatic smooth camera fly-to when focusPointUm changes
  useEffect(() => {
    if (!focusPointUm || !viewerRef.current?.viewport) return;
    const effectiveMppX = mppX || 0.25;
    const effectiveMppY = mppY || effectiveMppX;
    const imgX = focusPointUm[0] / effectiveMppX;
    const imgY = focusPointUm[1] / effectiveMppY;
    const vpPoint = viewerRef.current.viewport.imageToViewportCoordinates(new (OpenSeadragon as any).Point(imgX, imgY));

    const targetMag = focusMag || 25.0;
    const baseMpp = 0.25;
    const imageZoom = (targetMag * effectiveMppX) / (40.0 * baseMpp);
    const vpZoom = viewerRef.current.viewport.imageToViewportZoom(imageZoom);

    viewerRef.current.viewport.panTo(vpPoint, false);
    viewerRef.current.viewport.zoomTo(vpZoom, vpPoint, false);
    viewerRef.current.viewport.applyConstraints();
  }, [focusPointUm, focusMag, mppX, mppY]);

  const isNormFallbackToOrig = activeLayer === "norm" && currentMag > 10.0;

  const updateScalebar = () => {
    const viewer = viewerRef.current;
    if (!viewer?.viewport) return;
    const zoom = viewer.viewport.getZoom(true);
    const imageZoom = viewer.viewport.viewportToImageZoom(zoom);

    const effectiveMppX = mppX || 0.25;
    const umPerPx = effectiveMppX / (imageZoom || 1.0);
    const targetUm = 120 * umPerPx;

    const niceScales = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];
    const chosenScaleUm = niceScales.reduce((prev, curr) =>
      Math.abs(curr - targetUm) < Math.abs(prev - targetUm) ? curr : prev
    );

    const actualWidthPx = Math.max(30, Math.min(220, Math.round(chosenScaleUm / umPerPx)));

    setScaleLengthUm(chosenScaleUm);
    setScalebarWidthPx(actualWidthPx);

    const calculatedMag = imageZoom * (40.0 * 0.25 / effectiveMppX);
    setCurrentMag(calculatedMag);
    if (!isEditingZoom) {
      setCustomZoomInput(calculatedMag.toFixed(1));
    }
  };

  const updateMarkers = (markers?: ViewerDetectionMarker[]) => {
    const viewer = viewerRef.current;
    if (!viewer?.viewport) return;
    const currentMarkers = markers || detectionMarkersRef.current || detectionMarkers || [];
    if (!currentMarkers.length) {
      setSvgMarkers([]);
      return;
    }

    const effectiveMppX = mppX || 0.25;
    const effectiveMppY = mppY || effectiveMppX;

    const pts = currentMarkers.map((m) => {
      const imgX = m.x_um / effectiveMppX;
      const imgY = m.y_um / effectiveMppY;
      const vpPoint = viewer.viewport.imageToViewportCoordinates(new (OpenSeadragon as any).Point(imgX, imgY));
      const pixelPoint = viewer.viewport.pixelFromPoint(vpPoint, true);
      return {
        id: m.id,
        x: pixelPoint.x,
        y: pixelPoint.y,
        label: m.label,
        conf: m.conf,
        in_hpf: m.in_hpf
      };
    });
    setSvgMarkers(pts);
  };

  const updatePolygons = (items?: ViewerHotspot[]) => {
    const viewer = viewerRef.current;
    if (!viewer?.viewport) return;
    const currentHotspots = items || hotspotsRef.current || hotspots || [];
    if (!currentHotspots.length) {
      setSvgPolygons([]);
      return;
    }

    const effectiveMppX = mppX || 0.25;
    const effectiveMppY = mppY || effectiveMppX;

    const polys = currentHotspots.map((hs) => {
      const pts: { x: number; y: number }[] = [];
      let sumX = 0;
      let sumY = 0;

      for (const pt of (hs.polygon_um || [])) {
        const imgX = pt[0] / effectiveMppX;
        const imgY = pt[1] / effectiveMppY;
        const vpPoint = viewer.viewport.imageToViewportCoordinates(new (OpenSeadragon as any).Point(imgX, imgY));
        const pixelPoint = viewer.viewport.pixelFromPoint(vpPoint, true);
        pts.push({ x: pixelPoint.x, y: pixelPoint.y });
        sumX += pixelPoint.x;
        sumY += pixelPoint.y;
      }

      const center = pts.length > 0 ? { x: sumX / pts.length, y: sumY / pts.length } : { x: 0, y: 0 };
      const points = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

      return {
        id: hs.id,
        points,
        center,
        excluded: hs.excluded
      };
    });

    setSvgPolygons(polys);
  };

  useEffect(() => {
    if (!containerRef.current) return;

    // Capture previous viewport bounds so zooming/panning is preserved across layer switches
    const prevBounds = viewerRef.current?.viewport ? viewerRef.current.viewport.getBounds() : null;

    if (viewerRef.current) {
      viewerRef.current.destroy();
      viewerRef.current = null;
    }

    const maxDim = Math.max(imageWidthPx, imageHeightPx);
    const maxLevel = Math.ceil(Math.log2(maxDim)) || 11;

    const tileSource: any = {
      width: imageWidthPx,
      height: imageHeightPx,
      tileSize: 256,
      tileOverlap: 0,
      minLevel: 0,
      maxLevel: maxLevel,
      getTileUrl: (level: number, x: number, y: number) => {
        // Beyond 10x level (level > maxLevel - 2), normalized pyramid falls back to original colors
        const effectiveLayer = (activeLayer === "norm" && level > maxLevel - 2) ? "orig" : activeLayer;
        if (tileUrlTemplate) {
          let url = tileUrlTemplate;
          if (url.includes("{layer}")) {
            url = url.replace("{layer}", effectiveLayer);
          } else {
            url = url.replace("/orig/", `/${effectiveLayer}/`).replace("/norm/", `/${effectiveLayer}/`);
          }
          return url
            .replace("{z}", level.toString())
            .replace("{x}", x.toString())
            .replace("{y}", y.toString());
        }
        return `${API_BASE}/api/v1/cases/${caseId}/tiles/${effectiveLayer}/${level}/${x}_${y}.png`;
      }
    };

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "https://openseadragon.github.io/openseadragon/images/",
      tileSources: tileSource,
      showNavigationControl: false,
      animationTime: 0.3,
      blendTime: 0.1,
      maxZoomPixelRatio: 4.0,
      visibilityRatio: 0.9,
      constrainDuringPan: true,
      homeFillsViewer: false
    });

    viewerRef.current = viewer;

    const onViewportChange = () => {
      updateScalebar();
      updatePolygons();
      updateMarkers();
    };

    viewer.addHandler("open", () => {
      onViewportChange();
      if (viewer.viewport) {
        if (prevBounds) {
          viewer.viewport.fitBounds(prevBounds, true);
        } else {
          viewer.viewport.goHome(true);
        }
        viewer.viewport.applyConstraints();
      }
    });

    viewer.addHandler("animation", onViewportChange);
    viewer.addHandler("animation-finish", onViewportChange);
    viewer.addHandler("pan", onViewportChange);
    viewer.addHandler("zoom", onViewportChange);
    viewer.addHandler("resize", onViewportChange);
    viewer.addHandler("update-viewport", onViewportChange);

    viewer.addHandler("canvas-click", (event: any) => {
      if (!isAddingRoiModeRef.current) return;
      event.preventUserAction = true;
      if (!viewer.viewport) return;
      const vpPoint = viewer.viewport.pointFromPixel(event.position);
      const imgPoint = viewer.viewport.viewportToImageCoordinates(vpPoint);
      const x_um = imgPoint.x * mppX;
      const y_um = imgPoint.y * (mppY || mppX);
      if (onAddRoiClickRef.current) {
        onAddRoiClickRef.current(x_um, y_um);
      }
    });

    return () => {
      overlayItemRef.current = null;
      currentOverlayUriRef.current = null;
      isAddingOverlayRef.current = false;
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, [caseId, activeLayer, imageWidthPx, imageHeightPx, mppX, mppY, tileUrlTemplate]);

  // Sync heatmap overlay and opacity smoothly without re-downloading or stacking duplicate images
  useEffect(() => {
    try {
      const viewer = viewerRef.current;
      if (!viewer?.world) return;
      const world = viewer.world;

      const targetOpacity = showOverlay ? overlayOpacity : 0.0;

      // When overlay URI changes, remove old overlay and load the new one
      if (overlayImageUri !== currentOverlayUriRef.current) {
        if (overlayItemRef.current) {
          try {
            world.removeItem(overlayItemRef.current);
          } catch (_) {}
          overlayItemRef.current = null;
        }
        currentOverlayUriRef.current = overlayImageUri;

        // Purge any orphan overlays, keeping only the primary slide at index 0
        try {
          while (world.getItemCount() > 1) {
            world.removeItem(world.getItemAt(1));
          }
        } catch (_) {}

        if (overlayImageUri) {
          isAddingOverlayRef.current = true;
          viewer.addSimpleImage({
            url: overlayImageUri,
            opacity: targetOpacity,
            x: 0,
            y: 0,
            width: 1.0,
            success: (event: any) => {
              try {
                overlayItemRef.current = event.item;
                isAddingOverlayRef.current = false;
                if (event.item && typeof event.item.setOpacity === "function") {
                  event.item.setOpacity(targetOpacity);
                }
                if (viewer && typeof viewer.requestRedraw === "function") {
                  viewer.requestRedraw();
                } else if (viewer && typeof viewer.forceRedraw === "function") {
                  viewer.forceRedraw();
                }
              } catch (_) {}
            },
            error: () => {
              isAddingOverlayRef.current = false;
            }
          });
        }
      } else {
        // Same URI: smoothly update opacity on all overlay items and trigger redraw
        const count = world.getItemCount();
        for (let i = 1; i < count; i++) {
          const item = world.getItemAt(i);
          if (item && typeof item.setOpacity === "function") {
            try {
              item.setOpacity(targetOpacity);
            } catch (_) {}
          }
        }
        if (overlayItemRef.current && typeof overlayItemRef.current.setOpacity === "function") {
          try {
            overlayItemRef.current.setOpacity(targetOpacity);
          } catch (_) {}
        }
        if (viewer && typeof viewer.requestRedraw === "function") {
          viewer.requestRedraw();
        } else if (viewer && typeof viewer.forceRedraw === "function") {
          viewer.forceRedraw();
        }
      }
    } catch (err) {
      console.warn("[OpenSeadragonViewer Overlay Sync Note]", err);
    }
  }, [overlayOpacity, showOverlay, overlayImageUri]);



  // Update SVG polygon coordinates when hotspots change or are added
  useEffect(() => {
    hotspotsRef.current = hotspots;
    updatePolygons(hotspots);
  }, [hotspots]);

  // Display animated glowing beacon and bounding box when a hotspot is located/selected
  // Note: Slide zoom and pan position remain completely stationary as requested
  useEffect(() => {
    if (!selectedHotspotId) return;
    setFocusedHotspotId(selectedHotspotId);
    updatePolygons();

    const timer = setTimeout(() => {
      setFocusedHotspotId(null);
    }, 6000);
    return () => clearTimeout(timer);
  }, [selectedHotspotId]);

  const handleZoomIn = () => {
    if (viewerRef.current?.viewport) {
      viewerRef.current.viewport.zoomBy(1.3);
      viewerRef.current.viewport.applyConstraints();
    }
  };

  const handleZoomOut = () => {
    if (viewerRef.current?.viewport) {
      viewerRef.current.viewport.zoomBy(1 / 1.3);
      viewerRef.current.viewport.applyConstraints();
    }
  };

  const handleResetZoom = () => {
    if (viewerRef.current?.viewport) {
      viewerRef.current.viewport.goHome();
    }
  };

  const applyPower = (power: number) => {
    if (!viewerRef.current?.viewport) return;
    const targetImageZoom = power * (mppX / (40.0 * 0.25));
    const targetViewportZoom = viewerRef.current.viewport.imageToViewportZoom(targetImageZoom);
    viewerRef.current.viewport.zoomTo(targetViewportZoom);
    viewerRef.current.viewport.applyConstraints();
    setShowDropdown(false);
  };

  const handleCustomZoomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const val = parseFloat(customZoomInput);
    if (!isNaN(val) && val > 0) {
      applyPower(val);
    }
    setIsEditingZoom(false);
  };

  return (
    <div className="relative w-full h-full bg-slate-950 flex flex-col">
      {/* Top Floating Controls Bar */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between pointer-events-none">
        {/* Layer Selector Bar */}
        <div className="pointer-events-auto bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg shadow-lg p-1 flex items-center space-x-1">
          <button
            onClick={() => setActiveLayer("orig")}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center space-x-1.5 transition ${
              activeLayer === "orig"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-white hover:bg-slate-800"
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Original Colors</span>
          </button>
          <button
            onClick={() => setActiveLayer("norm")}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center space-x-1.5 transition ${
              activeLayer === "norm"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-white hover:bg-slate-800"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Normalized 10×</span>
          </button>
        </div>

        {/* Fallback Badge when zoomed beyond 10x in normalized view */}
        {isNormFallbackToOrig && (
          <div className="pointer-events-auto bg-amber-950/80 border border-amber-800/80 text-amber-300 px-2.5 py-1 rounded-full text-xs font-medium flex items-center space-x-1 shadow-md">
            <Info className="w-3.5 h-3.5 text-amber-400" />
            <span>Original Colors (&gt;10× zoom)</span>
          </div>
        )}

        {/* Zoom & Power Controls */}
        <div className="pointer-events-auto bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg shadow-lg p-1 flex items-center space-x-1">
          <button
            onClick={handleZoomOut}
            className="p-2 hover:bg-slate-800 text-slate-300 hover:text-white rounded transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          <button
            onClick={handleZoomIn}
            className="p-2 hover:bg-slate-800 text-slate-300 hover:text-white rounded transition"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <button
            onClick={handleResetZoom}
            className="p-2 hover:bg-slate-800 text-slate-300 hover:text-white rounded transition"
            title="Reset View"
          >
            <Maximize className="w-4 h-4" />
          </button>

          <div className="h-4 w-[1px] bg-slate-800 mx-1" />

          {/* Editable Custom Zoom Input & Presets Menu */}
          <div className="relative">
            <div className="flex items-center space-x-1 bg-slate-800/80 border border-slate-700 rounded px-2 py-1">
              {isEditingZoom ? (
                <form onSubmit={handleCustomZoomSubmit} className="flex items-center">
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="100"
                    value={customZoomInput}
                    onChange={(e) => setCustomZoomInput(e.target.value)}
                    onBlur={() => setIsEditingZoom(false)}
                    autoFocus
                    className="w-12 bg-slate-900 text-white text-xs font-mono px-1 py-0.5 rounded outline-none border border-sky-500"
                  />
                  <span className="text-xs font-mono text-slate-400 ml-0.5">x</span>
                </form>
              ) : (
                <button
                  onClick={() => setIsEditingZoom(true)}
                  className="text-xs font-mono font-semibold text-sky-400 hover:text-sky-300 transition"
                  title="Click to enter custom zoom magnification"
                >
                  {currentMag.toFixed(1)}x
                </button>
              )}

              <button
                onClick={() => setShowDropdown(!showDropdown)}
                className="p-0.5 text-slate-400 hover:text-white transition"
              >
                <ChevronDown className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Presets Dropdown */}
            {showDropdown && (
              <div className="absolute right-0 mt-2 w-32 bg-slate-900 border border-slate-800 rounded-lg shadow-xl py-1 z-20">
                <div className="px-3 py-1 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                  Presets
                </div>
                {ZOOM_PRESETS.map((power) => (
                  <button
                    key={power}
                    onClick={() => applyPower(power)}
                    className="w-full px-3 py-1.5 text-left text-xs text-slate-300 hover:bg-sky-600 hover:text-white flex items-center justify-between transition font-mono"
                  >
                    <span>{power}x</span>
                    {Math.abs(currentMag - power) < 0.2 && (
                      <Check className="w-3 h-3 text-sky-400" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main OSD Outer Wrapper */}
      <div 
        className={`flex-1 w-full h-full relative overflow-hidden ${isAddingRoiMode ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"}`}
      >
        {/* Dedicated OpenSeadragon Mount Target */}
        <div ref={containerRef} className="absolute inset-0 w-full h-full z-0" />

        {/* SVG Hotspot Polygons & Location Beacon Overlay */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none z-20"
          style={{ overflow: "visible" }}
        >
          {svgPolygons.map((poly) => {
            if (poly.excluded) return null;
            const isFocused = focusedHotspotId === poly.id;
            const isSelected = selectedHotspotId === poly.id;

            // Only render if mask is turned on OR this specific hotspot was located/focused
            if (!showHotspotMask && !isFocused && !isSelected) return null;

            return (
              <g key={poly.id} className="cursor-pointer pointer-events-auto transition-opacity">
                {/* Hotspot Boundary Box */}
                <polygon
                  points={poly.points}
                  fill={isFocused ? "rgba(14, 165, 233, 0.45)" : isSelected ? "rgba(14, 165, 233, 0.35)" : "rgba(245, 158, 11, 0.22)"}
                  stroke={isFocused ? "#38bdf8" : isSelected ? "#38bdf8" : "#f59e0b"}
                  strokeWidth={isFocused ? "4" : isSelected ? "3.5" : "2"}
                  strokeDasharray={poly.id.startsWith("user") ? "6,3" : undefined}
                  className={`transition-all ${isFocused ? "filter drop-shadow-[0_0_8px_rgba(56,189,248,0.8)]" : "hover:fill-amber-500/40"}`}
                  onClick={() => onSelectHotspot && onSelectHotspot(poly.id)}
                />

                {/* Floating Numbered Pin / Badge */}
                <g transform={`translate(${poly.center.x}, ${poly.center.y})`}>
                  <rect
                    x={isFocused ? "-42" : "-32"}
                    y={isFocused ? "-14" : "-12"}
                    width={isFocused ? "84" : "64"}
                    height={isFocused ? "28" : "24"}
                    rx="6"
                    fill={isFocused ? "rgba(14, 165, 233, 0.95)" : "rgba(15, 23, 42, 0.90)"}
                    stroke={isFocused ? "#ffffff" : isSelected ? "#38bdf8" : "#f59e0b"}
                    strokeWidth={isFocused ? "2" : "1.5"}
                    className={isFocused ? "shadow-[0_0_12px_rgba(56,189,248,0.9)]" : "shadow-lg"}
                  />
                  <text
                    x="0"
                    y={isFocused ? "4.5" : "4"}
                    fill="#ffffff"
                    fontSize={isFocused ? "11.5" : "11"}
                    fontWeight="700"
                    textAnchor="middle"
                    className="select-none pointer-events-none font-mono"
                  >
                    {isFocused ? `🎯 ${poly.id}` : poly.id}
                  </text>
                </g>
              </g>
            );
          })}

          {/* Candidate Detection Markers (Zoom-Adaptive & HPF Scoped) */}
          {showCandidateMarkers && svgMarkers.map((m) => {
            const isSelected = selectedCandidateId === m.id || selectedHotspotId === `cand_${m.id}`;
            // If zoomed out (< 15x), only show if inside an active HPF or explicitly selected
            if (currentMag < 15.0 && !m.in_hpf && !isSelected) return null;

            const color = m.label === "mitosis" ? "#10b981" : (m.label === "not_mitosis" ? "#94a3b8" : "#f59e0b");
            const r = isSelected ? 8 : (currentMag >= 25.0 ? 5.5 : 4.0);

            return (
              <g
                key={`marker-${m.id}`}
                className="cursor-pointer pointer-events-auto"
                onClick={() => onSelectCandidate && onSelectCandidate(m.id)}
              >
                <circle
                  cx={m.x}
                  cy={m.y}
                  r={r}
                  fill={color}
                  stroke={isSelected ? "#38bdf8" : "#0f172a"}
                  strokeWidth={isSelected ? 3 : 1.5}
                  className={isSelected ? "filter drop-shadow-[0_0_8px_rgba(56,189,248,0.9)]" : "hover:stroke-sky-300 hover:stroke-[2.5] transition-colors"}
                />
                {isSelected && (
                  <circle
                    cx={m.x}
                    cy={m.y}
                    r={r + 6}
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2"
                    strokeDasharray="3 3"
                  />
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Continuous Calibrated Dynamic Scalebar */}
      <div className="absolute bottom-4 left-4 z-10 pointer-events-none">
        <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-2 shadow-lg flex flex-col items-center">
          <div
            className="h-1.5 bg-sky-400 rounded-full mb-1 transition-all duration-150 shadow-sm"
            style={{ width: `${scalebarWidthPx}px` }}
          />
          <div className="text-[10px] font-mono font-semibold text-slate-300 tracking-wider">
            {scaleLengthUm >= 1000 ? `${scaleLengthUm / 1000} mm` : `${scaleLengthUm} µm`}
          </div>
        </div>
      </div>
    </div>
  );
}
