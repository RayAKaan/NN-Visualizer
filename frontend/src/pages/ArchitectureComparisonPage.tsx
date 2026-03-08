import React, { useCallback, useEffect, useRef, useState } from "react";
import { HeaderStrip } from "../components/header/HeaderStrip";
import { ComparisonCard } from "../components/comparison/ComparisonCard";
import { PipelineCascade } from "../components/pipeline/PipelineCascade";
import { ProbabilityLandscape } from "../components/probability/ProbabilityLandscape";
import { DisagreementHighlighter } from "../components/comparison/DisagreementHighlighter";
import { useComparisonStore } from "../store/predictionStore";

const INTERNAL = 560;
const DISPLAY = 280;

function pixelizeCanvas(canvas: HTMLCanvasElement): Uint8Array {
  const srcCtx = canvas.getContext("2d");
  if (!srcCtx) return new Uint8Array(28 * 28);
  const data = srcCtx.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const v = data[(y * canvas.width + x) * 4];
      if (v > 10) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < minX || maxY < minY) return new Uint8Array(28 * 28);

  const w = maxX - minX + 1;
  const h = maxY - minY + 1;
  const side = Math.max(w, h);
  const pad = Math.floor(side * 0.2);
  const cropX = Math.max(0, minX - pad);
  const cropY = Math.max(0, minY - pad);
  const cropW = Math.min(canvas.width - cropX, side + pad * 2);
  const cropH = Math.min(canvas.height - cropY, side + pad * 2);

  const temp = document.createElement("canvas");
  temp.width = 28;
  temp.height = 28;
  const tctx = temp.getContext("2d");
  if (!tctx) return new Uint8Array(28 * 28);
  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, 28, 28);
  tctx.imageSmoothingEnabled = true;
  tctx.drawImage(canvas, cropX, cropY, cropW, cropH, 4, 4, 20, 20);
  const out = tctx.getImageData(0, 0, 28, 28).data;
  const pixels = new Uint8Array(28 * 28);
  for (let i = 0; i < out.length; i += 4) pixels[i / 4] = out[i];
  return pixels;
}

export function ArchitectureComparisonPage() {
  const runPrediction = useComparisonStore((s) => s.runPrediction);
  const setInputPixels = useComparisonStore((s) => s.setInputPixels);
  const showTrace = useComparisonStore((s) => s.showTrace);
  const toggleTrace = useComparisonStore((s) => s.toggleTrace);
  const clearResults = useComparisonStore((s) => s.clearResults);

  const [showGrid, setShowGrid] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const debounceRef = useRef<number | null>(null);

  const initCanvas = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, INTERNAL, INTERNAL);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 34;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const syncPixelsAndPredict = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const pixels = pixelizeCanvas(cv);
    setInputPixels(pixels);
    void runPrediction();
  }, [runPrediction, setInputPixels]);

  const schedulePredict = useCallback(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(syncPixelsAndPredict, 300);
  }, [syncPixelsAndPredict]);

  const onDraw = (e: React.MouseEvent) => {
    if (!drawingRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const k = canvasRef.current.width / rect.width;
    const x = (e.clientX - rect.left) * k;
    const y = (e.clientY - rect.top) * k;
    if (!lastRef.current) {
      lastRef.current = { x, y };
      return;
    }
    const mx = (lastRef.current.x + x) / 2;
    const my = (lastRef.current.y + y) / 2;
    ctx.beginPath();
    ctx.moveTo(lastRef.current.x, lastRef.current.y);
    ctx.quadraticCurveTo(lastRef.current.x, lastRef.current.y, mx, my);
    ctx.stroke();
    lastRef.current = { x, y };
    schedulePredict();
  };

  const startDraw = (e: React.MouseEvent) => {
    drawingRef.current = true;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const k = canvasRef.current.width / rect.width;
    const x = (e.clientX - rect.left) * k;
    const y = (e.clientY - rect.top) * k;
    lastRef.current = { x, y };
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.fillStyle = "white";
    ctx.fill();
    schedulePredict();
  };

  const stopDraw = () => {
    drawingRef.current = false;
    lastRef.current = null;
    schedulePredict();
  };

  const clearCanvas = () => {
    initCanvas();
    clearResults();
    setInputPixels(new Uint8Array(28 * 28));
  };

  useEffect(() => {
    initCanvas();
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [initCanvas]);

  return (
    <div className="flex flex-col h-full bg-transparent gap-3">
      <HeaderStrip
        title="Prediction - Architecture Comparison"
        subtitle="Run ANN, CNN, and RNN on the same handwritten input in parallel."
        actions={[
          { label: "Run All", onClick: () => void syncPixelsAndPredict() },
          { label: showTrace ? "Hide Trace" : "Show Trace", onClick: () => toggleTrace(!showTrace) },
          { label: showGrid ? "Grid On" : "Grid Off", onClick: () => setShowGrid((v) => !v) },
        ]}
      />

      <div className="rounded-xl border border-cyan-400/10 bg-slate-900/60 p-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative" style={{ width: DISPLAY, height: DISPLAY }}>
            <canvas
              ref={canvasRef}
              width={INTERNAL}
              height={INTERNAL}
              className="absolute inset-0 rounded-xl border-2 border-cyan-400/35 bg-black cursor-crosshair"
              style={{ width: DISPLAY, height: DISPLAY }}
              role="img"
              aria-label="Digit drawing canvas for architecture comparison"
              onMouseDown={startDraw}
              onMouseMove={onDraw}
              onMouseUp={stopDraw}
              onMouseLeave={stopDraw}
            />
            {showGrid && (
              <div
                className="absolute inset-0 rounded-xl pointer-events-none"
                style={{
                  backgroundImage:
                    "linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px),linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px)",
                  backgroundSize: "calc(100% / 28) calc(100% / 28)",
                }}
              />
            )}
          </div>
          <div className="flex flex-col gap-2 text-xs">
            <button onClick={() => void syncPixelsAndPredict()} className="h-9 px-3 rounded border border-cyan-400/45 bg-cyan-500/15 text-cyan-200">
              Predict All
            </button>
            <button onClick={clearCanvas} className="h-9 px-3 rounded border border-rose-400/35 bg-rose-500/10 text-rose-300">
              Clear
            </button>
            <p className="text-slate-400 max-w-[260px]">Auto-runs all models 300ms after drawing stops. Use this view to compare confidence and disagreements.</p>
          </div>
        </div>
      </div>

      <div className="order-last md:order-first">
        <PipelineCascade />
      </div>

      <div className="flex-1 overflow-auto grid gap-3 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
        <ComparisonCard arch="ANN" />
        <ComparisonCard arch="CNN" />
        <ComparisonCard arch="RNN" />
      </div>

      <ProbabilityLandscape />
      <DisagreementHighlighter />
    </div>
  );
}
