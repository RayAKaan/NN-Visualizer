import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient } from "../../api/client";
import { LayerInfo, ModelType, PredictionResult } from "../../types";
import { Columns, Grid3X3, Mountain, Orbit, Play, Redo2, Undo2, X, ChevronDown } from "lucide-react";
import { ArchitectureComparisonPage } from "../../pages/ArchitectureComparisonPage";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";

type ProbView = "bars" | "radial" | "terrain";

interface HistoryItem {
  id: string;
  modelType: ModelType;
  result: PredictionResult;
  pixels: number[];
  thumbnail: string;
}

const DISPLAY = 280;
const INTERNAL = 560;
const R = 120;
const C = 2 * Math.PI * R;
const COLORS: Record<ModelType, string> = { ann: "#0072B2", cnn: "#00806A", rnn: "#A64D85" };

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

const hashPixels = (pixels: number[], model: ModelType) =>
  `${model}:${pixels.map((p) => Math.round(p * 9)).join("")}`;

const buildAnnLayers = (raw: any, pixels: number[]) => {
  const l = raw?.layers && typeof raw.layers === "object" ? raw.layers : {};
  const input = Array.from({ length: 28 }, (_, i) => {
    const row = pixels.slice(i * 28, i * 28 + 28);
    return row.reduce((a, b) => a + b, 0) / Math.max(1, row.length);
  });
  return [
    { name: "Input", type: "Input", shape: [28], activations: input },
    { name: "Hidden 1", type: "Dense", shape: [Array.isArray(l.hidden1) ? l.hidden1.length : 0], activations: l.hidden1 ?? [] },
    { name: "Hidden 2", type: "Dense", shape: [Array.isArray(l.hidden2) ? l.hidden2.length : 0], activations: l.hidden2 ?? [] },
    { name: "Hidden 3", type: "Dense", shape: [Array.isArray(l.hidden3) ? l.hidden3.length : 0], activations: l.hidden3 ?? [] },
    { name: "Output", type: "Softmax", shape: [Array.isArray(raw?.probabilities) ? raw.probabilities.length : 10], activations: raw?.probabilities ?? [] },
  ].filter((x) => x.shape[0] > 0) as LayerInfo[];
};

const sanitize = (raw: any, pixels: number[], modelType: ModelType): PredictionResult => {
  let layers: LayerInfo[] = Array.isArray(raw?.layers) ? raw.layers : [];
  const mt = typeof raw?.model_type === "string" ? raw.model_type : modelType;
  if (mt === "ann" && layers.length === 0) layers = buildAnnLayers(raw, pixels);
  const probs = Array.isArray(raw?.probabilities) ? raw.probabilities.map((p: unknown) => (typeof p === "number" ? p : 0)) : Array(10).fill(0);
  return {
    prediction: Number.isFinite(Number(raw?.prediction)) ? Number(raw.prediction) : 0,
    confidence: Number.isFinite(Number(raw?.confidence)) ? Number(raw.confidence) : 0,
    probabilities: probs,
    layers,
    model_type: mt,
    explanation: raw?.explanation,
  };
};

const toThumb = (pixels: number[]) => {
  const cv = document.createElement("canvas");
  cv.width = 56;
  cv.height = 56;
  const ctx = cv.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 56, 56);
  const step = 2;
  for (let i = 0; i < pixels.length; i += 1) {
    const g = Math.round(clamp01(pixels[i]) * 255);
    const x = i % 28;
    const y = Math.floor(i / 28);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(x * step, y * step, step, step);
  }
  return cv.toDataURL("image/png");
};

export default function PredictionMode() {
  const [comparisonMode, setComparisonMode] = useState(false);
  const [modelType, setModelType] = useState<ModelType>("ann");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [autoPredict, setAutoPredict] = useState(true);
  const [probView, setProbView] = useState<ProbView>("bars");
  const [traceOpen, setTraceOpen] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [undo, setUndo] = useState<ImageData[]>([]);
  const [redo, setRedo] = useState<ImageData[]>([]);
  const [available, setAvailable] = useState<string[]>([]);
  const [latency, setLatency] = useState<number | null>(null);
  const [samples, setSamples] = useState<Record<string, number[]>>({});

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prepRef = useRef<HTMLCanvasElement | null>(null);
  const prepCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const drawRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const timerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const cacheRef = useRef<Map<string, PredictionResult>>(new Map());

  const color = COLORS[modelType];
  const modelReady = available.includes(modelType);
  const conf = clamp01(result?.confidence ?? 0);
  const offset = C * (1 - conf);
  const probs = result?.probabilities ?? Array(10).fill(0);
  const topIdx = probs.reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0);
  const confColor = conf >= 0.9 ? "#15803D" : conf >= 0.7 ? color : conf >= 0.5 ? "#B45309" : "#B91C1C";

  const resetCanvas = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, INTERNAL, INTERNAL);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 34;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const capture = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return null;
    return ctx.getImageData(0, 0, INTERNAL, INTERNAL);
  }, []);

  const pushUndo = useCallback(() => {
    const snap = capture();
    if (!snap) return;
    setUndo((prev) => [...prev, snap].slice(-10));
    setRedo([]);
  }, [capture]);

  const getPixels = useCallback(() => {
    const source = canvasRef.current;
    const srcCtx = source?.getContext("2d");
    if (!source || !srcCtx) return null;
    const data = srcCtx.getImageData(0, 0, source.width, source.height).data;
    let minX = source.width, minY = source.height, maxX = -1, maxY = -1;
    for (let y = 0; y < source.height; y += 1) {
      for (let x = 0; x < source.width; x += 1) {
        const v = data[(y * source.width + x) * 4];
        if (v > 10) {
          minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
        }
      }
    }
    if (maxX < minX || maxY < minY) return Array(784).fill(0);
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const side = Math.max(w, h);
    const pad = Math.floor(side * 0.2);
    const cropX = Math.max(0, minX - pad);
    const cropY = Math.max(0, minY - pad);
    const cropW = Math.min(source.width - cropX, side + pad * 2);
    const cropH = Math.min(source.height - cropY, side + pad * 2);
    if (!prepRef.current) {
      prepRef.current = document.createElement("canvas");
      prepRef.current.width = 28;
      prepRef.current.height = 28;
      prepCtxRef.current = prepRef.current.getContext("2d");
    }
    const pctx = prepCtxRef.current;
    if (!pctx) return null;
    pctx.fillStyle = "black";
    pctx.fillRect(0, 0, 28, 28);
    pctx.imageSmoothingEnabled = true;
    pctx.drawImage(source, cropX, cropY, cropW, cropH, 4, 4, 20, 20);
    const out = pctx.getImageData(0, 0, 28, 28).data;
    const px: number[] = [];
    for (let i = 0; i < out.length; i += 4) px.push(out[i] / 255);
    return px;
  }, []);

  const doPredict = useCallback(async (pixelsArg?: number[]) => {
    const pixels = pixelsArg ?? getPixels();
    if (!pixels) return;
    const nonZero = pixels.filter((p) => p > 0.1).length / 784;
    if (nonZero < 0.05) {
      setResult(null);
      return;
    }
    if (!modelReady) {
      setError(`Model ${modelType.toUpperCase()} is not loaded.`);
      return;
    }

    const key = hashPixels(pixels, modelType);
    if (cacheRef.current.has(key)) {
      setResult(cacheRef.current.get(key)!);
      return;
    }

    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setIsPredicting(true);
    setError(null);
    const started = performance.now();
    try {
      const res = await apiClient.post("/predict", { pixels, model_type: modelType }, { signal: ctl.signal });
      const next = sanitize(res.data, pixels, modelType);
      const dt = performance.now() - started;
      setLatency((prev) => (prev == null ? dt : prev * 0.7 + dt * 0.3));
      setResult(next);
      cacheRef.current.set(key, next);
      if (cacheRef.current.size > 20) {
        const first = cacheRef.current.keys().next().value;
        if (first) cacheRef.current.delete(first);
      }
      setHistory((prev) => [{ id: String(Date.now()), modelType, result: next, pixels, thumbnail: toThumb(pixels) }, ...prev].slice(0, 20));
    } catch (e: any) {
      if (e?.name !== "CanceledError" && e?.code !== "ERR_CANCELED") setError("Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }, [getPixels, modelReady, modelType]);

  const schedule = useCallback(() => {
    if (!autoPredict) return;
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => void doPredict(), 300);
  }, [autoPredict, doPredict]);

  const onDraw = (e: React.MouseEvent) => {
    if (!drawRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const k = canvasRef.current.width / rect.width;
    const x = (e.clientX - rect.left) * k;
    const y = (e.clientY - rect.top) * k;
    if (!lastRef.current) { lastRef.current = { x, y }; return; }
    const mx = (lastRef.current.x + x) / 2;
    const my = (lastRef.current.y + y) / 2;
    ctx.beginPath();
    ctx.moveTo(lastRef.current.x, lastRef.current.y);
    ctx.quadraticCurveTo(lastRef.current.x, lastRef.current.y, mx, my);
    ctx.stroke();
    lastRef.current = { x, y };
    schedule();
  };

  const startDraw = (e: React.MouseEvent) => {
    pushUndo();
    drawRef.current = true;
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
    schedule();
  };

  const stopDraw = () => {
    drawRef.current = false;
    lastRef.current = null;
    schedule();
  };

  const clear = useCallback(() => {
    pushUndo();
    resetCanvas();
    setResult(null);
    setError(null);
  }, [pushUndo, resetCanvas]);

  const undoOne = useCallback(() => {
    if (undo.length === 0) return;
    const current = capture();
    const prev = undo[undo.length - 1];
    if (!current || !prev) return;
    setUndo((s) => s.slice(0, -1));
    setRedo((s) => [...s, current].slice(-10));
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) ctx.putImageData(prev, 0, 0);
    schedule();
  }, [capture, schedule, undo]);

  const redoOne = useCallback(() => {
    if (redo.length === 0) return;
    const current = capture();
    const next = redo[redo.length - 1];
    if (!current || !next) return;
    setRedo((s) => s.slice(0, -1));
    setUndo((s) => [...s, current].slice(-10));
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) ctx.putImageData(next, 0, 0);
    schedule();
  }, [capture, redo, schedule]);

  const loadPixels = useCallback((pixels: number[]) => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const small = document.createElement("canvas");
    small.width = 28; small.height = 28;
    const sctx = small.getContext("2d");
    if (!sctx) return;
    const image = sctx.createImageData(28, 28);
    for (let i = 0; i < pixels.length; i += 1) {
      const v = Math.round(clamp01(pixels[i]) * 255);
      const idx = i * 4;
      image.data[idx] = v; image.data[idx + 1] = v; image.data[idx + 2] = v; image.data[idx + 3] = 255;
    }
    sctx.putImageData(image, 0, 0);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.drawImage(small, 0, 0, 28, 28, 0, 0, cv.width, cv.height);
    void doPredict(pixels);
  }, [doPredict]);

  useEffect(() => {
    resetCanvas();
    void (async () => {
      try {
        const m = await apiClient.get("/models/available");
        const s = await apiClient.get("/samples");
        setAvailable(Array.isArray(m.data?.available) ? m.data.available : []);
        setSamples(s.data && typeof s.data === "object" ? s.data : {});
      } catch {
        setAvailable([]);
      }
    })();
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [resetCanvas]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      if (ev.code === "Space") { ev.preventDefault(); void doPredict(); }
      if (ev.ctrlKey && !ev.shiftKey && ev.key.toLowerCase() === "z") { ev.preventDefault(); undoOne(); }
      if (ev.ctrlKey && ev.shiftKey && ev.key.toLowerCase() === "z") { ev.preventDefault(); redoOne(); }
      if (ev.ctrlKey && ev.key.toLowerCase() === "x") { ev.preventDefault(); clear(); }
      if (ev.key === "1") setModelType("ann");
      if (ev.key === "2") setModelType("cnn");
      if (ev.key === "3") setModelType("rnn");
      if (ev.key.toLowerCase() === "b") setProbView("bars");
      if (ev.key.toLowerCase() === "r") setProbView("radial");
      if (ev.key.toLowerCase() === "t") setProbView("terrain");
      if (ev.key.toLowerCase() === "g") setShowGrid((v) => !v);
      if (ev.key === "Escape") setTraceOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clear, doPredict, redoOne, undoOne]);

  const annActs = useMemo(() => {
    const layer = result?.layers?.find((l) => l.name.includes("Hidden 1"));
    return Array.isArray(layer?.activations) ? (layer!.activations as number[]).slice(0, 96) : [];
  }, [result]);

  if (comparisonMode) {
    return (
      <div className="h-full overflow-auto">
        <div className="page-shell [--shell-max:90rem] py-4 space-y-4">
          <div className="flex justify-end">
            <NeuralButton size="sm" onClick={() => setComparisonMode(false)}>
              Back To Single Model
            </NeuralButton>
          </div>
          <ArchitectureComparisonPage />
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto text-ink">
      <div className="page-shell [--shell-max:90rem] py-4 space-y-4">
        <div className="sticky top-0 z-20 -mx-1 border-b border-barley-line bg-barley-page/85 px-1 pb-2 backdrop-blur-md">
          <PageHeader
            title="Prediction"
            subtitle="Draw a digit and watch each architecture read it."
            actions={
              <>
                {latency != null && (
                  <span className="font-mono text-xs text-ink-mute">~{latency.toFixed(0)}ms</span>
                )}
                <span
                  className={`h-2.5 w-2.5 rounded-full ${modelReady ? "bg-status-success" : "bg-arch-rnn"}`}
                  role="status"
                  aria-label={modelReady ? "Model ready" : "Model loading"}
                />
                <NeuralButton size="sm" onClick={() => setComparisonMode(true)}>
                  Compare Architectures
                </NeuralButton>
              </>
            }
          >
            <div className="flex gap-2" role="tablist" aria-label="Architecture selector">
              {(["ann", "cnn", "rnn"] as ModelType[]).map((m) => (
                <button key={m} role="tab" aria-selected={modelType === m} onClick={() => setModelType(m)}
                  className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${modelType === m ? "font-semibold" : "opacity-70 hover:opacity-100"}`}
                  style={{ color: modelType === m ? COLORS[m] : "#79716B", borderColor: modelType === m ? `${COLORS[m]}66` : "#D8CFC0", background: modelType === m ? `${COLORS[m]}14` : "transparent" }}>
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          </PageHeader>
        </div>

        <div className="rounded-2xl border border-barley-linestrong bg-barley-page p-4">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(280px,340px)_minmax(0,1fr)_minmax(280px,340px)] gap-4 items-center">
            <div className="space-y-3">
              <div className="relative mx-auto" style={{ width: DISPLAY, height: DISPLAY }}>
                <canvas ref={canvasRef} width={INTERNAL} height={INTERNAL}
                  className="absolute inset-0 rounded-2xl border-2 cursor-crosshair"
                  role="img" aria-label="Drawing canvas for digit prediction"
                  style={{ width: DISPLAY, height: DISPLAY, borderColor: `${color}66`, boxShadow: `inset 0 0 35px ${color}22, 0 0 20px ${color}22` }}
                  onMouseDown={startDraw} onMouseMove={onDraw} onMouseUp={stopDraw} onMouseLeave={stopDraw} />
                {showGrid && <div className="absolute inset-0 rounded-2xl pointer-events-none" style={{ backgroundImage: "linear-gradient(to right, rgba(28,25,23,0.05) 1px, transparent 1px),linear-gradient(to bottom, rgba(28,25,23,0.05) 1px, transparent 1px)", backgroundSize: "calc(100% / 28) calc(100% / 28)" }} />}
              </div>

              <div className="w-[280px] mx-auto space-y-2">
                <div className="flex gap-2">
                  <button onClick={undoOne} disabled={undo.length === 0} aria-label="Undo stroke" title="Undo" className="h-9 w-9 grid place-items-center rounded-md border border-barley-linestrong bg-barley-sunken hover:bg-barley-wash disabled:opacity-35 disabled:cursor-not-allowed"><Undo2 size={16} /></button>
                  <button onClick={redoOne} disabled={redo.length === 0} aria-label="Redo stroke" title="Redo" className="h-9 w-9 grid place-items-center rounded-md border border-barley-linestrong bg-barley-sunken hover:bg-barley-wash disabled:opacity-35 disabled:cursor-not-allowed"><Redo2 size={16} /></button>
                  <button onClick={clear} aria-label="Clear canvas" title="Clear" className="h-9 w-9 grid place-items-center rounded-md border border-status-danger/35 bg-status-danger/5 hover:bg-status-danger/15 text-status-danger active:scale-95"><X size={16} /></button>
                  <button onClick={() => setShowGrid((v) => !v)} aria-label="Toggle grid overlay" aria-pressed={showGrid} title="Grid" className={`h-9 w-9 grid place-items-center rounded-md border ${showGrid ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash"}`}><Grid3X3 size={16} /></button>
                  <button onClick={() => void doPredict()} className="h-9 flex-1 rounded-md border border-ember-600/40 bg-ember-600/15 hover:bg-ember-600/25 active:scale-[0.98] text-ember-700 text-xs inline-flex items-center justify-center gap-1"><Play size={14} /> Predict</button>
                </div>
                <div className="text-[12px] text-ink-mute flex justify-between">
                  <span>Undo {undo.length}/10</span>
                  <button className="underline decoration-dotted" aria-pressed={autoPredict} onClick={() => setAutoPredict((v) => !v)}>Auto {autoPredict ? "on" : "off"}</button>
                </div>
                <div className="grid grid-cols-10 gap-1">
                  {Array.from({ length: 10 }, (_, i) => (
                    <button key={i} aria-label={`Load sample digit ${i}`} className="h-7 rounded border border-barley-line bg-white text-xs font-mono hover:border-ember-600/50 hover:text-ember-700"
                      onClick={() => { const px = samples[String(i)]; if (Array.isArray(px)) loadPixels(px); }}>{i}</button>
                  ))}
                </div>
              </div>
            </div>

            <div className="hidden md:flex items-center justify-center">
              <div className="text-xs text-ink-mute">Neural signal path activates on each inference</div>
            </div>

            <div className="mx-auto relative w-[300px] h-[300px]">
              <svg className="absolute inset-0" viewBox="0 0 300 300">
                <g transform="translate(150,150)">
                  <circle r={R} fill="none" stroke="rgba(28,25,23,0.10)" strokeWidth="6" />
                  <circle r={R} fill="none" stroke={confColor} strokeWidth="6" strokeDasharray={C} strokeDashoffset={offset} strokeLinecap="round" transform="rotate(-90)"
                    style={{ transition: "stroke-dashoffset 320ms ease-out" }} />
                </g>
              </svg>
              <div className="absolute inset-0 grid place-items-center text-center">
                <div className={`text-[150px] font-bold leading-none ${isPredicting ? "animate-pulse" : ""}`} style={{ color }}>
                  {error ? "!" : result ? result.prediction : "?"}
                </div>
                <div className="text-sm font-medium" style={{ color: confColor }}>
                  {error ? error : result ? `${(conf * 100).toFixed(1)}% confident` : "Draw to predict"}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-barley-linestrong bg-white p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold text-ember-700">Probability Landscape</div>
            <div className="flex gap-1">
              <button onClick={() => setProbView("bars")} aria-label="Bar view" aria-pressed={probView === "bars"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "bars" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Columns size={14} /></button>
              <button onClick={() => setProbView("radial")} aria-label="Radial view" aria-pressed={probView === "radial"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "radial" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Orbit size={14} /></button>
              <button onClick={() => setProbView("terrain")} aria-label="Terrain view" aria-pressed={probView === "terrain"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "terrain" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Mountain size={14} /></button>
            </div>
          </div>

          {probView === "bars" && (
            <div className="h-[180px] grid grid-cols-10 gap-2 items-end">
              {probs.map((p, i) => (
                <div key={i} className="relative h-full rounded-md bg-barley-sunken border border-barley-line overflow-hidden">
                  <div className="absolute bottom-0 left-0 right-0 origin-bottom transition-transform duration-300"
                    style={{ height: "100%", transform: `scaleY(${Math.max(0.02, p)})`, background: i === topIdx ? color : `${color}99`, boxShadow: i === topIdx ? `0 0 18px ${color}66` : "none" }} />
                  <div className="absolute bottom-1 inset-x-0 text-center text-[12px] font-mono">{i}</div>
                </div>
              ))}
            </div>
          )}

          {probView === "radial" && (
            <div className="h-[190px] grid place-items-center">
              <svg width="320" height="190" viewBox="0 0 320 190">
                <g transform="translate(160,95)">
                  {probs.map((p, i) => {
                    const a = (Math.PI * 2 * i) / 10 - Math.PI / 2;
                    const len = 26 + p * 65;
                    return <line key={i} x1={0} y1={0} x2={Math.cos(a) * len} y2={Math.sin(a) * len} stroke={i === topIdx ? color : `${color}66`} strokeWidth={i === topIdx ? 5 : 3} />;
                  })}
                  <circle r={22} fill="rgba(28,25,23,0.04)" stroke={`${color}66`} />
                  <text x="0" y="1" fill={color} fontSize="18" textAnchor="middle" dominantBaseline="middle">{result?.prediction ?? "?"}</text>
                </g>
              </svg>
            </div>
          )}

          {probView === "terrain" && (
            <div className="h-[190px] flex items-end justify-center gap-2 [perspective:700px]">
              {probs.map((p, i) => (
                <div key={i} className="relative w-8">
                  <div className="absolute bottom-0 w-8 rounded-t-sm transition-all duration-300"
                    style={{ height: Math.max(4, p * 140), background: `linear-gradient(to top, ${color}55, ${color})`, boxShadow: i === topIdx ? `0 0 18px ${color}66` : "none", transform: `rotateX(20deg)` }} />
                  <div className="absolute -bottom-5 w-full text-center text-[12px] font-mono">{i}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-barley-linestrong bg-white overflow-hidden">
          <button onClick={() => setTraceOpen((v) => !v)} aria-expanded={traceOpen} className="w-full h-12 px-4 flex items-center justify-between text-sm">
            <span className="text-ember-700 inline-flex items-center gap-1">
              <ChevronDown size={14} className={`transition-transform duration-200 ${traceOpen ? "" : "-rotate-90"}`} />
              What the network saw
            </span>
            <span className="text-xs text-ink-mute">Layer-by-layer breakdown</span>
          </button>
          {traceOpen && (
            <div className="p-4 border-t border-barley-line">
              {modelType === "ann" && annActs.length > 0 && (
                <div className="grid gap-[2px]" style={{ gridTemplateColumns: "repeat(32, minmax(0, 1fr))" }}>
                  {annActs.map((a, i) => <div key={i} className="h-5 rounded-[2px]" style={{ background: `rgba(194,65,12,${0.1 + clamp01(a) * 0.9})` }} />)}
                </div>
              )}
              {modelType === "cnn" && <div className="text-xs text-ink-soft">Top filters: {(result?.explanation?.active_filters ?? []).slice(0, 3).map((f: any) => `${f.layer}/${f.filter}`).join(", ") || "n/a"}</div>}
              {modelType === "rnn" && <div className="text-xs text-ink-soft">Key timesteps: {(result?.explanation?.timestep_importance ?? []).slice(0, 8).join(", ") || "n/a"}</div>}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-barley-linestrong bg-barley-page p-3">
          <div className="text-xs text-ink-mute mb-2">History</div>
          <div className="flex gap-2 overflow-x-auto pb-1" role="list">
            {history.map((h) => (
              <button key={h.id} role="listitem" aria-label={`Restore prediction ${h.result.prediction}`} className="shrink-0 w-[62px] text-center"
                onClick={() => { setModelType(h.modelType); loadPixels(h.pixels); }}>
                <img src={h.thumbnail} alt="" className="w-14 h-14 rounded-md border border-barley-linestrong bg-ink mx-auto" />
                <div className="text-[12px] mt-1" style={{ color: COLORS[h.modelType] }}>{h.result.prediction}</div>
                <div className="text-[12px] text-ink-mute">{Math.round(h.result.confidence * 100)}%</div>
              </button>
            ))}
            {history.length === 0 && <div className="text-xs text-ink-faint py-4">No predictions yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
