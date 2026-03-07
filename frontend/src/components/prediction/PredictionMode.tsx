import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient } from "../../api/client";
import { PredictionResult, ModelType, LayerInfo } from "../../types";
import NetworkView2D from "../visualization/NetworkView2D";
import Network3D from "../visualization3d/Network3D";
import { Eraser, ScanEye, Box, Layers } from "lucide-react";

export default function PredictionMode() {
  const CANVAS_DISPLAY_SIZE = 280;
  const CANVAS_INTERNAL_SIZE = 560;
  const [modelType, setModelType] = useState<ModelType>("ann");
  const [view3D, setView3D] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastPixels, setLastPixels] = useState<number[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mnistPreviewRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const preprocessCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const isDrawing = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const predictTimerRef = useRef<number | null>(null);
  const isPredictingRef = useRef(false);
  const pendingPredictRef = useRef(false);
  const liveLoopRef = useRef<number | null>(null);
  const pendingResultRef = useRef<PredictionResult | null>(null);
  const lastUiUpdateRef = useRef(0);

  const downsampleInput = (pixels: number[], buckets: number) => {
    if (pixels.length === 0 || buckets <= 0) return [];
    const bucketSize = Math.floor(pixels.length / buckets) || 1;
    const out: number[] = [];
    for (let i = 0; i < buckets; i++) {
      const start = i * bucketSize;
      const end = i === buckets - 1 ? pixels.length : Math.min(pixels.length, start + bucketSize);
      if (start >= pixels.length) {
        out.push(0);
        continue;
      }
      const segment = pixels.slice(start, end);
      const mean = segment.reduce((a, b) => a + b, 0) / Math.max(segment.length, 1);
      out.push(mean);
    }
    return out;
  };

  const buildAnnLayers = (raw: any, pixels: number[]) => {
    const rawLayers = raw?.layers && typeof raw.layers === "object" ? raw.layers : {};
    const hidden1 = Array.isArray(rawLayers.hidden1) ? rawLayers.hidden1 : [];
    const hidden2 = Array.isArray(rawLayers.hidden2) ? rawLayers.hidden2 : [];
    const hidden3 = Array.isArray(rawLayers.hidden3) ? rawLayers.hidden3 : [];
    const output = Array.isArray(raw?.probabilities) ? raw.probabilities : [];
    const input = downsampleInput(pixels, 28);

    const layers: LayerInfo[] = [
      { name: "Input", type: "Input", shape: [input.length], activations: input },
      { name: "Hidden 1", type: "Dense", shape: [hidden1.length], activations: hidden1 },
      { name: "Hidden 2", type: "Dense", shape: [hidden2.length], activations: hidden2 },
      { name: "Hidden 3", type: "Dense", shape: [hidden3.length], activations: hidden3 },
      { name: "Output", type: "Softmax", shape: [output.length], activations: output },
    ];

    return layers.filter((l) => l.shape[0] > 0);
  };

  const sanitizePrediction = (raw: any, pixels: number[]): PredictionResult => {
    let layers: LayerInfo[] = Array.isArray(raw?.layers) ? raw.layers : [];
    const inferredModelType = typeof raw?.model_type === "string" ? raw.model_type : modelType;
    if (inferredModelType === "ann" && (!Array.isArray(raw?.layers) || layers.length === 0)) {
      layers = buildAnnLayers(raw, pixels);
    }
    const probabilitiesRaw = Array.isArray(raw?.probabilities) ? raw.probabilities : [];
    const probabilities =
      probabilitiesRaw.length > 0
        ? probabilitiesRaw.map((p: unknown) => (typeof p === "number" ? p : 0))
        : Array.from({ length: 10 }, () => 0);

    return {
      prediction: Number.isFinite(Number(raw?.prediction)) ? Number(raw.prediction) : 0,
      confidence: Number.isFinite(Number(raw?.confidence)) ? Number(raw.confidence) : 0,
      probabilities,
      layers,
      model_type: inferredModelType,
      explanation: raw?.explanation,
    };
  };

  const buildIdleLayers = (): LayerInfo[] => {
    const base = (count: number, v: number) => Array.from({ length: count }, () => v);
    const inputActs = lastPixels.length > 0 ? downsampleInput(lastPixels, 28) : base(28, 0.0);
    const hidden1 = base(32, 0.08);
    const hidden2 = base(24, 0.08);
    const hidden3 = base(16, 0.08);
    const out = base(10, 0.1);

    return [
      { name: "Input", type: "Input", shape: [28], activations: inputActs },
      { name: "Hidden 1", type: "Dense", shape: [32], activations: hidden1 },
      { name: "Hidden 2", type: "Dense", shape: [24], activations: hidden2 },
      { name: "Hidden 3", type: "Dense", shape: [16], activations: hidden3 },
      { name: "Output", type: "Softmax", shape: [10], activations: out },
    ];
  };

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if(ctx) {
        ctx.fillStyle = "black";
        ctx.fillRect(0,0, CANVAS_INTERNAL_SIZE, CANVAS_INTERNAL_SIZE);
        ctx.strokeStyle = "white";
        ctx.lineWidth = 36;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.imageSmoothingEnabled = true;
    }
    const pctx = mnistPreviewRef.current?.getContext("2d");
    if (pctx) {
      pctx.fillStyle = "black";
      pctx.fillRect(0, 0, 112, 112);
    }
  }, []);

  useEffect(() => {
    setResult(null);
    setErrorMessage(null);
  }, [modelType]);

  useEffect(() => {
    return () => {
      if (predictTimerRef.current) window.clearTimeout(predictTimerRef.current);
      if (liveLoopRef.current) window.clearInterval(liveLoopRef.current);
    };
  }, []);

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scale = canvasRef.current.width / rect.width;
    const x = (e.clientX - rect.left) * scale;
    const y = (e.clientY - rect.top) * scale;
    if (!lastPoint.current) {
      lastPoint.current = { x, y };
      return;
    }
    const midX = (lastPoint.current.x + x) / 2;
    const midY = (lastPoint.current.y + y) / 2;
    ctx.beginPath();
    ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
    ctx.quadraticCurveTo(lastPoint.current.x, lastPoint.current.y, midX, midY);
    ctx.stroke();
    lastPoint.current = { x, y };
  };

  const startDraw = (e: React.MouseEvent) => {
    isDrawing.current = true;
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      const rect = canvasRef.current!.getBoundingClientRect();
      const scale = canvasRef.current!.width / rect.width;
      const x = (e.clientX - rect.left) * scale;
      const y = (e.clientY - rect.top) * scale;
      lastPoint.current = { x, y };
      ctx.beginPath();
      ctx.arc(x, y, 18, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
    }
    draw(e);
    if (liveLoopRef.current) window.clearInterval(liveLoopRef.current);
    liveLoopRef.current = window.setInterval(() => {
      if (isDrawing.current) void handlePredict();
    }, 320);
  };

  const stopDraw = () => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    lastPoint.current = null;
    if (liveLoopRef.current) {
      window.clearInterval(liveLoopRef.current);
      liveLoopRef.current = null;
    }
    if (pendingResultRef.current) {
      setResult(pendingResultRef.current);
      pendingResultRef.current = null;
      lastUiUpdateRef.current = performance.now();
    }
    void handlePredict();
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvasRef.current?.width ?? CANVAS_INTERNAL_SIZE, canvasRef.current?.height ?? CANVAS_INTERNAL_SIZE);
        setResult(null);
        setErrorMessage(null);
        setLastPixels([]);
        lastPoint.current = null;
    }
    const pctx = mnistPreviewRef.current?.getContext("2d");
    if (pctx) {
      pctx.fillStyle = "black";
      pctx.fillRect(0, 0, 112, 112);
    }
  };

  const mnistPreprocess = (source: HTMLCanvasElement) => {
    const srcCtx = source.getContext("2d");
    if (!srcCtx) return null;
    const src = srcCtx.getImageData(0, 0, source.width, source.height);
    const data = src.data;
    let minX = source.width, minY = source.height, maxX = -1, maxY = -1;
    for (let y = 0; y < source.height; y++) {
      for (let x = 0; x < source.width; x++) {
        const idx = (y * source.width + x) * 4;
        const v = data[idx];
        if (v > 10) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < minX || maxY < minY) {
      return new Array(28 * 28).fill(0);
    }

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const side = Math.max(w, h);
    const pad = Math.floor(side * 0.2);
    const cropX = Math.max(0, minX - pad);
    const cropY = Math.max(0, minY - pad);
    const cropW = Math.min(source.width - cropX, side + pad * 2);
    const cropH = Math.min(source.height - cropY, side + pad * 2);

    if (!preprocessCanvasRef.current) {
      preprocessCanvasRef.current = document.createElement("canvas");
      preprocessCanvasRef.current.width = 28;
      preprocessCanvasRef.current.height = 28;
      preprocessCtxRef.current = preprocessCanvasRef.current.getContext("2d");
    }
    const work = preprocessCanvasRef.current;
    const wctx = preprocessCtxRef.current;
    if (!wctx) return null;
    wctx.fillStyle = "black";
    wctx.fillRect(0, 0, 28, 28);
    wctx.imageSmoothingEnabled = true;
    wctx.drawImage(source, cropX, cropY, cropW, cropH, 4, 4, 20, 20);

    const out = wctx.getImageData(0, 0, 28, 28);
    const pixels: number[] = [];
    for (let i = 0; i < out.data.length; i += 4) {
      pixels.push(out.data[i] / 255);
    }
    return pixels;
  };

  const renderMnistPreview = (pixels: number[]) => {
    const canvas = mnistPreviewRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const cell = canvas.width / 28;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < pixels.length; i++) {
      const v = Math.max(0, Math.min(1, pixels[i]));
      const x = i % 28;
      const y = Math.floor(i / 28);
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  };

  const handlePredict = useCallback(async () => {
    if (!canvasRef.current) return;
    if (isPredictingRef.current) {
      pendingPredictRef.current = true;
      return;
    }
    isPredictingRef.current = true;
    
    const pixels = mnistPreprocess(canvasRef.current);
    if (!pixels) return;
    setLastPixels(pixels);
    renderMnistPreview(pixels);

    try {
        if (!isDrawing.current) {
          setIsPredicting(true);
        }
        setErrorMessage(null);
        const res = await apiClient.post("/predict", { pixels, model_type: modelType });
        const next = sanitizePrediction(res.data, pixels);
        const now = performance.now();
        if (isDrawing.current) {
          if (now - lastUiUpdateRef.current > 260) {
            setResult(next);
            lastUiUpdateRef.current = now;
            pendingResultRef.current = null;
          } else {
            pendingResultRef.current = next;
          }
        } else {
          setResult(next);
          lastUiUpdateRef.current = now;
          pendingResultRef.current = null;
        }
    } catch (err) {
        setErrorMessage("Prediction failed. Ensure the selected model is trained and loaded.");
        console.error("Prediction failed. Ensure model is trained/loaded.", err);
    } finally {
        if (!isDrawing.current) {
          setIsPredicting(false);
        }
        isPredictingRef.current = false;
        if (pendingPredictRef.current) {
          pendingPredictRef.current = false;
          void handlePredict();
        }
    }
  }, [modelType]);

  const displayLayers = useMemo(() => {
    if (Array.isArray(result?.layers) && result.layers.length > 0) return result.layers;
    return buildIdleLayers();
  }, [result, lastPixels]);

  return (
    <div className="flex flex-col xl:flex-row h-full min-w-0 text-slate-100 bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900 overflow-auto xl:overflow-hidden">
        <aside className="w-full xl:w-[22rem] xl:min-w-[22rem] bg-slate-900/80 backdrop-blur border-b xl:border-b-0 xl:border-r border-slate-700 p-4 flex flex-col gap-5 overflow-y-auto">
            <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                <div className="text-xs uppercase tracking-wide font-semibold text-slate-400">Model Selection</div>
                <div className="flex gap-2 mt-2">
                    {["ann", "cnn", "rnn"].map(m => (
                        <button 
                            key={m} 
                            onClick={() => setModelType(m as ModelType)}
                            className={`flex-1 py-2 px-1 rounded-md text-xs uppercase font-bold border transition-colors ${modelType === m ? "bg-cyan-700/80 border-cyan-500 text-cyan-100" : "bg-slate-800 border-slate-600 hover:bg-slate-700"}`}
                        >
                            {m}
                        </button>
                    ))}
                </div>
            </div>

            <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3 flex flex-col items-center gap-2">
                <div className="relative w-[280px] h-[280px]">
                  <canvas 
                      ref={canvasRef}
                      width={CANVAS_INTERNAL_SIZE}
                      height={CANVAS_INTERNAL_SIZE}
                      className="absolute inset-0 border-2 border-slate-600 rounded-lg bg-black cursor-crosshair touch-none shadow-[0_0_0_1px_rgba(0,0,0,0.3)]"
                      style={{ width: `${CANVAS_DISPLAY_SIZE}px`, height: `${CANVAS_DISPLAY_SIZE}px` }}
                      onMouseDown={startDraw}
                      onMouseMove={draw}
                      onMouseUp={stopDraw}
                      onMouseLeave={stopDraw}
                  />
                  <div
                    className="absolute inset-0 pointer-events-none rounded-lg"
                    style={{
                      backgroundImage:
                        "linear-gradient(to right, rgba(148,163,184,0.18) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.18) 1px, transparent 1px)",
                      backgroundSize: "calc(100% / 28) calc(100% / 28)",
                    }}
                  />
                </div>
                <div className="w-full text-[11px] text-slate-400 text-center">
                  Drawing grid: 28 x 28 cells
                </div>
                <div className="w-full flex items-center justify-between bg-slate-800/60 border border-slate-700 rounded-md p-2">
                  <div className="text-[11px] text-slate-300">MNIST-style 28x28 input</div>
                  <canvas
                    ref={mnistPreviewRef}
                    width={112}
                    height={112}
                    className="w-20 h-20 rounded border border-slate-600 bg-black"
                    style={{ imageRendering: "pixelated" }}
                  />
                </div>
                <div className="flex w-full gap-2">
                    <button onClick={clearCanvas} className="flex-1 flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 p-2 rounded-md">
                        <Eraser size={16}/> Clear
                    </button>
                    <button onClick={handlePredict} className="flex-1 flex items-center justify-center gap-2 bg-cyan-700 hover:bg-cyan-600 p-2 rounded-md">
                        <ScanEye size={16}/> {isPredicting ? "Predicting..." : "Predict"}
                    </button>
                </div>
                {errorMessage && <div className="w-full text-xs text-rose-300 bg-rose-950/50 border border-rose-800 rounded p-2">{errorMessage}</div>}
            </div>

            <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Live Status</div>
                  <div className={`text-[11px] px-2 py-1 rounded-full border ${isPredicting ? "bg-cyan-500/20 border-cyan-400/40 text-cyan-300" : "bg-emerald-500/15 border-emerald-400/35 text-emerald-300"}`}>
                    {isPredicting ? "Streaming" : "Ready"}
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-800/70 border border-slate-700 rounded p-2">
                    <div className="text-slate-400">Model</div>
                    <div className="font-semibold text-cyan-300 uppercase">{modelType}</div>
                  </div>
                  <div className="bg-slate-800/70 border border-slate-700 rounded p-2">
                    <div className="text-slate-400">View</div>
                    <div className="font-semibold text-emerald-300">{view3D ? "3D" : "2D"}</div>
                  </div>
                </div>
            </div>

            {result && (
                <div className="bg-slate-800/70 p-4 rounded-xl border border-slate-600 animate-in fade-in slide-in-from-bottom-4">
                    <div className="text-4xl font-bold text-center text-cyan-400 mb-2">{result.prediction}</div>
                    <div className="text-xs text-center text-slate-400">Confidence: {(result.confidence * 100).toFixed(1)}%</div>
                    
                    <div className="mt-4 space-y-1">
                        {result.probabilities.map((p, i) => (
                             <div key={i} className="flex items-center gap-2 text-xs">
                                 <span className="w-4 text-right">{i}</span>
                                 <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                                     <div className="h-full bg-cyan-500" style={{ width: `${Math.max(0, Math.min(100, p * 100))}%` }}/>
                                 </div>
                             </div>
                        ))}
                    </div>
                </div>
            )}
        </aside>

        <section className="flex-1 flex flex-col p-4 gap-4 min-w-0 min-h-[540px] xl:min-h-0 overflow-hidden">
            <div className="flex justify-between items-center bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-3">
                <h2 className="text-xl font-bold flex items-center gap-2">
                    {view3D ? <Box size={24} className="text-purple-400"/> : <Layers size={24} className="text-green-400"/>}
                    {view3D ? "3D Neural Topology" : "2D Neural Flow"}
                </h2>
                <div className="text-xs text-slate-400 hidden md:block">Neurons and connections are both activation-driven</div>
                <button 
                    onClick={() => setView3D(!view3D)}
                    className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-md border border-slate-600"
                >
                    Switch to {view3D ? "2D" : "3D"}
                </button>
            </div>

            <div className="flex-1 min-h-0 border border-slate-700 rounded-xl overflow-hidden bg-slate-900 relative shadow-[0_20px_60px_-35px_rgba(34,211,238,0.45)]">
                {view3D ? (
                    <Network3D layers={displayLayers} />
                ) : (
                    <NetworkView2D layers={displayLayers} modelType={modelType} />
                )}
                {!result && (
                  <div className="absolute bottom-3 left-3 text-xs text-slate-300 bg-black/40 border border-white/10 rounded px-2 py-1">
                    Live idle mode. Draw to stream activations in real time.
                  </div>
                )}
            </div>
        </section>
    </div>
  );
}
