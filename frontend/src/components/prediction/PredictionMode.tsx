import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient } from "../../api/client";
import { ModelType, CatalogModel } from "../../types";
import {
  Columns,
  Grid3X3,
  Mountain,
  Orbit,
  Play,
  Redo2,
  Undo2,
  X,
  ChevronDown,
  ImagePlus,
  FileImage,
  Trash2 as ClearImage,
  Cpu,
  Zap,
  Database,
  Braces,
  RefreshCw,
  FileText,
} from "lucide-react";
import { ArchitectureComparisonPage } from "../../pages/ArchitectureComparisonPage";
import { PageHeader } from "@/design-system/components/PageHeader";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { WORKSPACE_EVENTS } from "../../utils/workspaceEvents";

type ProbView = "bars" | "radial" | "terrain";

interface PredResult {
  prediction: string; // display label of predicted class
  predictedIndex: number;
  confidence: number;
  probabilities: number[];
  labels: string[];
  topK: { index: number; label: string; probability: number }[];
  modelId: string;
  modelName: string;
  latencyMs?: number;
  device?: string;
  layers?: Record<string, number[]>; // legacy ANN trace payloads
  explanation?: any;
  architecture?: string | null;
  parameterCount?: number | null;
  dataset?: string | null;
  preprocessing?: string | null;
}

interface HistoryItem {
  id: string;
  family: ModelType;
  modelId: string;
  modelName: string;
  prediction: string;
  confidence: number;
  thumbnail?: string;
  pixels?: number[];
  image?: string;
  text?: string;
}

const DISPLAY = 280;
const INTERNAL = 560;
const R = 120;
const C = 2 * Math.PI * R;
const FAMILY_COLOR: Record<ModelType, string> = {
  ann: "#0072B2",
  cnn: "#00806A",
  rnn: "#A64D85",
};

const DEFAULT_MODEL: Record<ModelType, string> = {
  ann: "ann-nn-visualizer",
  cnn: "cnn-mobilenetv3small",
  rnn: "rnn-bilstm-imdb-kerasio",
};

const SAMPLE_IMAGES = ["grace_hopper.jpg", "labrador.jpg", "cat.jpg"];
const RNN_PRESETS = [
  "A wonderful, clever film with superb acting and a touching story.",
  "This movie was an absolute disaster, boring and badly acted.",
];

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function sanitize(raw: any): PredResult | null {
  if (!raw || typeof raw !== "object") return null;
  const labels = Array.isArray(raw.labels) ? raw.labels.map(String) : [];
  const probs = Array.isArray(raw.probabilities)
    ? raw.probabilities.map((p: unknown) => (typeof p === "number" ? p : 0))
    : [];
  const topK = Array.isArray(raw.top_k) ? raw.top_k : [];
  const predictedIndex = Number.isFinite(Number(raw.predicted_index)) ? Number(raw.predicted_index) : -1;
  const details = raw.details && typeof raw.details === "object" ? raw.details : {};
  const layers =
    details.layers && typeof details.layers === "object"
      ? (details.layers as Record<string, number[]>)
      : undefined;
  return {
    prediction: String(raw.predicted_class ?? (labels[predictedIndex] ?? predictedIndex)),
    predictedIndex,
    confidence: Number.isFinite(Number(raw.confidence)) ? Number(raw.confidence) : 0,
    probabilities: probs,
    labels,
    topK: topK.map((t: any) => ({
      index: Number(t.index),
      label: String(t.label ?? t.index),
      probability: Number(t.probability ?? 0),
    })),
    modelId: String(raw.model_id ?? ""),
    modelName: String(raw.model_name ?? ""),
    latencyMs: Number.isFinite(Number(raw.latency_ms)) ? Number(raw.latency_ms) : undefined,
    device: raw.device ? String(raw.device) : undefined,
    layers,
    explanation: details.explanation ?? raw.explanation,
    architecture: raw.architecture ? String(raw.architecture) : null,
    parameterCount: raw.parameter_count != null ? Number(raw.parameter_count) : null,
    dataset: raw.dataset ? String(raw.dataset) : null,
    preprocessing: raw.preprocessing ? String(raw.preprocessing) : null,
  };
}

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
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [selected, setSelected] = useState<Record<ModelType, string>>({ ...DEFAULT_MODEL });
  const [result, setResult] = useState<PredResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [autoPredict, setAutoPredict] = useState(true);
  const [probView, setProbView] = useState<ProbView>("bars");
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [traceOpen, setTraceOpen] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [undo, setUndo] = useState<ImageData[]>([]);
  const [redo, setRedo] = useState<ImageData[]>([]);
  const [samples, setSamples] = useState<Record<string, number[]>>({});
  const [cnnImage, setCnnImage] = useState<string | null>(null); // data URL (preview + b64)
  const [rnnText, setRnnText] = useState<string>("");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const prepRef = useRef<HTMLCanvasElement | null>(null);
  const prepCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const drawRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const timerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const cnnRef = useRef(cnnImage);
  cnnRef.current = cnnImage;
  const rnnRef = useRef(rnnText);
  rnnRef.current = rnnText;

  const color = FAMILY_COLOR[modelType];
  const modelRow = catalog.find((m) => m.id === selected[modelType]);
  const conf = clamp01(result?.confidence ?? 0);
  const offset = C * (1 - conf);
  const isReady = modelRow != null && (modelRow.status === "available" || modelRow.status === "loaded");

  // Bars: natural order for small label sets (digits, positive/negative),
  // otherwise the top-5 classes.
  const bars = useMemo(() => {
    if (!result) return [];
    const n = result.labels.length;
    if (n > 0 && n <= 12) {
      return result.labels.map((label, i) => ({
        label: String(label),
        prob: result.probabilities[i] ?? 0,
        index: i,
      }));
    }
    return result.topK.slice(0, 5).map((t) => ({ label: t.label, prob: t.probability, index: t.index }));
  }, [result]);
  const topIdx = bars.length > 0 ? bars.reduce((best, b, i, arr) => (b.prob > arr[best].prob ? i : best), 0) : -1;
  const confColor = conf >= 0.9 ? "#15803D" : conf >= 0.7 ? color : conf >= 0.5 ? "#B45309" : "#B91C1C";

  // ------------------------------------------------------------------
  // ANN drawing primitives
  // ------------------------------------------------------------------
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

  // ------------------------------------------------------------------
  // Core prediction
  // ------------------------------------------------------------------
  const predictCurrent = useCallback(
    async (pixelsArg?: number[]) => {
      const modelId = selectedRef.current[modelType];
      const row = catalog.find((m) => m.id === modelId);
      if (!row) {
        setError("This model is not in the registry catalog.");
        return;
      }
      if (row.status === "unavailable") {
        setError(row.unavailable_reason ?? "This model is unavailable.");
        return;
      }
      if (row.status === "error") {
        setError(row.error ?? "This model failed to load.");
        return;
      }

      let payload: Record<string, unknown> = { model_id: modelId };
      if (modelType === "ann") {
        const pixels = pixelsArg ?? getPixels();
        if (!pixels) return;
        const nonZero = pixels.filter((p) => p > 0.1).length / 784;
        if (nonZero < 0.05) {
          setResult(null);
          return;
        }
        payload.pixels = pixels;
      } else if (modelType === "cnn") {
        const img = cnnRef.current;
        if (!img) {
          setError("Upload an image first (or pick an example below).");
          return;
        }
        payload.image = img; // data URL — the API strips the prefix
      } else {
        const text = rnnRef.current.trim();
        if (!text) {
          setError("Type some text first (or pick an example below).");
          return;
        }
        payload.text = text;
      }

      abortRef.current?.abort();
      const ctl = new AbortController();
      abortRef.current = ctl;
      setIsPredicting(true);
      setError(null);
      try {
        const res = await apiClient.post("/predict", payload, { signal: ctl.signal });
        const next = sanitize(res.data);
        if (!next) {
          setError("Unexpected prediction response.");
          return;
        }
        let thumb: string | undefined;
        let pixelsSaved: number[] | undefined;
        let imageSaved: string | undefined;
        let textSaved: string | undefined;
        if (modelType === "ann" && Array.isArray(payload.pixels)) {
          pixelsSaved = payload.pixels as number[];
          thumb = toThumb(pixelsSaved);
        } else if (modelType === "cnn") {
          imageSaved = cnnRef.current ?? undefined;
          thumb = imageSaved;
        } else {
          textSaved = rnnRef.current;
        }
        setResult(next);
        setHistory((prev) =>
          [
            {
              id: String(Date.now()),
              family: modelType,
              modelId: next.modelId,
              modelName: next.modelName,
              prediction: next.prediction,
              confidence: next.confidence,
              thumbnail: thumb,
              pixels: pixelsSaved,
              image: imageSaved,
              text: textSaved,
            },
            ...prev,
          ].slice(0, 12),
        );
      } catch (e: any) {
        if (e?.name !== "CanceledError" && e?.code !== "ERR_CANCELED") {
          const detail = e?.response?.data?.detail;
          setError(typeof detail === "string" ? detail : "Prediction failed.");
        }
      } finally {
        setIsPredicting(false);
      }
    },
    [catalog, getPixels, modelType],
  );

  const predictRef = useRef(predictCurrent);
  predictRef.current = predictCurrent;

  useEffect(() => {
    const onCommandPredict = () => void predictRef.current();
    const onSelectModel = (event: Event) => {
      const model = (event as CustomEvent<CatalogModel>).detail;
      if (!model?.id || !model.family) return;
      const family = model.family.toLowerCase() as ModelType;
      setModelType(family);
      setSelected((prev) => ({ ...prev, [family]: model.id }));
      setResult(null);
      setError(null);
    };
    window.addEventListener(WORKSPACE_EVENTS.predict, onCommandPredict);
    window.addEventListener(WORKSPACE_EVENTS.selectModel, onSelectModel);
    return () => {
      window.removeEventListener(WORKSPACE_EVENTS.predict, onCommandPredict);
      window.removeEventListener(WORKSPACE_EVENTS.selectModel, onSelectModel);
    };
  }, []);

  const loadPixels = useCallback(
    (pixels: number[]) => {
      const cv = canvasRef.current;
      const ctx = cv?.getContext("2d");
      if (!cv || !ctx) return;
      const small = document.createElement("canvas");
      small.width = 28;
      small.height = 28;
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
      void predictCurrent(pixels);
    },
    [predictCurrent],
  );

  const loadPixelsRef = useRef(loadPixels);
  loadPixelsRef.current = loadPixels;

  const schedule = useCallback(() => {
    if (!autoPredict || modelType !== "ann") return;
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => void predictCurrent(), 300);
  }, [autoPredict, modelType, predictCurrent]);

  const onDraw = (e: React.PointerEvent) => {
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

  const startDraw = (e: React.PointerEvent) => {
    if (modelType !== "ann") return;
    e.currentTarget.setPointerCapture(e.pointerId);
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

  const stopDraw = (e?: React.PointerEvent) => {
    if (e && e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    drawRef.current = false;
    lastRef.current = null;
    schedule();
  };

  const clear = useCallback(() => {
    if (modelType !== "ann") return;
    pushUndo();
    resetCanvas();
    setResult(null);
    setError(null);
  }, [modelType, pushUndo, resetCanvas]);

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

  // ------------------------------------------------------------------
  // Data bootstrap
  // ------------------------------------------------------------------
  useEffect(() => {
    resetCanvas();
    setCatalogLoading(true);
    void (async () => {
      try {
        const [cat, s] = await Promise.all([
          apiClient.get("/models/catalog"),
          apiClient.get("/samples"),
        ]);
        const list: CatalogModel[] = Array.isArray(cat.data?.models) ? cat.data.models : [];
        const loadedIds = new Set<string>(Array.isArray(cat.data?.loaded) ? cat.data.loaded : []);
        const withLoaded = list.map((m) =>
          loadedIds.has(m.id) && m.status !== "loaded" ? { ...m, status: "loaded" as const } : m,
        );
        setCatalog(withLoaded);
        setSamples(s.data && typeof s.data === "object" ? s.data : {});
      } catch {
        setCatalog([]);
        setError("Could not fetch the model catalog. Is the backend running?");
      } finally {
        setCatalogLoading(false);
      }
    })();
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [resetCanvas]);

  const runnableFor = useCallback(
    (fam: ModelType) =>
      catalog.filter(
        (m) =>
          m.family.toUpperCase() === fam.toUpperCase() &&
          (m.status === "available" || m.status === "loaded"),
      ),
    [catalog],
  );

  // Revert selections that became invalid after a catalog refresh.
  useEffect(() => {
    setSelected((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const fam of ["ann", "cnn", "rnn"] as ModelType[]) {
        if (!runnableFor(fam).some((m) => m.id === next[fam])) {
          const first = runnableFor(fam)[0];
          if (first && first.id !== next[fam]) {
            next[fam] = first.id;
            changed = true;
          }
        }
      }
      return changed ? next : prev;
    });
  }, [runnableFor]);

  // Auto re-run when the *model inside the current family* changes.
  const selTrack = useRef<{ fam: ModelType; id: string }>({ fam: "ann", id: DEFAULT_MODEL.ann });
  useEffect(() => {
    const prev = selTrack.current;
    const cur = { fam: modelType, id: selected[modelType] };
    if (cur.fam === prev.fam && cur.id !== prev.id) {
      setResult(null);
      setError(null);
      if (cur.fam === "ann") void predictCurrent();
      else if (cur.fam === "cnn" && cnnRef.current) void predictCurrent();
      else if (cur.fam === "rnn" && rnnRef.current.trim()) void predictCurrent();
    }
    selTrack.current = cur;
  }, [selected, modelType, predictCurrent]);

  // ------------------------------------------------------------------
  // CNN input handlers
  // ------------------------------------------------------------------
  const onImageFile = (file?: File | null) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCnnImage(String(reader.result ?? ""));
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const loadSampleImage = async (name: string) => {
    try {
      const resp = await fetch(`/samples/${name}`);
      const blob = await resp.blob();
      const b64 = await new Promise<string>((resolve) => {
        const fr = new FileReader();
        fr.onload = () => resolve(String(fr.result ?? ""));
        fr.readAsDataURL(blob);
      });
      cnnRef.current = b64;
      setCnnImage(b64);
      setResult(null);
      setError(null);
      void predictCurrent();
    } catch {
      setError(`Could not load example image ${name}.`);
    }
  };

  // ------------------------------------------------------------------
  // Family switch
  // ------------------------------------------------------------------
  const switchFamily = (fam: ModelType) => {
    if (fam === modelType) return;
    setModelType(fam);
    setResult(null);
    setError(null);
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      const target = ev.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable=\"true\"]")) return;
      if (ev.code === "Space") { ev.preventDefault(); void predictCurrent(); }
      if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && ev.key.toLowerCase() === "z") { ev.preventDefault(); if (modelType === "ann") undoOne(); }
      if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && ev.key.toLowerCase() === "z") { ev.preventDefault(); if (modelType === "ann") redoOne(); }
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "x") { ev.preventDefault(); if (modelType === "ann") clear(); }
      if (ev.key === "1") switchFamily("ann");
      if (ev.key === "2") switchFamily("cnn");
      if (ev.key === "3") switchFamily("rnn");
      if (ev.key.toLowerCase() === "b") setProbView("bars");
      if (ev.key.toLowerCase() === "r") setProbView("radial");
      if (ev.key.toLowerCase() === "t") setProbView("terrain");
      if (ev.key.toLowerCase() === "g") setShowGrid((v) => !v);
      if (ev.key === "Escape") setTraceOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [predictCurrent, undoOne, redoOne, clear, modelType]);

  const annActs = useMemo(() => {
    const layer = result?.layers?.hidden1;
    return Array.isArray(layer) ? layer.slice(0, 96) : [];
  }, [result]);

  const clickHistory = (h: HistoryItem) => {
    setModelType(h.family);
    setSelected((prev) => ({ ...prev, [h.family]: h.modelId }));
    setResult(null);
    setError(null);
    if (h.family === "cnn" && h.image) setCnnImage(h.image);
    if (h.family === "rnn" && h.text != null) setRnnText(h.text);
    if (h.family === "ann" && h.pixels) {
      window.setTimeout(() => loadPixelsRef.current(h.pixels!), 90);
    } else {
      window.setTimeout(() => void predictRef.current(), 90);
    }
  };

  const fmtParams = (n: number | null | undefined) =>
    n == null ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;

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

  const hasLayers = result?.layers != null && Object.keys(result.layers).length > 0;

  return (
    <div className="h-full overflow-auto text-ink">
      <div className="page-shell [--shell-max:90rem] py-4 space-y-4">
        <div className="sticky top-0 z-20 -mx-1 border-b border-barley-line bg-barley-page/85 px-1 pb-2 backdrop-blur-md">
          <PageHeader
            eyebrow="Use a trained model"
            title="Prediction"
            subtitle="Input → model → result. Start with one task, then open analysis when you need it."
            status={
              <span className={`prediction-ready-pill ${catalogLoading ? "is-loading" : isReady ? "is-ready" : "is-not-ready"}`} role="status">
                <span aria-hidden="true" /> {catalogLoading ? "Loading model registry" : isReady ? "Ready" : "Select a runnable model"}
              </span>
            }
            actions={
              <NeuralButton size="sm" variant="secondary" onClick={() => setComparisonMode(true)}>
                Compare architectures
              </NeuralButton>
            }
          >
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <div className="flex gap-2" role="tablist" aria-label="Architecture selector">
                {(["ann", "cnn", "rnn"] as ModelType[]).map((m) => (
                  <button
                    key={m}
                    role="tab"
                    aria-selected={modelType === m}
                    type="button"
                    onClick={() => switchFamily(m)}
                    onKeyDown={(event) => {
                      if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) return;
                      event.preventDefault();
                      const familyOrder = ["ann", "cnn", "rnn"] as ModelType[];
                      const index = familyOrder.indexOf(m);
                      const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? familyOrder.length - 1 : (index + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + familyOrder.length) % familyOrder.length;
                      switchFamily(familyOrder[nextIndex]);
                      document.getElementById(`family-tab-${familyOrder[nextIndex]}`)?.focus();
                    }}
                    id={`family-tab-${m}`}
                    aria-controls="prediction-workbench"
                    className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${modelType === m ? "font-semibold" : "opacity-70 hover:opacity-100"}`}
                    style={{
                      color: modelType === m ? FAMILY_COLOR[m] : "#79716B",
                      borderColor: modelType === m ? `${FAMILY_COLOR[m]}66` : "#D8CFC0",
                      background: modelType === m ? `${FAMILY_COLOR[m]}14` : "transparent",
                    }}
                  >
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <label htmlFor="registry-model" className="text-[11px] uppercase tracking-wide text-ink-faint">
                  Model
                </label>
                <select
                  id="registry-model"
                  value={selected[modelType]}
                  onChange={(e) =>
                    setSelected((prev) => ({ ...prev, [modelType]: e.target.value }))
                  }
                  className="h-8 max-w-[320px] rounded-md border border-barley-linestrong bg-white px-2 text-xs font-medium text-ink focus:outline-none focus:ring-2 focus:ring-ember-600/40"
                  aria-label={`${modelType.toUpperCase()} pretrained model`}
                >
                  {catalogLoading ? <option value="" disabled>Loading model registry…</option> : null}
                  {runnableFor(modelType).map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} · {fmtParams(m.parameter_count)} params
                    </option>
                  ))}
                  {runnableFor(modelType).length === 0 && (
                    <option value={DEFAULT_MODEL[modelType]} disabled>
                      No runnable {modelType.toUpperCase()} model — check Models page
                    </option>
                  )}
                </select>
              </div>
            </div>
          </PageHeader>
        </div>

        <section className="prediction-contract" aria-label="Current prediction contract">
          <div>
            <span>Task</span>
            <strong>{modelRow?.dataset ?? (modelType === "ann" ? "MNIST digit classification" : modelType === "cnn" ? "ImageNet image classification" : "IMDB sentiment classification")}</strong>
          </div>
          <div>
            <span>Input</span>
            <strong>{modelRow?.input_shape ? `[${modelRow.input_shape.join(" × ")}]` : modelType === "ann" ? "28 × 28 grayscale" : modelType === "cnn" ? "224 × 224 RGB image" : "Movie review text"}</strong>
          </div>
          <div>
            <span>Model</span>
            <strong>{modelRow?.name ?? "Choose a model above"}</strong>
          </div>
          <div className="prediction-contract-status">
            <span>Status</span>
            <strong>{catalogLoading ? "Loading model registry" : isReady ? "Ready for inference" : modelRow?.unavailable_reason ?? "Model not ready"}</strong>
          </div>
        </section>

        {modelRow && (
          <div className="prediction-model-meta">
            <span className="inline-flex items-center gap-1"><Cpu size={12} /> {modelRow.framework}</span>
            {modelRow.dataset && <span className="inline-flex items-center gap-1"><Database size={12} /> {modelRow.dataset}</span>}
            <span className="inline-flex items-center gap-1"><Braces size={12} /> {modelRow.num_classes} classes</span>
            {modelRow.architecture && (
              <span className="inline-flex items-center gap-1"><Zap size={12} /> {modelRow.architecture}</span>
            )}
          </div>
        )}

        <section id="prediction-workbench" className="prediction-workbench rounded-2xl border border-barley-linestrong bg-barley-page p-4" aria-label="Prediction workbench">
          <div className="sr-only" role="status" aria-live="polite">{isPredicting ? "Running inference" : result ? `Prediction ready: ${result.prediction}` : ""}</div>
          <div className="prediction-workbench-flow">
            {/* ------------------------- input ------------------------- */}
            <div className="prediction-region space-y-3">
              <div className="prediction-region-heading">
                <span className="prediction-step">1</span>
                <div><strong>Input</strong><small>{modelType === "ann" ? "Draw a digit" : modelType === "cnn" ? "Upload an image" : "Write a review"}</small></div>
              </div>
              {modelType === "ann" && (
                <>
                  <div className="relative mx-auto" style={{ width: DISPLAY, height: DISPLAY }}>
                    <canvas
                      ref={canvasRef}
                      width={INTERNAL}
                      height={INTERNAL}
                      className="absolute inset-0 h-full w-full rounded-2xl border-2 cursor-crosshair"
                      role="img"
                      aria-label="Drawing canvas for digit prediction"
                      style={{
                        borderColor: `${color}66`,
                        boxShadow: `inset 0 0 35px ${color}22, 0 0 20px ${color}22`,
                        touchAction: "none",
                      }}
                      onPointerDown={startDraw}
                      onPointerMove={onDraw}
                      onPointerUp={stopDraw}
                      onPointerCancel={stopDraw}
                      onPointerLeave={stopDraw}
                    />
                    {showGrid && (
                      <div
                        className="absolute inset-0 rounded-2xl pointer-events-none"
                        style={{
                          backgroundImage:
                            "linear-gradient(to right, rgba(28,25,23,0.05) 1px, transparent 1px),linear-gradient(to bottom, rgba(28,25,23,0.05) 1px, transparent 1px)",
                          backgroundSize: "calc(100% / 28) calc(100% / 28)",
                        }}
                      />
                    )}
                  </div>
                  <div className="w-[280px] mx-auto space-y-2">
                    <div className="flex gap-2">
                      <button type="button" onClick={undoOne} disabled={undo.length === 0} aria-label="Undo stroke" title="Undo" className="h-9 w-9 grid place-items-center rounded-md border border-barley-linestrong bg-barley-sunken hover:bg-barley-wash disabled:opacity-35 disabled:cursor-not-allowed"><Undo2 size={16} /></button>
                      <button type="button" onClick={redoOne} disabled={redo.length === 0} aria-label="Redo stroke" title="Redo" className="h-9 w-9 grid place-items-center rounded-md border border-barley-linestrong bg-barley-sunken hover:bg-barley-wash disabled:opacity-35 disabled:cursor-not-allowed"><Redo2 size={16} /></button>
                      <button type="button" onClick={clear} aria-label="Clear canvas" title="Clear" className="h-9 w-9 grid place-items-center rounded-md border border-status-danger/35 bg-status-danger/5 hover:bg-status-danger/15 text-status-danger active:scale-95"><X size={16} /></button>
                      <button type="button" onClick={() => setShowGrid((v) => !v)} aria-label="Toggle grid overlay" aria-pressed={showGrid} title="Grid" className={`h-9 w-9 grid place-items-center rounded-md border ${showGrid ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash"}`}><Grid3X3 size={16} /></button>
                      <button type="button" onClick={() => void predictCurrent()} className="h-9 flex-1 rounded-md border border-ember-600/40 bg-ember-600/15 hover:bg-ember-600/25 active:scale-[0.98] text-ember-700 text-xs inline-flex items-center justify-center gap-1"><Play size={14} /> Predict</button>
                    </div>
                    <div className="text-[12px] text-ink-mute flex justify-between">
                      <span>Undo {undo.length}/10</span>
                      <button type="button" className="underline decoration-dotted" aria-pressed={autoPredict} onClick={() => setAutoPredict((v) => !v)}>Auto {autoPredict ? "on" : "off"}</button>
                    </div>
                    <div className="grid grid-cols-10 gap-1">
                      {Array.from({ length: 10 }, (_, i) => (
                        <button
                          type="button"
                          key={i}
                          aria-label={`Load sample digit ${i}`}
                          className="h-7 rounded border border-barley-line bg-white text-xs font-mono hover:border-ember-600/50 hover:text-ember-700"
                          onClick={() => {
                            const px = samples[String(i)];
                            if (Array.isArray(px)) loadPixels(px);
                          }}
                        >
                          {i}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {modelType === "cnn" && (
                <div className="space-y-3 mx-auto w-[320px] max-w-full">
                  <label
                    className="relative flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-6 cursor-pointer transition-colors hover:bg-white/60"
                    style={{ borderColor: `${color}77`, background: cnnImage ? "#fff" : `${color}0a` }}
                  >
                    {cnnImage ? (
                      <>
                        <img src={cnnImage} alt="Uploaded input" className="max-h-44 rounded-lg border border-barley-linestrong" />
                        <span className="text-[12px] text-ink-soft inline-flex items-center gap-1">
                          <FileImage size={13} /> Loaded — 224×224 RGB → model preprocess_input
                        </span>
                      </>
                    ) : (
                      <>
                        <ImagePlus size={26} className="text-ink-faint" aria-hidden="true" />
                        <span className="text-xs font-medium text-ink-soft">Upload an image</span>
                        <span className="text-[11px] text-ink-faint text-center">
                          Resized to 224×224, then fed through the ImageNet preprocessing of the selected CNN.
                        </span>
                      </>
                    )}
                    <input
                      type="file"
                      accept="image/*"
                      className="sr-only"
                      aria-label="Upload an image to classify"
                      onChange={(e) => onImageFile(e.target.files?.[0])}
                    />
                  </label>
                  <div className="flex items-center gap-2 text-[12px]">
                    <span className="text-ink-faint uppercase tracking-wide">Examples</span>
                    {SAMPLE_IMAGES.map((name) => (
                      <button
                        type="button"
                        key={name}
                        onClick={() => void loadSampleImage(name)}
                        className="px-2 py-1 rounded-md border border-barley-line bg-white hover:bg-barley-wash text-[11px] font-medium"
                        style={{ color }}
                      >
                        {name.replace(/\.jpg$/, "").replace(/_/g, " ")}
                      </button>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => void predictCurrent()}
                      disabled={!cnnImage}
                      className="flex-1 h-9 rounded-md border text-xs font-semibold inline-flex items-center justify-center gap-1 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                      style={{ borderColor: `${color}66`, background: `${color}18`, color }}
                    >
                      <Play size={14} /> Classify image
                    </button>
                    {cnnImage && (
                      <button
                        type="button"
                        onClick={() => { setCnnImage(null); setResult(null); setError(null); }}
                        aria-label="Clear image"
                        title="Clear image"
                        className="h-9 w-9 grid place-items-center rounded-md border border-status-danger/35 bg-status-danger/5 hover:bg-status-danger/15 text-status-danger"
                      >
                        <ClearImage size={15} />
                      </button>
                    )}
                  </div>
                </div>
              )}

              {modelType === "rnn" && (
                <div className="space-y-3 mx-auto w-[340px] max-w-full">
                  <div className="rounded-2xl border border-barley-linestrong bg-white p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[12px] text-ink-faint uppercase tracking-wide">Movie review</span>
                      <span className="font-mono text-[11px] text-ink-mute">{rnnText.trim().split(/\s+/).filter(Boolean).length} words</span>
                    </div>
                    <textarea
                      value={rnnText}
                      onChange={(e) => { setRnnText(e.target.value); setResult(null); setError(null); }}
                      rows={5}
                      placeholder="Type an IMDB-style review… the BiLSTM will classify it as positive or negative."
                      className="w-full resize-y rounded-lg border border-barley-line bg-barley-sunken/60 p-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-ember-600/30"
                    />
                    <div className="flex flex-wrap gap-2 mt-2">
                      {RNN_PRESETS.map((s, i) => (
                        <button
                          type="button"
                          key={i}
                          onClick={() => { setRnnText(s); setResult(null); setError(null); }}
                          className="px-2 py-1 rounded-md border border-barley-line bg-barley-sunken hover:bg-barley-wash text-[11px]"
                        >
                          {i === 0 ? "Positive sample" : "Negative sample"}
                        </button>
                      ))}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => void predictCurrent()}
                    disabled={!rnnText.trim()}
                    className="w-full h-9 rounded-md border text-xs font-semibold inline-flex items-center justify-center gap-1 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    style={{ borderColor: `${color}66`, background: `${color}18`, color }}
                  >
                    <Play size={14} /> Classify sentiment
                  </button>
                </div>
              )}
            </div>

            {/* ---------------------- middle info ---------------------- */}
            <div className="prediction-flow-connector hidden md:flex flex-col items-center justify-center gap-2 text-center px-2" aria-hidden="true">
              <span>Inference</span>
              <div className="prediction-flow-line" />
              {modelRow?.preprocessing && (
                <div className="text-[12px] text-ink-soft leading-relaxed max-w-[32ch]">
                  <span className="text-ink-faint uppercase tracking-wide text-[10px] block mb-1">Preprocessing</span>
                  {modelRow.preprocessing}
                </div>
              )}
              {isPredicting ? (
                <RefreshCw size={16} className="animate-spin text-ink-faint" />
              ) : (
                <div className="text-[11px] text-ink-faint font-mono">
                  {modelType === "ann"
                    ? "Draw on the 28×28 grid — live predictions as you draw"
                    : modelType === "cnn"
                      ? "224×224 RGB input, resized before inference"
                      : "Tokenizer + padded sequence → bidirectional LSTM"}
                </div>
              )}
            </div>

            {/* ------------------------- output ------------------------ */}
            <div className="prediction-region prediction-result-region mx-auto">
              <div className="prediction-region-heading">
                <span className="prediction-step">2</span>
                <div><strong>Result</strong><small>Prediction, confidence, latency</small></div>
              </div>
              <div className="relative w-[300px] max-w-full h-[300px] mx-auto">
              <svg className="absolute inset-0" viewBox="0 0 300 300">
                <g transform="translate(150,150)">
                  <circle r={R} fill="none" stroke="rgba(28,25,23,0.10)" strokeWidth="6" />
                  <circle
                    r={R}
                    fill="none"
                    stroke={confColor}
                    strokeWidth="6"
                    strokeDasharray={C}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    transform="rotate(-90)"
                    style={{ transition: "stroke-dashoffset 320ms ease-out" }}
                  />
                </g>
              </svg>
              <div className="absolute inset-0 grid place-items-center text-center px-6">
                {error ? (
                  <div role="alert" className="text-status-danger text-sm leading-snug max-w-[250px] break-words">{error}</div>
                ) : result ? (
                  <div>
                    <div
                      className="font-bold leading-none break-words"
                      style={{
                        color,
                        fontSize: result.prediction.length > 12 ? "26px" : result.prediction.length > 6 ? "44px" : "100px",
                        lineHeight: 1.05,
                      }}
                    >
                      {result.prediction}
                    </div>
                    <div className="text-sm font-medium mt-2" style={{ color: confColor }}>
                      {(conf * 100).toFixed(1)}% confident
                    </div>
                    {result.device && (
                      <div className="text-[11px] text-ink-faint font-mono mt-1">{result.modelName} · {result.device}</div>
                    )}
                    {result.latencyMs != null && (
                      <div className="prediction-result-metrics" aria-label="Prediction run metrics">
                        <span><b>{result.latencyMs.toFixed(1)} ms</b> latency</span>
                        <span><b>{result.topK.length || result.labels.length}</b> classes scored</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-ink-faint">
                    {modelType === "ann" && <div className="text-5xl font-bold mb-1">?</div>}
                    {modelType === "cnn" && <ImagePlus size={34} className="mx-auto mb-2 opacity-40" />}
                    {modelType === "rnn" && <FileText size={30} className="mx-auto mb-2 opacity-40" />}
                    <div className="text-sm">
                      {modelType === "ann"
                        ? "Draw to predict"
                        : modelType === "cnn"
                          ? "Upload an image to begin"
                          : "Type a review to begin"}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          </div>
        </section>

        {/* ---------------------- probability views -------------------- */}
        <section className="prediction-analysis rounded-2xl border border-barley-linestrong bg-white overflow-hidden">
          <button
            type="button"
            className="prediction-analysis-toggle"
            onClick={() => setAnalysisOpen((open) => !open)}
            aria-expanded={analysisOpen}
            aria-controls="prediction-analysis-content"
          >
            <span>
              <strong>Analysis</strong>
              <small>Top predictions, probability shape, and model internals</small>
            </span>
            <ChevronDown size={16} className={analysisOpen ? "rotate-180" : ""} aria-hidden="true" />
          </button>
          {analysisOpen ? <div id="prediction-analysis-content" className="p-4 border-t border-barley-line">
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold text-ember-700">
              Probability landscape
              {result?.modelName && <span className="ml-2 text-[11px] font-normal text-ink-mute">· {result.modelName}</span>}
            </div>
            <div className="flex gap-1">
              <button type="button" onClick={() => setProbView("bars")} aria-label="Bar view" aria-pressed={probView === "bars"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "bars" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Columns size={14} /></button>
              <button type="button" onClick={() => setProbView("radial")} aria-label="Radial view" aria-pressed={probView === "radial"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "radial" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Orbit size={14} /></button>
              <button type="button" onClick={() => setProbView("terrain")} aria-label="Terrain view" aria-pressed={probView === "terrain"} className={`h-8 w-9 grid place-items-center rounded-md border ${probView === "terrain" ? "border-ember-600/40 bg-ember-600/15 text-ember-700" : "border-barley-linestrong bg-barley-sunken hover:bg-barley-wash text-ink-mute"}`}><Mountain size={14} /></button>
            </div>
          </div>

          {bars.length === 0 ? (
            <div className="h-[160px] grid place-items-center text-sm text-ink-faint">No prediction yet.</div>
          ) : (
            <>
              {probView === "bars" && (
                <div className="h-[180px] flex items-end gap-2" style={{ maxWidth: Math.min(860, bars.length * 96) }}>
                  {bars.map((b, i) => (
                    <div key={b.index} className="relative h-full flex-1 min-w-0 rounded-md bg-barley-sunken border border-barley-line overflow-hidden">
                      <div
                        className="absolute bottom-0 left-0 right-0 origin-bottom transition-transform duration-300"
                        style={{
                          height: "100%",
                          transform: `scaleY(${Math.max(0.02, b.prob)})`,
                          background: i === topIdx ? color : `${color}99`,
                          boxShadow: i === topIdx ? `0 0 18px ${color}66` : "none",
                        }}
                      />
                      <div className="absolute bottom-1 inset-x-0 text-center text-[12px] font-mono text-ink px-0.5 truncate" title={b.label}>
                        {b.label}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {probView === "radial" && (
                <div className="h-[200px] grid place-items-center overflow-hidden">
                  <svg width={Math.max(320, bars.length * 36)} height="200" viewBox={`0 0 ${Math.max(320, bars.length * 36)} 200`}>
                    <g transform={`translate(${Math.max(160, bars.length * 18)},95)`}>
                      {bars.map((b, i) => {
                        const a = (Math.PI * 2 * i) / bars.length - Math.PI / 2;
                        const len = 26 + b.prob * 65;
                        return (
                          <line
                            key={b.index}
                            x1={0} y1={0} x2={Math.cos(a) * len} y2={Math.sin(a) * len}
                            stroke={i === topIdx ? color : `${color}66`}
                            strokeWidth={i === topIdx ? 5 : 3}
                          />
                        );
                      })}
                      <circle r={22} fill="rgba(28,25,23,0.04)" stroke={`${color}66`} />
                      <text x="0" y="4" fill={color} fontSize="13" textAnchor="middle" dominantBaseline="middle" style={{ fontWeight: 700 }}>
                        {result?.prediction ?? "?"}
                      </text>
                    </g>
                  </svg>
                </div>
              )}

              {probView === "terrain" && (
                <div className="h-[200px] flex items-end justify-center gap-2 [perspective:700px] overflow-hidden">
                  {bars.map((b, i) => (
                    <div key={b.index} className="relative w-10 shrink-0">
                      <div
                        className="absolute bottom-0 left-0 w-10 rounded-t-sm transition-all duration-300"
                        style={{
                          height: Math.max(4, b.prob * 150),
                          background: `linear-gradient(to top, ${color}55, ${color})`,
                          boxShadow: i === topIdx ? `0 0 18px ${color}66` : "none",
                          transform: "rotateX(20deg)",
                        }}
                      />
                      <div className="absolute -bottom-5 w-full text-center text-[11px] font-mono truncate px-0.5">{b.label}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          </div> : null}
        </section>

        {/* ---------------------- run details -------------------------- */}
        <div className="rounded-xl border border-barley-linestrong bg-white overflow-hidden">
          <button type="button" onClick={() => setTraceOpen((v) => !v)} aria-expanded={traceOpen} className="w-full h-12 px-4 flex items-center justify-between text-sm">
            <span className="text-ember-700 inline-flex items-center gap-1">
              <ChevronDown size={14} className={`transition-transform duration-200 ${traceOpen ? "" : "-rotate-90"}`} />
              What the network saw
            </span>
            <span className="text-xs text-ink-mute">Run details &amp; internals</span>
          </button>
          {traceOpen && (
            <div className="p-4 border-t border-barley-line space-y-3">
              {hasLayers && (
                <div className="grid gap-[2px]" style={{ gridTemplateColumns: "repeat(32, minmax(0, 1fr))" }}>
                  {annActs.map((a, i) => (
                    <div key={i} className="h-5 rounded-[2px]" style={{ background: `rgba(194,65,12,${0.1 + clamp01(a) * 0.9})` }} />
                  ))}
                </div>
              )}
              {!hasLayers && result?.explanation && (
                <div className="text-xs text-ink-soft break-words">{JSON.stringify(result.explanation).slice(0, 400)}</div>
              )}
              {result && modelRow && (
                <dl className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1.5 text-[11px]">
                  <div className="flex gap-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Model</dt><dd className="font-mono text-ink-soft break-all">{result.modelId}</dd></div>
                  <div className="flex gap-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Latency</dt><dd className="font-mono text-ink-soft">{result.latencyMs?.toFixed(1)} ms</dd></div>
                  <div className="flex gap-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Device</dt><dd className="font-mono text-ink-soft">{result.device ?? "CPU"}</dd></div>
                  <div className="flex gap-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Params</dt><dd className="font-mono text-ink-soft">{fmtParams(result.parameterCount ?? modelRow.parameter_count)}</dd></div>
                  <div className="flex gap-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Classes</dt><dd className="font-mono text-ink-soft">{result.labels.length || modelRow.num_classes}</dd></div>
                  <div className="flex gap-2 col-span-2"><dt className="text-ink-faint uppercase w-16 shrink-0">Dataset</dt><dd className="text-ink-soft">{result.dataset ?? modelRow.dataset ?? "—"}</dd></div>
                </dl>
              )}
            </div>
          )}
        </div>

        {/* ------------------------- history --------------------------- */}
        <div className="rounded-xl border border-barley-linestrong bg-barley-page p-3">
          <div className="text-xs text-ink-mute mb-2">History</div>
          <div className="flex gap-2 overflow-x-auto pb-1" role="list">
            {history.map((h) => (
              <div key={h.id} role="listitem" className="shrink-0 w-[72px] text-center group">
                <button
                  type="button"
                  aria-label={`Restore ${h.modelName} prediction ${h.prediction}`}
                  className="w-full text-center"
                  onClick={() => clickHistory(h)}
                >
                {h.thumbnail ? (
                  <img src={h.thumbnail} alt="" className="w-14 h-14 rounded-md border border-barley-linestrong bg-ink mx-auto object-cover" />
                ) : (
                  <div
                    className="w-14 h-14 rounded-md border border-barley-linestrong grid place-items-center text-[10px] font-bold mx-auto"
                    style={{ background: `${FAMILY_COLOR[h.family]}12`, color: FAMILY_COLOR[h.family] }}
                  >
                    {h.family.toUpperCase()}
                  </div>
                )}
                <div className="text-[11px] mt-1 truncate px-0.5" style={{ color: FAMILY_COLOR[h.family] }} title={h.prediction}>
                  {h.prediction}
                </div>
                <div className="text-[11px] text-ink-mute">{Math.round(h.confidence * 100)}%</div>
                </button>
              </div>
            ))}
            {history.length === 0 && <div className="text-xs text-ink-faint py-4">No predictions yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
