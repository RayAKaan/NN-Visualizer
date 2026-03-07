import React, { useEffect, useMemo, useRef, useState } from "react";
import { apiClient } from "../../api/client";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { EdgeState, NeuronState } from "./types";

interface Props {
  onHoverPosition: (x: number, y: number) => void;
  onHoverNeuronData: (neuron: NeuronState | null) => void;
  onSelectNeuronData: (neuron: NeuronState | null) => void;
  inputPixels?: number[] | null;
}

type Speed = "slow" | "medium" | "fast";

interface CNNPredictPayload {
  probabilities?: number[];
  feature_maps?: Array<{
    layer_name: string;
    shape: number[];
    activation_ranking?: number[];
    feature_maps: number[][][];
  }>;
  kernels?: Array<{
    layer_name: string;
    kernel_shape: number[];
    kernels: number[][][][];
  }>;
  dense_layers?: Record<string, number[]>;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const relu = (v: number) => (v > 0 ? v : 0);

const viridis = (v: number) => {
  const t = clamp01(v);
  const r = Math.round(68 + t * 185);
  const g = Math.round(1 + t * 230);
  const b = Math.round(84 + (1 - t) * 110);
  return `rgb(${r},${g},${b})`;
};

const inferno = (v: number) => {
  const t = clamp01(v);
  const r = Math.round(20 + t * 235);
  const g = Math.round(10 + t * 120);
  const b = Math.round(15 + (1 - t) * 70);
  return `rgb(${r},${g},${b})`;
};

const normalizeMap = (arr: number[]) => {
  const mn = Math.min(...arr);
  const mx = Math.max(...arr);
  const d = mx - mn || 1;
  return arr.map((v) => (v - mn) / d);
};

function HeatmapCanvas({
  data,
  size,
  cell,
  color,
  className,
  highlight,
  highlightPatch,
  onHoverCell,
  onClickCell,
}: {
  data: number[];
  size: number;
  cell: number;
  color: "viridis" | "inferno" | "gray" | "signed";
  className?: string;
  highlight?: { r: number; c: number } | null;
  highlightPatch?: { r: number; c: number; h: number; w: number } | null;
  onHoverCell?: (r: number, c: number, v: number, ev: React.MouseEvent<HTMLCanvasElement>) => void;
  onClickCell?: (r: number, c: number, v: number, ev: React.MouseEvent<HTMLCanvasElement>) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    cv.width = size * cell;
    cv.height = size * cell;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const vals = color === "signed" ? data.map((v) => (v + 1) / 2) : normalizeMap(data);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const i = r * size + c;
        const raw = data[i] ?? 0;
        let fill = "#000";
        if (color === "gray") {
          const g = Math.round(clamp01(raw) * 255);
          fill = `rgb(${g},${g},${g})`;
        } else if (color === "viridis") {
          fill = viridis(vals[i] ?? 0);
        } else if (color === "inferno") {
          fill = inferno(vals[i] ?? 0);
        } else {
          const a = Math.min(0.95, Math.abs(raw));
          fill = raw >= 0 ? `rgba(239,68,68,${a})` : `rgba(59,130,246,${a})`;
        }
        ctx.fillStyle = fill;
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }

    if (highlightPatch) {
      ctx.strokeStyle = "rgba(34,211,238,0.95)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(highlightPatch.c * cell + 1, highlightPatch.r * cell + 1, highlightPatch.w * cell - 2, highlightPatch.h * cell - 2);
      ctx.setLineDash([]);
    }

    if (highlight) {
      ctx.strokeStyle = "rgba(132,204,22,0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(highlight.c * cell + 1, highlight.r * cell + 1, cell - 2, cell - 2);
    }
  }, [data, size, cell, color, highlight, highlightPatch]);

  const locate = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const c = Math.max(0, Math.min(size - 1, Math.floor(((ev.clientX - rect.left) / rect.width) * size)));
    const r = Math.max(0, Math.min(size - 1, Math.floor(((ev.clientY - rect.top) / rect.height) * size)));
    return { r, c, v: data[r * size + c] ?? 0 };
  };

  return (
    <canvas
      ref={ref}
      className={className}
      onMouseMove={(ev) => {
        if (!onHoverCell) return;
        const { r, c, v } = locate(ev);
        onHoverCell(r, c, v, ev);
      }}
      onClick={(ev) => {
        if (!onClickCell) return;
        const { r, c, v } = locate(ev);
        onClickCell(r, c, v, ev);
      }}
    />
  );
}

function maxPool2(map: number[], size: number) {
  const outSize = Math.floor(size / 2);
  const out = Array.from({ length: outSize * outSize }, () => 0);
  for (let r = 0; r < outSize; r++) {
    for (let c = 0; c < outSize; c++) {
      let mx = -Infinity;
      for (let dr = 0; dr < 2; dr++) {
        for (let dc = 0; dc < 2; dc++) {
          const rr = r * 2 + dr;
          const cc = c * 2 + dc;
          mx = Math.max(mx, map[rr * size + cc]);
        }
      }
      out[r * outSize + c] = mx;
    }
  }
  return out;
}

const makeNeuron = (id: string, layerType: NeuronState["layerType"], activation: number, incoming: EdgeState[] = []): NeuronState => ({
  id,
  layerType,
  activation,
  bias: 0,
  gradient: 0,
  incomingEdges: incoming,
  outgoingEdges: [],
});

const flatten2D = (m: number[][]) => m.flat();

const sampleFallback = () => Array.from({ length: 784 }, (_, i) => {
  const r = Math.floor(i / 28);
  const c = i % 28;
  const d = Math.hypot(c - 13.5, r - 13.5);
  return clamp01(Math.exp(-(d * d) / 58));
});

export default function CNNVisualizer({ onHoverPosition, onHoverNeuronData, onSelectNeuronData, inputPixels }: Props) {
  const mode = useNeurofluxStore((s) => s.mode);
  const setHoveredNeuron = useNeurofluxStore((s) => s.setHoveredNeuron);
  const setSelectedNeuron = useNeurofluxStore((s) => s.setSelectedNeuron);

  const [selectedFilter, setSelectedFilter] = useState(0);
  const [speed, setSpeed] = useState<Speed>("medium");
  const [playingConv, setPlayingConv] = useState(false);
  const [convStep, setConvStep] = useState(0);
  const [poolStep, setPoolStep] = useState(0);
  const [showRF, setShowRF] = useState(true);
  const [rfPatch, setRfPatch] = useState<{ r: number; c: number; h: number; w: number } | null>(null);
  const [sampleDigit, setSampleDigit] = useState(7);
  const hasLiveInput = Array.isArray(inputPixels) && inputPixels.length === 784;
  const requestInFlightRef = useRef(false);
  const hasLoadedOnceRef = useRef(false);
  const [refreshTick, setRefreshTick] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [input, setInput] = useState<number[]>(sampleFallback());
  const [conv1, setConv1] = useState<number[][]>([]);
  const [conv1Raw, setConv1Raw] = useState<number[][]>([]);
  const [pool1, setPool1] = useState<number[][]>([]);
  const [conv2, setConv2] = useState<number[][]>([]);
  const [pool2, setPool2] = useState<number[][]>([]);
  const [probs, setProbs] = useState<number[]>(Array.from({ length: 10 }, () => 0.1));
  const [dense, setDense] = useState<number[]>(Array.from({ length: 128 }, () => 0));
  const [k1All, setK1All] = useState<number[][]>(Array.from({ length: 8 }, () => Array.from({ length: 9 }, () => 0)));
  const [conv1Ranking, setConv1Ranking] = useState<number[]>(Array.from({ length: 8 }, (_, i) => i));
  const realtimeActive = mode === "training" || hasLiveInput;

  useEffect(() => {
    if (!realtimeActive) return;
    const id = window.setInterval(() => {
      setRefreshTick((t) => t + 1);
    }, 450);
    return () => window.clearInterval(id);
  }, [realtimeActive]);

  useEffect(() => {
    let canceled = false;
    const load = async () => {
      if (requestInFlightRef.current) return;
      requestInFlightRef.current = true;
      if (!hasLoadedOnceRef.current) setLoading(true);
      setError(null);
      try {
                let sample: number[];
        if (hasLiveInput) {
          sample = inputPixels as number[];
        } else {
          const samplesRes = await apiClient.get("/samples");
          sample = Array.isArray(samplesRes.data?.[String(sampleDigit)])
            ? (samplesRes.data[String(sampleDigit)] as number[])
            : sampleFallback();
        }

        const predRes = await apiClient.post<CNNPredictPayload>("/predict", {
          pixels: sample,
          model_type: "cnn",
        });

        if (canceled) return;

        const payload = predRes.data;
        const fmap = Array.isArray(payload.feature_maps) ? payload.feature_maps : [];
        const bySize = (sz: number) => fmap.find((x) => Array.isArray(x.shape) && x.shape[0] === sz);

        const conv1Layer = bySize(26);
        const pool1Layer = bySize(13);
        const conv2Layer = bySize(11);
        const pool2Layer = bySize(5);

        const conv1Maps = (conv1Layer?.feature_maps ?? []).map((m) => flatten2D(m));
        const conv1Rank = (conv1Layer?.activation_ranking ?? []).slice(0, Math.max(8, conv1Maps.length));

        const pool1Maps = (pool1Layer?.feature_maps ?? []).map((m) => flatten2D(m));
        const conv2Maps = (conv2Layer?.feature_maps ?? []).map((m) => flatten2D(m));
        const pool2Maps = (pool2Layer?.feature_maps ?? []).map((m) => flatten2D(m));

        const kernels = Array.isArray(payload.kernels) ? payload.kernels : [];
        const k1 = kernels[0]?.kernels;
        const k1Parsed = Array.from({ length: Math.max(8, conv1Maps.length || 8) }, (_, f) => {
          const outIdx = conv1Rank[f] ?? f;
          const vals: number[] = [];
          for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
              vals.push(k1?.[r]?.[c]?.[0]?.[outIdx] ?? 0);
            }
          }
          return vals;
        });

        const denseLayer = payload.dense_layers
          ? Object.values(payload.dense_layers)[0] ?? []
          : [];

        const c1 = conv1Maps.length > 0 ? conv1Maps : [sample.slice(0, 26 * 26)];
        const p1 = pool1Maps.length > 0 ? pool1Maps : c1.map((m) => maxPool2(m, 26));
        const c2 = conv2Maps.length > 0 ? conv2Maps : p1.slice(0, 4).map((m) => m.slice(0, 11 * 11));
        const p2 = pool2Maps.length > 0 ? pool2Maps : c2.map((m) => maxPool2(m, 11));

        setInput(sample);
        setConv1(c1);
        setConv1Raw(c1);
        setPool1(p1);
        setConv2(c2);
        setPool2(p2);
        setProbs(Array.isArray(payload.probabilities) ? payload.probabilities : Array.from({ length: 10 }, () => 0.1));
        setDense(denseLayer.length > 0 ? denseLayer : Array.from({ length: 128 }, () => 0));
        setConv1Ranking(conv1Rank.length > 0 ? conv1Rank : Array.from({ length: c1.length }, (_, i) => i));
        setK1All(k1Parsed);
        setSelectedFilter((f) => Math.min(f, Math.max(0, c1.length - 1)));
        hasLoadedOnceRef.current = true;
      } catch {
        if (canceled) return;
        setError("Using fallback synthetic CNN tensors (backend CNN introspection unavailable).");
      } finally {
        requestInFlightRef.current = false;
        if (!canceled) setLoading(false);
      }
    };

    void load();
    return () => {
      canceled = true;
    };
  }, [sampleDigit, hasLiveInput, inputPixels, refreshTick]);

  useEffect(() => {
    if (!playingConv) return;
    const ms = speed === "slow" ? 700 : speed === "medium" ? 180 : 40;
    const id = window.setInterval(() => {
      setConvStep((prev) => (prev + 1) % (26 * 26));
    }, ms);
    return () => window.clearInterval(id);
  }, [playingConv, speed]);

  useEffect(() => {
    const id = window.setInterval(() => {
      setPoolStep((p) => (p + 1) % (13 * 13));
    }, 260);
    return () => window.clearInterval(id);
  }, []);

  const predicted = probs.indexOf(Math.max(...probs));
  const convPos = { r: Math.floor(convStep / 26), c: convStep % 26 };
  const poolPos = { r: Math.floor(poolStep / 13), c: poolStep % 13 };
  const flatten = useMemo(() => pool2.flat(), [pool2]);

  const onHoverNeuron = (n: NeuronState, ev: React.MouseEvent) => {
    setHoveredNeuron(n.id);
    onHoverPosition(ev.clientX, ev.clientY);
    onHoverNeuronData(n);
  };

  const onSelectNeuron = (n: NeuronState) => {
    setSelectedNeuron(n.id);
    onSelectNeuronData(n);
  };

  const currentKernel = k1All[selectedFilter] ?? Array.from({ length: 9 }, () => 0);

  const convNeuronAtStep = useMemo(() => {
    const incoming: EdgeState[] = [];
    for (let kr = 0; kr < 3; kr++) {
      for (let kc = 0; kc < 3; kc++) {
        const rr = convPos.r + kr;
        const cc = convPos.c + kc;
        const px = input[rr * 28 + cc] ?? 0;
        const w = currentKernel[kr * 3 + kc] ?? 0;
        incoming.push({
          id: `I_${rr}_${cc}_to_C1_${selectedFilter}_${convPos.r}_${convPos.c}_${kr}_${kc}`,
          from: `I_${rr}_${cc}`,
          to: `C1_${selectedFilter}_${convPos.r}_${convPos.c}`,
          weight: w,
          gradient: 0,
          contribution: w * px,
        });
      }
    }
    const z = incoming.reduce((s, e) => s + e.contribution, 0);
    return makeNeuron(`C1_${selectedFilter}_${convPos.r}_${convPos.c}`, "conv", relu(z), incoming);
  }, [selectedFilter, convPos, input, currentKernel]);

  const poolNeuronAtStep = useMemo(() => {
    const map = conv1[selectedFilter] ?? Array.from({ length: 26 * 26 }, () => 0);
    const incoming: EdgeState[] = [];
    for (let dr = 0; dr < 2; dr++) {
      for (let dc = 0; dc < 2; dc++) {
        const rr = poolPos.r * 2 + dr;
        const cc = poolPos.c * 2 + dc;
        const v = map[rr * 26 + cc] ?? 0;
        incoming.push({
          id: `C1_${selectedFilter}_${rr}_${cc}_to_P1_${selectedFilter}_${poolPos.r}_${poolPos.c}`,
          from: `C1_${selectedFilter}_${rr}_${cc}`,
          to: `P1_${selectedFilter}_${poolPos.r}_${poolPos.c}`,
          weight: 1,
          gradient: 0,
          contribution: v,
        });
      }
    }
    return makeNeuron(`P1_${selectedFilter}_${poolPos.r}_${poolPos.c}`, "pool", Math.max(...incoming.map((e) => e.contribution)), incoming);
  }, [conv1, selectedFilter, poolPos]);

  const rfForPool2 = (r: number, c: number) => ({ r: r * 4, c: c * 4, h: 10, w: 10 });

  return (
    <div className="h-full w-full rounded-xl border border-slate-700 bg-[#0a0a0f] nv-viz-surface p-3 overflow-auto">
      <div className="min-w-[1650px] space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-mono text-cyan-300">CNN Pipeline: Input -&gt; Conv1 -&gt; Pool1 -&gt; Conv2 -&gt; Pool2 -&gt; Flatten -&gt; Dense -&gt; Output</h3>
          <div className="flex items-center gap-2">
            {!hasLiveInput && (
            <select
              value={sampleDigit}
              onChange={(e) => setSampleDigit(Number(e.target.value))}
              className="text-xs bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200"
            >
              {Array.from({ length: 10 }, (_, d) => (
                <option key={d} value={d}>Sample {d}</option>
              ))}
            </select>
            )}
            {hasLiveInput && <div className="text-xs text-cyan-300 font-mono px-2 py-1 border border-cyan-700 rounded bg-cyan-950/30">Live Lab Input</div>}
            <label className="text-xs text-slate-300 border border-slate-700 rounded px-2 py-1 bg-slate-900 flex items-center gap-2">
              <input type="checkbox" checked={showRF} onChange={() => setShowRF((v) => !v)} className="accent-cyan-500" /> Show Receptive Field
            </label>
          </div>
        </div>

        {loading && <div className="text-xs text-cyan-300 font-mono">Loading real CNN tensors from backend...</div>}
        {error && <div className="text-xs text-amber-300 font-mono">{error}</div>}
        {realtimeActive && <div className="text-xs text-emerald-300 font-mono">Realtime updates: ON (refresh ~450ms)</div>}

        <div className="grid grid-cols-[220px_460px_300px_300px] gap-4">
          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 1: Input Image</div>
            <HeatmapCanvas
              data={input}
              size={28}
              cell={6}
              color="gray"
              className="rounded border border-slate-700"
              highlightPatch={showRF ? rfPatch : null}
              onHoverCell={(r, c, v, ev) => {
                onHoverPosition(ev.clientX, ev.clientY);
                onHoverNeuronData(makeNeuron(`I_${r}_${c}`, "input", v));
              }}
              onClickCell={(r, c, v) => onSelectNeuron(makeNeuron(`I_${r}_${c}`, "input", v))}
            />
            <div className="text-[11px] text-slate-400 mt-2 font-mono">Input: 28x28x1</div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs text-slate-300 font-mono">Stage 2: Convolution Operation (Conv1)</div>
              <div className="flex items-center gap-2">
                <button onClick={() => setPlayingConv((p) => !p)} className="text-xs px-2 py-1 rounded border border-cyan-700 bg-cyan-900/30 text-cyan-300">
                  {playingConv ? "Pause" : "Play"}
                </button>
                <select value={speed} onChange={(e) => setSpeed(e.target.value as Speed)} className="text-xs bg-slate-900 border border-slate-700 rounded px-2 py-1">
                  <option value="slow">Slow</option>
                  <option value="medium">Medium</option>
                  <option value="fast">Fast</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-[170px_110px_170px] gap-3 items-start">
              <HeatmapCanvas
                data={input}
                size={28}
                cell={5}
                color="gray"
                className="rounded border border-slate-700"
                highlightPatch={{ r: convPos.r, c: convPos.c, h: 3, w: 3 }}
              />

              <div className="rounded border border-slate-700 p-2 bg-slate-950">
                <div className="text-[10px] text-slate-400 mb-1">3x3 Kernel F{selectedFilter + 1}</div>
                <div className="grid grid-cols-3 gap-1">
                  {currentKernel.map((w, i) => (
                    <div key={i} className="w-7 h-7 rounded bg-slate-800 border border-slate-700 text-[10px] text-cyan-300 flex items-center justify-center">
                      {w.toFixed(2)}
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-[10px] text-slate-500">z(i,j)=sum(K*I)+b</div>
                <div className="mt-2 grid grid-cols-4 gap-1">
                  {Array.from({ length: Math.min(8, conv1.length || 8) }, (_, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedFilter(i)}
                      className={`h-5 rounded border text-[9px] ${i === selectedFilter ? "border-cyan-400 bg-cyan-900/40 text-cyan-300" : "border-slate-700 bg-slate-900 text-slate-400"}`}
                    >
                      F{i + 1}
                    </button>
                  ))}
                </div>
              </div>

              <HeatmapCanvas
                data={conv1[selectedFilter] ?? Array.from({ length: 26 * 26 }, () => 0)}
                size={26}
                cell={5}
                color="inferno"
                className="rounded border border-slate-700"
                highlight={convPos}
                onHoverCell={(r, c, v, ev) => onHoverNeuron(makeNeuron(`C1_${selectedFilter}_${r}_${c}`, "conv", v), ev)}
                onClickCell={(r, c, v) => onSelectNeuron(makeNeuron(`C1_${selectedFilter}_${r}_${c}`, "conv", v, convNeuronAtStep.incomingEdges))}
              />
            </div>

            <div className="mt-2 text-[11px] font-mono text-slate-400">
              Step ({convPos.r},{convPos.c}) z = {convNeuronAtStep.incomingEdges.map((e) => `${e.weight.toFixed(2)}*${(e.contribution / (e.weight || 1)).toFixed(2)}`).join(" + ")} = {convNeuronAtStep.activation.toFixed(4)}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 3: Feature Map Gallery (Conv1)</div>
            <div className="grid grid-cols-2 gap-2">
              {(conv1.length ? conv1 : [Array.from({ length: 26 * 26 }, () => 0)]).slice(0, 8).map((m, i) => (
                <div
                  key={i}
                  className={`rounded border ${selectedFilter === i ? "border-cyan-400" : "border-slate-700"} p-1 transition-transform hover:scale-[1.05]`}
                  onClick={() => {
                    setSelectedFilter(i);
                    onSelectNeuron(makeNeuron(`C1_filter_${conv1Ranking[i] ?? i}`, "conv", m.reduce((a, b) => a + b, 0) / m.length));
                  }}
                >
                  <HeatmapCanvas data={m} size={26} cell={2} color="viridis" className="rounded" />
                  <div className="text-[10px] text-slate-400 mt-1 font-mono">Filter {i + 1}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 4-5: ReLU + Pooling</div>
            <div className="grid grid-cols-[1fr_24px_1fr] gap-2 items-center">
              <HeatmapCanvas data={conv1Raw[selectedFilter] ?? Array.from({ length: 26 * 26 }, () => 0)} size={26} cell={3} color="signed" className="rounded border border-slate-700" />
              <svg viewBox="0 0 24 72" className="w-6 h-20">
                <polyline points="2,60 10,60 20,10" fill="none" stroke="#22d3ee" strokeWidth="2" />
              </svg>
              <HeatmapCanvas data={conv1[selectedFilter] ?? Array.from({ length: 26 * 26 }, () => 0)} size={26} cell={3} color="inferno" className="rounded border border-slate-700" highlightPatch={{ r: poolPos.r * 2, c: poolPos.c * 2, h: 2, w: 2 }} />
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <HeatmapCanvas
                data={conv1[selectedFilter] ?? Array.from({ length: 26 * 26 }, () => 0)}
                size={26}
                cell={3}
                color="viridis"
                className="rounded border border-slate-700"
                highlightPatch={{ r: poolPos.r * 2, c: poolPos.c * 2, h: 2, w: 2 }}
              />
              <HeatmapCanvas
                data={pool1[selectedFilter] ?? Array.from({ length: 13 * 13 }, () => 0)}
                size={13}
                cell={6}
                color="inferno"
                className="rounded border border-slate-700"
                highlight={poolPos}
                onHoverCell={(r, c, v, ev) => onHoverNeuron(makeNeuron(`P1_${selectedFilter}_${r}_${c}`, "pool", v, poolNeuronAtStep.incomingEdges), ev)}
                onClickCell={(r, c, v) => onSelectNeuron(makeNeuron(`P1_${selectedFilter}_${r}_${c}`, "pool", v, poolNeuronAtStep.incomingEdges))}
              />
            </div>
            <div className="text-[11px] text-slate-400 mt-2 font-mono">p(i,j)=max(x2i,2j, x2i+1,2j, x2i,2j+1, x2i+1,2j+1)</div>
          </div>
        </div>

        <div className="grid grid-cols-[360px_360px_1fr] gap-4">
          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 6: Conv2 Feature Maps (16 x 11x11)</div>
            <div className="grid grid-cols-4 gap-1.5">
              {(conv2.length ? conv2 : [Array.from({ length: 11 * 11 }, () => 0)]).slice(0, 16).map((m, f) => (
                <div key={f} className="rounded border border-slate-700 p-1">
                  <HeatmapCanvas data={m} size={11} cell={3} color="inferno" className="rounded" />
                  <div className="text-[9px] text-slate-500 mt-1">F{f + 1}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 7: Pool2 (5x5x16) -&gt; Flatten 400</div>
            <div className="grid grid-cols-4 gap-1.5">
              {(pool2.length ? pool2 : [Array.from({ length: 5 * 5 }, () => 0)]).slice(0, 16).map((m, f) => (
                <div key={f} className="rounded border border-slate-700 p-1">
                  <HeatmapCanvas
                    data={m}
                    size={5}
                    cell={8}
                    color="viridis"
                    className="rounded"
                    onClickCell={(r, c, v, ev) => {
                      const n = makeNeuron(`P2_${f}_${r}_${c}`, "pool", v);
                      onSelectNeuron(n);
                      if (showRF) setRfPatch(rfForPool2(r, c));
                      onHoverPosition(ev.clientX, ev.clientY);
                    }}
                  />
                  <div className="text-[9px] text-slate-500 mt-1">P2-{f + 1}</div>
                </div>
              ))}
            </div>
            <div className="mt-2 text-[11px] text-slate-400 font-mono">Flatten: 5x5x16 -&gt; 400x1</div>
            <div className="mt-1 h-10 rounded border border-slate-700 bg-slate-950 overflow-hidden">
              <svg viewBox="0 0 400 24" className="w-full h-full">
                {flatten.slice(0, 400).map((v, i) => (
                  <line key={i} x1={i} y1={24} x2={i} y2={24 - clamp01(v) * 22} stroke={`rgba(34,211,238,${0.22 + clamp01(v) * 0.78})`} strokeWidth={1} />
                ))}
              </svg>
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Dense 128 -&gt; Output Softmax</div>
            <div className="grid grid-cols-[1fr_220px] gap-4">
              <svg viewBox="0 0 360 260" className="w-full h-[260px] rounded border border-slate-700 bg-slate-950/60">
                {dense.slice(0, 24).map((a, i) => {
                  const y = 16 + i * 10;
                  const r = 4 + clamp01(a) * 5;
                  return <circle key={i} cx={70} cy={y} r={r} fill={`rgba(34,211,238,${0.25 + clamp01(a) * 0.75})`} stroke="#334155" />;
                })}
                {probs.map((p, i) => {
                  const y = 20 + i * 22;
                  return (
                    <g key={i}>
                      <circle
                        cx={260}
                        cy={y}
                        r={6 + p * 8}
                        fill={i === predicted ? "rgba(52,211,153,0.95)" : "rgba(59,130,246,0.85)"}
                        stroke="#334155"
                        onMouseMove={(ev) => {
                          onHoverPosition(ev.clientX, ev.clientY);
                          onHoverNeuronData(makeNeuron(`O_${i}`, "dense", p));
                        }}
                        onMouseLeave={() => onHoverNeuronData(null)}
                        onClick={() => onSelectNeuron(makeNeuron(`O_${i}`, "dense", p))}
                      />
                      <text x={275} y={y + 3} fontSize="9" fill="#cbd5e1" fontFamily="monospace">{i}</text>
                    </g>
                  );
                })}
                <text x={18} y={248} fontSize="10" fill="#64748b" fontFamily="monospace">Dense shown (24/128)</text>
              </svg>

              <div className="space-y-1">
                {probs.map((p, i) => (
                  <div key={i} className="group" onMouseMove={(ev) => onHoverPosition(ev.clientX, ev.clientY)}>
                    <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
                      <span className="w-4">{i}</span>
                      <div className="flex-1 h-2 rounded bg-slate-800 overflow-hidden">
                        <div className={`h-full transition-all duration-500 ${i === predicted ? "bg-emerald-400" : "bg-cyan-400"}`} style={{ width: `${Math.max(2, p * 100)}%` }} />
                      </div>
                      <span className="w-10 text-right">{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div className="hidden group-hover:block text-[10px] text-cyan-300 ml-6">P(digit={i}) = {p.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="text-[11px] text-slate-500 font-mono">
          Stage 8: Receptive Field Highlighter traces clicked Pool2 cell back to input (~10x10 region). Current mode: {mode}.
        </div>
      </div>
    </div>
  );
}


