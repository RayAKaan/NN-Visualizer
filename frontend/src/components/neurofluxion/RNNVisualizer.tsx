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

type CellMode = "rnn" | "lstm";

interface RNNPredictPayload {
  probabilities?: number[];
  timestep_activations?: number[];
  lstm_output?: number[] | number[][];
  dense_layers?: Record<string, number[]>;
  cell_state_summary?: {
    mean: number;
    std: number;
    max: number;
    min: number;
  };
}

const T = 10;
const D = 16;
const CELL_W = 140;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function HeatRow({ vals }: { vals: number[] }) {
  return (
    <div className="flex gap-[1px] h-4">
      {vals.map((v, i) => {
        const t = clamp01((v + 1) / 2);
        const r = Math.round(26 + t * 35);
        const g = Math.round(30 + t * 210);
        const b = Math.round(55 + (1 - t) * 155);
        return <div key={i} className="w-[4px] h-4" style={{ backgroundColor: `rgb(${r},${g},${b})` }} />;
      })}
    </div>
  );
}

function GateBox({ title, color, value }: { title: string; color: string; value: number }) {
  return (
    <div className="rounded border p-2 text-[11px] font-mono" style={{ borderColor: color, backgroundColor: `${color}22` }}>
      <div style={{ color }}>{title}</div>
      <div className="text-slate-200">{value.toFixed(3)}</div>
    </div>
  );
}

function gradColor(v: number) {
  if (v > 0.65) return "#22c55e";
  if (v > 0.35) return "#facc15";
  return "#ef4444";
}

function SequenceTimeline({
  activeT,
  seq,
  onSelect,
}: {
  activeT: number;
  seq: number[][];
  onSelect: (t: number) => void;
}) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
      <div className="text-xs text-slate-300 font-mono mb-2">Stage 1: Sequence Timeline</div>
      <div className="relative grid grid-cols-10 gap-2">
        {seq.map((x, t) => (
          <button
            key={t}
            onClick={() => onSelect(t)}
            className={`rounded border p-1 text-[10px] font-mono transition-colors ${
              activeT === t
                ? "border-cyan-400 bg-cyan-900/40 text-cyan-200 shadow-[0_0_16px_rgba(34,211,238,0.4)]"
                : "border-slate-700 bg-slate-900 text-slate-300"
            }`}
          >
            <div>t={t}</div>
            <div className="mt-1 h-5 flex items-end gap-[1px]">
              {x.map((v, i) => (
                <span key={i} className="w-[4px] bg-cyan-400/80" style={{ height: `${2 + clamp01((v + 1) / 2) * 18}px` }} />
              ))}
            </div>
          </button>
        ))}
        <div className="absolute top-[-4px] left-0 h-[2px] bg-cyan-400/80 transition-all duration-300" style={{ width: `${(activeT + 1) * 10}%` }} />
      </div>
    </div>
  );
}

function GradientFlowChart({ grads }: { grads: number[] }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
      <div className="text-xs text-slate-300 font-mono mb-2">Stage 5: Vanishing Gradient Visualizer</div>
      <div className="grid grid-cols-10 gap-2 items-end h-28">
        {grads.map((g, t) => (
          <div key={t} className="flex flex-col items-center gap-1">
            <div className="w-6 rounded-t" style={{ height: `${Math.max(3, g * 90)}px`, backgroundColor: gradColor(g) }} />
            <div className="text-[10px] text-slate-400 font-mono">t{t}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

const toSequence = (pixels: number[]) => {
  const seq: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row: number[] = [];
    for (let d = 0; d < D; d++) {
      const idx = Math.floor(((t * D + d) / (T * D)) * pixels.length);
      row.push((pixels[idx] ?? 0) * 2 - 1);
    }
    seq.push(row);
  }
  return seq;
};

const makeFallbackPixels = () =>
  Array.from({ length: 784 }, (_, i) => {
    const r = Math.floor(i / 28);
    const c = i % 28;
    const d = Math.hypot(c - 14, r - 14);
    return clamp01(Math.exp(-(d * d) / 56));
  });

const makeNeuron = (id: string, layerType: NeuronState["layerType"], activation: number, incoming: EdgeState[] = []): NeuronState => ({
  id,
  layerType,
  activation,
  bias: 0,
  gradient: 0,
  incomingEdges: incoming,
  outgoingEdges: [],
});

export default function RNNVisualizer({ onHoverPosition, onHoverNeuronData, onSelectNeuronData, inputPixels }: Props) {
  const mode = useNeurofluxStore((s) => s.mode);
  const setHoveredNeuron = useNeurofluxStore((s) => s.setHoveredNeuron);
  const setSelectedNeuron = useNeurofluxStore((s) => s.setSelectedNeuron);

  const [play, setPlay] = useState(true);
  const [activeT, setActiveT] = useState(0);
  const [selectedT, setSelectedT] = useState(3);
  const [cellMode, setCellMode] = useState<CellMode>("rnn");
  const [sampleDigit, setSampleDigit] = useState(7);
  const requestInFlightRef = useRef(false);
  const [refreshTick, setRefreshTick] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [seq, setSeq] = useState<number[][]>(toSequence(makeFallbackPixels()));
  const [hStates, setHStates] = useState<number[][]>(Array.from({ length: T }, () => Array.from({ length: 32 }, () => 0)));
  const [outputs, setOutputs] = useState<number[][]>(Array.from({ length: T }, () => Array.from({ length: 10 }, () => 0.1)));
  const [denseFinal, setDenseFinal] = useState<number[]>(Array.from({ length: 32 }, () => 0));
  const [gradNorms, setGradNorms] = useState<number[]>(Array.from({ length: T }, (_, t) => Math.exp(-(T - 1 - t) * 0.35)));

  const hasLiveInput = Array.isArray(inputPixels) && inputPixels.length === 784;
  const realtimeActive = mode === "training" || hasLiveInput;

  useEffect(() => {
    if (!realtimeActive) return;
    const id = window.setInterval(() => {
      setRefreshTick((t) => t + 1);
    }, 500);
    return () => window.clearInterval(id);
  }, [realtimeActive]);

  useEffect(() => {
    let canceled = false;
    const load = async () => {
      if (requestInFlightRef.current) return;
      requestInFlightRef.current = true;
      setLoading(true);
      setError(null);
      try {
        let pixels: number[];
        if (hasLiveInput) {
          pixels = inputPixels as number[];
        } else {
          const s = await apiClient.get("/samples");
          pixels = Array.isArray(s.data?.[String(sampleDigit)]) ? (s.data[String(sampleDigit)] as number[]) : makeFallbackPixels();
        }

        const pred = await apiClient.post<RNNPredictPayload>("/predict", { pixels, model_type: "rnn" });
        if (canceled) return;

        const data = pred.data;
        const seqData = toSequence(pixels);

        const rawLstm = data.lstm_output;
        let h: number[][] = [];
        if (Array.isArray(rawLstm) && Array.isArray(rawLstm[0])) {
          const rows = rawLstm as number[][];
          const pick = rows.length >= T ? rows.slice(0, T) : [...rows, ...Array.from({ length: T - rows.length }, () => rows[rows.length - 1] ?? Array.from({ length: 32 }, () => 0))];
          h = pick.map((r) => r.slice(0, 32));
        } else if (Array.isArray(rawLstm)) {
          const vec = (rawLstm as number[]).slice(0, 32);
          h = Array.from({ length: T }, (_, t) => vec.map((v) => v * ((t + 1) / T)));
        } else {
          h = Array.from({ length: T }, () => Array.from({ length: 32 }, () => 0));
        }

        const probs = Array.isArray(data.probabilities) ? data.probabilities : Array.from({ length: 10 }, () => 0.1);
        const perStepOut = Array.from({ length: T }, (_, t) => {
          const a = (t + 1) / T;
          const flat = probs.map((p) => p * a + (1 - a) * 0.1);
          const s2 = flat.reduce((x, y) => x + y, 0) || 1;
          return flat.map((v) => v / s2);
        });

        const denseVals = data.dense_layers ? Object.values(data.dense_layers)[0] ?? [] : [];
        const base = data.cell_state_summary?.std ?? 0.2;
        const g = Array.from({ length: T }, (_, t) => clamp01(Math.exp(-(T - 1 - t) * (0.22 + base))));

        setSeq(seqData);
        setHStates(h);
        setOutputs(perStepOut);
        setDenseFinal(denseVals.length > 0 ? denseVals.slice(0, 32) : h[h.length - 1]);
        setGradNorms(g);
      } catch {
        if (canceled) return;
        setError("Using fallback local rollout (backend RNN trace unavailable).");
      } finally {
        requestInFlightRef.current = false;
        if (!canceled) setLoading(false);
      }
    };

    void load();
    return () => {
      canceled = true;
    };
  }, [hasLiveInput, inputPixels, sampleDigit, refreshTick]);

  useEffect(() => {
    if (!play) return;
    const id = window.setInterval(() => {
      setActiveT((t) => (t + 1) % T);
    }, 450);
    return () => window.clearInterval(id);
  }, [play]);

  useEffect(() => {
    setSelectedT(activeT);
  }, [activeT]);

  const finalOut = outputs[T - 1] ?? Array.from({ length: 10 }, () => 0.1);
  const predicted = finalOut.indexOf(Math.max(...finalOut));

  const deepX = seq[selectedT] ?? Array.from({ length: D }, () => 0);
  const deepHPrev = hStates[Math.max(0, selectedT - 1)] ?? Array.from({ length: 32 }, () => 0);
  const deepH = hStates[selectedT] ?? Array.from({ length: 32 }, () => 0);
  const deepC = deepH.map((v, i) => v * 0.65 + deepHPrev[i] * 0.35);

  const pickNeuron = (id: string, activation: number, layerType: NeuronState["layerType"], incoming: EdgeState[] = []) => {
    const n: NeuronState = {
      id,
      layerType,
      activation,
      bias: 0,
      gradient: mode === "training" ? gradNorms[selectedT] * 0.05 : 0,
      incomingEdges: incoming,
      outgoingEdges: [],
    };
    setSelectedNeuron(id);
    onSelectNeuronData(n);
  };

  const hoverNeuron = (ev: React.MouseEvent, id: string, activation: number, layerType: NeuronState["layerType"]) => {
    const n: NeuronState = { id, layerType, activation, bias: 0, gradient: 0, incomingEdges: [], outgoingEdges: [] };
    setHoveredNeuron(id);
    onHoverPosition(ev.clientX, ev.clientY);
    onHoverNeuronData(n);
  };

  const clearHover = () => {
    setHoveredNeuron(null);
    onHoverNeuronData(null);
  };

  return (
    <div className="h-full w-full rounded-xl border border-slate-700 bg-[#0a0a0f] nv-viz-surface p-3 overflow-auto">
      <div className="min-w-[1500px] space-y-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-cyan-300 font-mono">RNN Temporal Lab</div>
          <div className="flex items-center gap-2">
            {!hasLiveInput && (
              <select value={sampleDigit} onChange={(e) => setSampleDigit(Number(e.target.value))} className="text-xs bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200">
                {Array.from({ length: 10 }, (_, d) => (
                  <option key={d} value={d}>Sample {d}</option>
                ))}
              </select>
            )}
            {hasLiveInput && <div className="text-xs text-cyan-300 font-mono px-2 py-1 border border-cyan-700 rounded bg-cyan-950/30">Live Lab Input</div>}
            <button onClick={() => setPlay((p) => !p)} className="text-xs px-2 py-1 rounded border border-cyan-700 bg-cyan-900/30 text-cyan-300">{play ? "Pause" : "Play"}</button>
            <div className="flex rounded border border-slate-700 overflow-hidden text-xs">
              <button className={`px-2 py-1 ${cellMode === "rnn" ? "bg-cyan-700 text-cyan-100" : "bg-slate-900 text-slate-300"}`} onClick={() => setCellMode("rnn")}>Simple RNN</button>
              <button className={`px-2 py-1 ${cellMode === "lstm" ? "bg-amber-700 text-amber-100" : "bg-slate-900 text-slate-300"}`} onClick={() => setCellMode("lstm")}>LSTM</button>
            </div>
          </div>
        </div>

        {loading && <div className="text-xs text-cyan-300 font-mono">Loading backend RNN tensors...</div>}
        {error && <div className="text-xs text-amber-300 font-mono">{error}</div>}
        {realtimeActive && <div className="text-xs text-emerald-300 font-mono">Realtime updates: ON (refresh ~500ms)</div>}

        <SequenceTimeline activeT={activeT} seq={seq} onSelect={setActiveT} />

        <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3 overflow-x-auto">
          <div className="text-xs text-slate-300 font-mono mb-2">Stage 2: Unrolled Chain (Hero)</div>
          <svg viewBox={`0 0 ${T * CELL_W + 80} 260`} className="w-full h-[260px] min-w-[1480px]">
            {cellMode === "lstm" && (
              <>
                <path d={`M 30 28 L ${T * CELL_W + 20} 28`} stroke="#f59e0b" strokeWidth="8" strokeLinecap="round" opacity="0.6" />
                <path d={`M 30 28 L ${T * CELL_W + 20} 28`} stroke="#fde68a" strokeWidth="2" strokeDasharray="8 6" className="flux-edge flux-edge-forward" style={{ ["--flux-speed" as string]: "1.4s" }} />
                <text x="36" y="18" fill="#fbbf24" fontSize="10" fontFamily="monospace">Cell State Highway</text>
              </>
            )}

            {Array.from({ length: T }, (_, t) => {
              const x = 30 + t * CELL_W;
              const hAct = hStates[t]?.[0] ?? 0;
              const outAct = outputs[t]?.[predicted] ?? 0;
              const gMag = gradNorms[t] ?? 0;
              const recOpacity = mode === "training" ? 0.2 + gMag * 0.8 : 0.8;
              return (
                <g key={t}>
                  <rect x={x} y={52} rx={14} width={118} height={160} fill="rgba(15,23,42,0.68)" stroke={t === activeT ? "#22d3ee" : "#334155"} strokeWidth={t === activeT ? 2.4 : 1.2} />

                  {t > 0 && (
                    <>
                      <line
                        x1={x - 18}
                        y1={132}
                        x2={x + 6}
                        y2={132}
                        stroke="#22d3ee"
                        strokeWidth={4}
                        strokeLinecap="round"
                        strokeOpacity={recOpacity}
                        strokeDasharray="8 5"
                        className="flux-edge flux-edge-forward"
                        style={{ ["--flux-speed" as string]: "1.0s" }}
                      />
                      <text x={x - 17} y={120} fill="#67e8f9" fontSize="9" fontFamily="monospace">W_h</text>
                    </>
                  )}

                  <circle
                    cx={x + 60}
                    cy={184}
                    r={9}
                    fill="rgba(56,189,248,0.5)"
                    stroke="#0284c7"
                    onMouseMove={(ev) => hoverNeuron(ev, `X_${t}`, seq[t][0], "input")}
                    onMouseLeave={clearHover}
                    onClick={() => pickNeuron(`X_${t}`, seq[t][0], "input")}
                  />
                  <text x={x + 54} y={201} fill="#7dd3fc" fontSize="9" fontFamily="monospace">x_t</text>

                  <circle
                    cx={x + 60}
                    cy={132}
                    r={22 + clamp01(Math.abs(hAct)) * 3}
                    fill={`rgba(34,211,238,${0.35 + clamp01(Math.abs(hAct)) * 0.55})`}
                    stroke="#22d3ee"
                    strokeWidth={1.8}
                    filter={t === activeT ? "url(#selected-glow)" : undefined}
                    onMouseMove={(ev) => hoverNeuron(ev, `${cellMode === "lstm" ? "LSTM_" : ""}H_${t}`, hAct, "recurrent")}
                    onMouseLeave={clearHover}
                    onClick={() => pickNeuron(`${cellMode === "lstm" ? "LSTM_" : ""}H_${t}`, hAct, "recurrent", [{ id: `x_to_h_${t}`, from: `X_${t}`, to: `H_${t}`, weight: 0.3, gradient: 0, contribution: 0.2 }])}
                  />
                  <text x={x + 53} y={136} fill="#ecfeff" fontSize="10" fontFamily="monospace">h_t</text>

                  <line x1={x + 60} y1={176} x2={x + 60} y2={152} stroke="#22d3ee" strokeWidth={1.8} />
                  <text x={x + 66} y={166} fill="#67e8f9" fontSize="9" fontFamily="monospace">W_x</text>

                  <circle
                    cx={x + 60}
                    cy={78}
                    r={8 + outAct * 5}
                    fill={t === T - 1 ? "rgba(52,211,153,0.9)" : "rgba(59,130,246,0.65)"}
                    stroke="#334155"
                    onMouseMove={(ev) => hoverNeuron(ev, `O_${t}`, outAct, "dense")}
                    onMouseLeave={clearHover}
                    onClick={() => pickNeuron(`O_${t}`, outAct, "dense")}
                  />
                  <text x={x + 64} y={81} fill="#cbd5e1" fontSize="9" fontFamily="monospace">o_t</text>
                  <line x1={x + 60} y1={110} x2={x + 60} y2={86} stroke="#38bdf8" strokeWidth={1.6} />
                  <text x={x + 66} y={98} fill="#67e8f9" fontSize="9" fontFamily="monospace">W_y</text>

                  <text x={x + 46} y={226} fill="#64748b" fontSize="10" fontFamily="monospace">t={t}</text>
                </g>
              );
            })}

            <defs>
              <filter id="selected-glow" x="-120%" y="-120%" width="340%" height="340%">
                <feGaussianBlur stdDeviation="5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
          </svg>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
          <div className="text-xs text-slate-300 font-mono mb-2">Stage 3: Hidden State Evolution Strip</div>
          <div className="grid grid-cols-10 gap-2">
            {hStates.map((h, t) => (
              <button
                key={t}
                onClick={() => {
                  setSelectedT(t);
                  pickNeuron(`${cellMode === "lstm" ? "LSTM_" : ""}H_${t}`, h[0], "recurrent");
                }}
                onMouseMove={(ev) => hoverNeuron(ev, `${cellMode === "lstm" ? "LSTM_" : ""}H_${t}`, h[0], "recurrent")}
                onMouseLeave={clearHover}
                className={`rounded border p-1 text-left ${selectedT === t ? "border-cyan-400" : "border-slate-700"}`}
                title={`h_${t} = [${h.slice(0, 6).map((v) => v.toFixed(2)).join(", ")}...]`}
              >
                <div className="text-[10px] text-slate-400 font-mono">h_{t}</div>
                <HeatRow vals={h} />
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-[1.2fr_1fr] gap-4">
          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 4/6: Cell Deep Dive (t={selectedT})</div>
            {cellMode === "rnn" ? (
              <div className="grid grid-cols-[120px_120px_1fr] gap-3 items-start">
                <div className="space-y-2">
                  <div className="text-[11px] text-slate-400 font-mono">x_t (1x16)</div>
                  <HeatRow vals={deepX} />
                  <div className="text-[11px] text-slate-400 font-mono">h_(t-1) (1x32)</div>
                  <HeatRow vals={deepHPrev} />
                </div>

                <div className="rounded border border-slate-700 bg-slate-950 p-2 text-[11px] font-mono text-slate-300">
                  <div>W_x x_t + W_h h_(t-1)</div>
                  <div className="mt-2 text-cyan-300">h_t = tanh(...)</div>
                  <div className="mt-2">h_t[0..6] = [{deepH.slice(0, 6).map((v) => v.toFixed(2)).join(", ")}]</div>
                </div>

                <div className="rounded border border-slate-700 bg-slate-950 p-2 text-[11px] font-mono text-slate-300">
                  <div className="text-slate-400">Flow</div>
                  <div className="mt-1">x_t -&gt; (W_x)</div>
                  <div>h_(t-1) -&gt; (W_h)</div>
                  <div>sum + b -&gt; tanh -&gt; h_t</div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <GateBox title="Forget Gate f_t" color="#ef4444" value={clamp01((deepHPrev[0] + deepX[0]) * 0.5 + 0.5)} />
                <GateBox title="Input Gate i_t" color="#22c55e" value={clamp01((deepHPrev[1] + deepX[1]) * 0.5 + 0.5)} />
                <GateBox title="Candidate C~_t" color="#a855f7" value={deepH[2]} />
                <GateBox title="Output Gate o_t" color="#3b82f6" value={clamp01((deepHPrev[3] + deepX[3]) * 0.5 + 0.5)} />
                <div className="col-span-2 rounded border border-amber-600/50 bg-amber-950/20 p-2 text-[11px] font-mono text-amber-200">
                  C_t = f_t * C_(t-1) + i_t * C~_t | h_t = o_t * tanh(C_t)
                </div>
                <div className="col-span-2 text-[11px] text-slate-400 font-mono">C_t[0..6] = [{deepC.slice(0, 6).map((v) => v.toFixed(2)).join(", ")}]</div>
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-900/65 p-3">
            <div className="text-xs text-slate-300 font-mono mb-2">Stage 7: Prediction Summary</div>
            <div className="text-[10px] text-slate-400 font-mono mb-2">Dense summary len: {denseFinal.length}</div>
            <div className="space-y-1.5">
              {finalOut.map((p, i) => (
                <div key={i} className="group" onMouseMove={(ev) => onHoverPosition(ev.clientX, ev.clientY)}>
                  <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
                    <span className="w-4">{i}</span>
                    <div className="flex-1 h-2 rounded bg-slate-800 overflow-hidden">
                      <div className={`h-full transition-all duration-500 ${i === predicted ? "bg-emerald-400" : "bg-cyan-400"}`} style={{ width: `${Math.max(2, p * 100)}%` }} />
                    </div>
                    <span className="w-10 text-right">{(p * 100).toFixed(1)}%</span>
                  </div>
                  <div className="hidden group-hover:block text-[10px] text-cyan-300 ml-6">P(class={i}) = {p.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {mode === "training" && <GradientFlowChart grads={gradNorms} />}

        {mode === "training" && (
          <div className="text-[11px] text-slate-500 font-mono">
            BPTT intuition: dL/dh_t = dL/dh_T * Product[k=t+1..T](dh_k/dh_(k-1)); dL/dW_h accumulates over all timesteps.
          </div>
        )}
      </div>
    </div>
  );
}
