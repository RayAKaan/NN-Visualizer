import React, { useEffect, useMemo, useState } from "react";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { EdgeState, IntrospectionMode, NeuronState } from "./types";

interface Props {
  onHoverPosition: (x: number, y: number) => void;
  onHoverNeuronData: (neuron: NeuronState | null) => void;
  onSelectNeuronData: (neuron: NeuronState | null) => void;
  inputPixels?: number[] | null;
}

interface VisibleNode {
  id: string;
  layer: "input" | "hidden1" | "hidden2" | "output";
  x: number;
  y: number;
  neuron: NeuronState | null;
  kind: "strip" | "neuron";
  label?: string;
}

const W = 1220;
const H = 760;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const relu = (v: number) => (v > 0 ? v : 0);

const hash01 = (key: string) => {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
  }
  return ((h >>> 0) % 10000) / 10000;
};

const weightFn = (a: string, b: string) => {
  const u = hash01(`${a}:${b}`);
  return (u * 2 - 1) * 1.2;
};

const softmax = (vals: number[]) => {
  const mx = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / (sum || 1));
};

const generateDigitPixels = (digit: number) => {
  const pixels = Array.from({ length: 784 }, (_, idx) => {
    const r = Math.floor(idx / 28);
    const c = idx % 28;
    const cx = 14 + 8 * Math.sin((digit + 1) * 0.7);
    const cy = 14 + 7 * Math.cos((digit + 2) * 0.55);
    const d = Math.hypot(c - cx, r - cy);
    const ring = Math.exp(-(d * d) / (2 * (5 + (digit % 3)) ** 2));
    const stroke = Math.exp(-Math.abs((c - r) - (digit - 4)) / 5.5) * 0.25;
    const grain = (hash01(`${digit}:${r}:${c}`) - 0.5) * 0.08;
    return clamp01(ring + stroke + grain);
  });
  return pixels;
};

const neuronFill = (a: number) => {
  const v = clamp01(a);
  const r = Math.round(8 + v * 45);
  const g = Math.round(22 + v * 205);
  const b = Math.round(64 + v * 255);
  return `rgb(${r}, ${g}, ${b})`;
};

const edgeColor = (w: number) => {
  if (Math.abs(w) < 0.05) return "rgba(148,163,184,0.12)";
  return w >= 0 ? "rgba(59,130,246,0.85)" : "rgba(239,68,68,0.85)";
};

function InputCanvasStage({
  pixels,
  heatmap,
  showHeatmap,
  pulseStep,
  onToggleHeatmap,
}: {
  pixels: number[];
  heatmap: number[];
  showHeatmap: boolean;
  pulseStep: number;
  onToggleHeatmap: () => void;
}) {
  const [hovered, setHovered] = useState<{ row: number; col: number; v: number } | null>(null);

  return (
    <div className="absolute left-3 top-3 w-[272px] rounded-xl border border-slate-700 bg-[#0a0a0f]/95 nv-floating-surface p-3 backdrop-blur">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-mono text-slate-200">Stage 1: Input Canvas</div>
        <button
          onClick={onToggleHeatmap}
          className={`text-[10px] px-2 py-1 rounded border ${showHeatmap ? "bg-cyan-900/40 border-cyan-500 text-cyan-300" : "bg-slate-900 border-slate-700 text-slate-300"}`}
        >
          Show Contribution Heatmap
        </button>
      </div>

      <div className={`relative grid grid-cols-28 gap-[1px] p-1 rounded bg-black ${pulseStep >= 1 ? "ann-input-pulse" : ""}`}>
        {pixels.map((p, idx) => {
          const row = Math.floor(idx / 28);
          const col = idx % 28;
          const base = Math.round(p * 255);
          const h = heatmap[idx] ?? 0;
          const hv = clamp01(Math.abs(h));
          const heat = h >= 0 ? `rgba(239,68,68,${hv * 0.75})` : `rgba(59,130,246,${hv * 0.75})`;
          return (
            <div
              key={idx}
              className="w-[8px] h-[8px]"
              style={{ backgroundColor: showHeatmap ? heat : `rgb(${base},${base},${base})` }}
              onMouseEnter={() => setHovered({ row, col, v: p })}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}
      </div>

      {hovered && (
        <div className="mt-2 text-[10px] font-mono text-cyan-300">
          Pixel[{hovered.row}][{hovered.col}] = {hovered.v.toFixed(2)}
        </div>
      )}

      <div className="mt-3 text-[10px] text-slate-400 font-mono">Flatten: 28x28 -&gt; 784x1</div>
      <div className="relative mt-1 h-8 rounded bg-slate-900 border border-slate-800 overflow-hidden">
        <svg viewBox="0 0 784 20" className="w-full h-full">
          {pixels.map((p, i) => (
            <line
              key={i}
              x1={i}
              y1={20}
              x2={i}
              y2={20 - p * 19}
              stroke={`rgba(34,211,238,${0.2 + p * 0.8})`}
              strokeWidth={0.8}
            />
          ))}
        </svg>
      </div>

      <div className="mt-2 h-4 flex items-center gap-2 text-[10px] text-slate-400">
        <span className="ann-flow-dot" />
        <span>flatten flow</span>
      </div>
    </div>
  );
}

export default function ANNVisualizer({ onHoverPosition, onHoverNeuronData, onSelectNeuronData, inputPixels }: Props) {
  const mode = useNeurofluxStore((s) => s.mode);
  const topology = useNeurofluxStore((s) => s.topology);
  const selectedNeuronId = useNeurofluxStore((s) => s.selectedNeuronId);
  const hoveredNeuronId = useNeurofluxStore((s) => s.hoveredNeuronId);
  const setSelectedNeuron = useNeurofluxStore((s) => s.setSelectedNeuron);
  const setHoveredNeuron = useNeurofluxStore((s) => s.setHoveredNeuron);

  const [expandedInput, setExpandedInput] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [lockedNeuronId, setLockedNeuronId] = useState<string | null>(null);
  const [pulseStep, setPulseStep] = useState(0);

  const ann = useMemo(() => {
    const outputFromTopology = topology.neurons.filter((n) => /^O_\d+$/.test(n.id));
    const guessedDigit = outputFromTopology.length > 0
      ? Number((outputFromTopology.sort((a, b) => b.activation - a.activation)[0]?.id.split("_")[1] ?? "7"))
      : 7;

    const pixels = Array.isArray(inputPixels) && inputPixels.length === 784 ? inputPixels : generateDigitPixels(Number.isFinite(guessedDigit) ? guessedDigit : 7);

    const inputNeurons: NeuronState[] = Array.from({ length: 784 }, (_, i) => ({
      id: `I_${i}`,
      layerType: "input",
      activation: pixels[i],
      bias: 0,
      gradient: 0,
      incomingEdges: [],
      outgoingEdges: [],
    }));

    const h1: NeuronState[] = Array.from({ length: 128 }, (_, j) => {
      let z = 0;
      const incoming: EdgeState[] = [];
      for (let k = 0; k < 16; k++) {
        const i = (j * 31 + k * 47) % 784;
        const w = weightFn(`I_${i}`, `H1_${j}`);
        const c = w * pixels[i];
        incoming.push({ id: `I_${i}_to_H1_${j}`, from: `I_${i}`, to: `H1_${j}`, weight: w, gradient: 0, contribution: c });
        z += c;
      }
      const a = relu(z * 0.55 + (hash01(`b1:${j}`) - 0.5) * 0.2);
      return { id: `H1_${j}`, layerType: "dense", activation: clamp01(a), bias: 0, gradient: (hash01(`g1:${j}`) - 0.5) * 0.03, incomingEdges: incoming, outgoingEdges: [] };
    });

    const h2: NeuronState[] = Array.from({ length: 64 }, (_, j) => {
      let z = 0;
      const incoming: EdgeState[] = [];
      for (let k = 0; k < 12; k++) {
        const i = (j * 13 + k * 11) % 128;
        const w = weightFn(`H1_${i}`, `H2_${j}`);
        const c = w * h1[i].activation;
        incoming.push({ id: `H1_${i}_to_H2_${j}`, from: `H1_${i}`, to: `H2_${j}`, weight: w, gradient: 0, contribution: c });
        z += c;
      }
      const a = relu(z * 0.8 + (hash01(`b2:${j}`) - 0.5) * 0.3);
      return { id: `H2_${j}`, layerType: "dense", activation: clamp01(a), bias: 0, gradient: (hash01(`g2:${j}`) - 0.5) * 0.03, incomingEdges: incoming, outgoingEdges: [] };
    });

    const logits = Array.from({ length: 10 }, (_, j) => {
      let z = 0;
      for (let k = 0; k < 10; k++) {
        const i = (j * 7 + k * 5) % 64;
        z += h2[i].activation * weightFn(`H2_${i}`, `O_${j}`);
      }
      return z;
    });
    const probs = softmax(logits);

    const out: NeuronState[] = Array.from({ length: 10 }, (_, j) => {
      const incoming: EdgeState[] = [];
      for (let k = 0; k < 10; k++) {
        const i = (j * 7 + k * 5) % 64;
        const w = weightFn(`H2_${i}`, `O_${j}`);
        incoming.push({ id: `H2_${i}_to_O_${j}`, from: `H2_${i}`, to: `O_${j}`, weight: w, gradient: (hash01(`go:${i}:${j}`) - 0.5) * 0.03, contribution: w * h2[i].activation });
      }
      return {
        id: `O_${j}`,
        layerType: "dense",
        activation: probs[j],
        bias: 0,
        gradient: (hash01(`gout:${j}`) - 0.5) * 0.04,
        incomingEdges: incoming,
        outgoingEdges: [],
      };
    });

    const byId = new Map<string, NeuronState>([...inputNeurons, ...h1, ...h2, ...out].map((n) => [n.id, n]));
    [...h1, ...h2, ...out].forEach((n) => {
      n.incomingEdges.forEach((e) => {
        const src = byId.get(e.from);
        if (src) src.outgoingEdges.push(e);
      });
    });

    const predicted = probs.indexOf(Math.max(...probs));

    return { pixels, inputNeurons, h1, h2, out, byId, predicted, probs };
  }, [topology, inputPixels]);

  useEffect(() => {
    setPulseStep(1);
    const t1 = window.setTimeout(() => setPulseStep(2), 300);
    const t2 = window.setTimeout(() => setPulseStep(3), 600);
    const t3 = window.setTimeout(() => setPulseStep(4), 900);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
    };
  }, [mode, ann.predicted, ann.probs]);

  const contributionHeatmap = useMemo(() => {
    const out = ann.predicted;
    const topH2 = [...ann.h2].sort((a, b) => b.activation - a.activation).slice(0, 12);
    const topH1 = [...ann.h1].sort((a, b) => b.activation - a.activation).slice(0, 16);

    return ann.pixels.map((p, i) => {
      let sum = 0;
      for (const h1 of topH1) {
        const w1 = weightFn(`I_${i}`, h1.id);
        let tail = 0;
        for (const h2 of topH2) {
          const w12 = weightFn(h1.id, h2.id);
          const w2o = weightFn(h2.id, `O_${out}`);
          tail += w12 * w2o * h2.activation;
        }
        sum += w1 * tail * h1.activation;
      }
      return clamp01(sum * p * 0.18);
    });
  }, [ann]);

  const visible = useMemo(() => {
    const h1Top = [...ann.h1].sort((a, b) => b.activation - a.activation).slice(0, 16);
    const h2Top = [...ann.h2].sort((a, b) => b.activation - a.activation).slice(0, 12);

    const nodes: VisibleNode[] = [];
    const addColumn = (arr: NeuronState[], layer: VisibleNode["layer"], x: number, y0: number, y1: number) => {
      arr.forEach((n, i) => {
        const y = arr.length === 1 ? (y0 + y1) / 2 : y0 + (i / (arr.length - 1)) * (y1 - y0);
        nodes.push({ id: n.id, layer, x, y, neuron: n, kind: "neuron", label: layer === "output" ? n.id.split("_")[1] : undefined });
      });
    };

    if (expandedInput) {
      const first20 = ann.inputNeurons.slice(0, 20);
      addColumn(first20, "input", 370, 110, 620);
    } else {
      nodes.push({ id: "INPUT_STRIP", layer: "input", x: 370, y: 365, neuron: null, kind: "strip", label: "784 neurons" });
    }

    addColumn(h1Top, "hidden1", 585, 120, 610);
    addColumn(h2Top, "hidden2", 800, 140, 590);
    addColumn(ann.out, "output", 1015, 150, 580);

    const pos = new Map(nodes.map((n) => [n.id, n]));
    return { nodes, pos };
  }, [ann, expandedInput]);

  const focusId = lockedNeuronId ?? hoveredNeuronId ?? selectedNeuronId;

  const edgesToDraw = useMemo(() => {
    if (!focusId || focusId === "INPUT_STRIP") return [] as Array<{ edge: EdgeState; from: VisibleNode; to: VisibleNode; dashed: boolean }>;
    const focus = ann.byId.get(focusId);
    if (!focus) return [];

    const incoming = focus.incomingEdges
      .map((e) => {
        const src = visible.pos.get(e.from);
        const to = visible.pos.get(focusId);
        if (!to) return null;
        if (src) return { edge: e, from: src, to, dashed: false };
        if (!expandedInput && e.from.startsWith("I_")) {
          const strip = visible.pos.get("INPUT_STRIP");
          if (strip) return { edge: e, from: strip, to, dashed: false };
        }
        return null;
      })
      .filter(Boolean) as Array<{ edge: EdgeState; from: VisibleNode; to: VisibleNode; dashed: boolean }>;

    if (!lockedNeuronId) return incoming;

    const outgoing = focus.outgoingEdges
      .map((e) => {
        const from = visible.pos.get(focusId);
        const to = visible.pos.get(e.to);
        return from && to ? { edge: e, from, to, dashed: true } : null;
      })
      .filter(Boolean) as Array<{ edge: EdgeState; from: VisibleNode; to: VisibleNode; dashed: boolean }>;

    return [...incoming, ...outgoing];
  }, [ann.byId, expandedInput, focusId, lockedNeuronId, visible.pos]);

  const handleHoverNeuron = (neuron: NeuronState | null, ev: React.MouseEvent) => {
    if (!neuron) return;
    setHoveredNeuron(neuron.id);
    onHoverPosition(ev.clientX, ev.clientY);
    onHoverNeuronData(neuron);
  };

  const handleLeaveNeuron = () => {
    setHoveredNeuron(null);
    onHoverNeuronData(null);
  };

  const handleSelectNeuron = (neuron: NeuronState) => {
    setSelectedNeuron(neuron.id);
    setLockedNeuronId((prev) => (prev === neuron.id ? null : neuron.id));
    onSelectNeuronData(neuron);
  };

  return (
    <div className="h-full w-full rounded-xl border border-slate-700 bg-[#0a0a0f] nv-viz-surface relative overflow-hidden">
      <InputCanvasStage
        pixels={ann.pixels}
        heatmap={contributionHeatmap}
        showHeatmap={showHeatmap}
        pulseStep={pulseStep}
        onToggleHeatmap={() => setShowHeatmap((v) => !v)}
      />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        <defs>
          <filter id="ann-neuron-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <text x="340" y="42" fill="#e2e8f0" fontFamily="monospace" fontSize="13">Input (784)</text>
        <text x="546" y="42" fill="#e2e8f0" fontFamily="monospace" fontSize="13">Hidden 1 (128)</text>
        <text x="760" y="42" fill="#e2e8f0" fontFamily="monospace" fontSize="13">Hidden 2 (64)</text>
        <text x="985" y="42" fill="#e2e8f0" fontFamily="monospace" fontSize="13">Output (10)</text>

        {edgesToDraw.map(({ edge, from, to, dashed }, i) => {
          const wAbs = Math.abs(edge.weight);
          const cAbs = Math.abs(edge.contribution);
          const sw = 0.5 + Math.min(3.5, wAbs * 2.5);
          const op = 0.15 + Math.min(0.85, cAbs * 0.9);
          return (
            <line
              key={`${edge.id}-${i}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={edgeColor(edge.weight)}
              strokeWidth={sw}
              strokeOpacity={op}
              strokeDasharray={dashed ? "5 4" : undefined}
            />
          );
        })}

        {visible.nodes.map((n) => {
          if (n.kind === "strip") {
            return (
              <g key={n.id}>
                <rect
                  x={n.x - 16}
                  y={n.y - 220}
                  width={32}
                  height={440}
                  rx={9}
                  fill="url(#stripGrad)"
                  stroke="#334155"
                  strokeWidth={1.2}
                  className="cursor-pointer"
                  onClick={() => setExpandedInput((v) => !v)}
                />
                <defs>
                  <linearGradient id="stripGrad" x1="0" y1="0" x2="0" y2="1">
                    {ann.pixels.slice(0, 40).map((p, i) => (
                      <stop key={i} offset={`${(i / 39) * 100}%`} stopColor={`rgba(34,211,238,${0.15 + p * 0.7})`} />
                    ))}
                  </linearGradient>
                </defs>
                <text x={n.x + 26} y={n.y} fill="#94a3b8" fontSize="10" fontFamily="monospace">
                  784 neurons
                </text>
                <text x={n.x + 26} y={n.y + 16} fill="#64748b" fontSize="10" fontFamily="monospace">
                  click to expand
                </text>
              </g>
            );
          }

          const neuron = n.neuron!;
          const selected = selectedNeuronId === neuron.id || lockedNeuronId === neuron.id;
          const radius = 8 + clamp01(neuron.activation) * 10;
          const isWinning = neuron.id === `O_${ann.predicted}`;
          const pulseBoost =
            pulseStep >= 4 && n.layer === "output"
              ? isWinning
                ? 1
                : 0.35
              : pulseStep >= 3 && n.layer === "hidden2"
                ? 0.45
                : pulseStep >= 2 && n.layer === "hidden1"
                  ? 0.45
                  : 0;

          return (
            <g key={n.id}>
              <circle
                cx={n.x}
                cy={n.y}
                r={radius + pulseBoost * 3}
                fill={neuronFill(clamp01(neuron.activation + pulseBoost * 0.25))}
                stroke={selected ? "#22d3ee" : "#475569"}
                strokeWidth={selected ? 3 : 1.1}
                filter={selected || isWinning ? "url(#ann-neuron-glow)" : undefined}
                className="cursor-pointer transition-all duration-300"
                onMouseEnter={(ev) => handleHoverNeuron(neuron, ev)}
                onMouseMove={(ev) => onHoverPosition(ev.clientX, ev.clientY)}
                onMouseLeave={handleLeaveNeuron}
                onClick={() => handleSelectNeuron(neuron)}
              />
              <text x={n.x + radius + 4} y={n.y + 4} fill="#cbd5e1" fontFamily="monospace" fontSize="10">
                {n.layer === "output" ? n.label : neuron.id}
              </text>
            </g>
          );
        })}

        <text x="546" y="632" fill="#64748b" fontFamily="monospace" fontSize="11">Top 16 shown ... (128 total)</text>
        <text x="760" y="632" fill="#64748b" fontFamily="monospace" fontSize="11">Top 12 shown ... (64 total)</text>
      </svg>

      <div className="absolute right-3 bottom-3 w-[215px] rounded-xl border border-slate-700 bg-slate-900/90 p-3">
        <div className="text-xs text-slate-300 font-mono mb-2">Stage 5: Output Decision</div>
        <div className="space-y-1.5">
          {ann.probs.map((p, i) => (
            <div key={i} className="group">
              <div className="flex items-center gap-2 text-[10px] text-slate-300 font-mono">
                <span className="w-4">{i}</span>
                <div className="flex-1 h-2 rounded bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${ann.predicted === i ? "bg-emerald-400" : "bg-cyan-400/80"}`}
                    style={{ width: `${Math.max(2, p * 100)}%` }}
                  />
                </div>
                <span className="w-10 text-right">{(p * 100).toFixed(1)}%</span>
              </div>
              <div className="hidden group-hover:block text-[10px] text-cyan-300 font-mono ml-6">P(digit={i}) = {p.toFixed(3)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="absolute left-[318px] top-[66px] text-[10px] text-slate-500 font-mono">
        Stage 3: Hover any neuron to reveal incoming weighted connections; click to lock and show outgoing dashed edges.
      </div>

      <div className="absolute left-[318px] top-[84px] text-[10px] text-slate-500 font-mono">
        Stage 4: {mode === "prediction" ? "Forward activation pulse (left -> right)" : "Gradient pulse (right -> left)"}
      </div>
    </div>
  );
}


