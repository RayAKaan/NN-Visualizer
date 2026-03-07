import React, { useMemo } from "react";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { EdgeState, NeuronState } from "./types";

interface NodePos {
  neuron: NeuronState;
  x: number;
  y: number;
}

interface Props {
  onHoverPosition: (x: number, y: number) => void;
}

const LAYERS = ["input", "hidden", "output"] as const;

const bucketOf = (id: string, layerType: NeuronState["layerType"]) => {
  if (layerType === "input" || id.startsWith("I_")) return "input";
  if (id.startsWith("O_")) return "output";
  return "hidden";
};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export default function NetworkGraph({ onHoverPosition }: Props) {
  const mode = useNeurofluxStore((s) => s.mode);
  const topology = useNeurofluxStore((s) => s.topology);
  const selectedNeuronId = useNeurofluxStore((s) => s.selectedNeuronId);
  const showNeuronHealth = useNeurofluxStore((s) => s.showNeuronHealth);
  const setHoveredNeuron = useNeurofluxStore((s) => s.setHoveredNeuron);
  const setSelectedNeuron = useNeurofluxStore((s) => s.setSelectedNeuron);

  const { nodePositions, nodeById, criticalNeuronIds } = useMemo(() => {
    const width = 900;
    const height = 560;
    const sidePad = 110;
    const topPad = 70;
    const bottomPad = 50;
    const usableHeight = height - topPad - bottomPad;

    const grouped: Record<(typeof LAYERS)[number], NeuronState[]> = {
      input: [],
      hidden: [],
      output: [],
    };

    topology.neurons.forEach((n) => {
      grouped[bucketOf(n.id, n.layerType)].push(n);
    });

    const importanceById = new Map<string, number>();
    topology.neurons.forEach((n) => {
      const score = n.outgoingEdges.reduce((sum, e) => sum + Math.abs(e.contribution), 0);
      importanceById.set(n.id, score);
    });

    const sortedImportance = Array.from(importanceById.values()).sort((a, b) => b - a);
    const topCount = Math.max(1, Math.ceil(sortedImportance.length * 0.1));
    const threshold = sortedImportance[topCount - 1] ?? Number.POSITIVE_INFINITY;
    const criticalIds = new Set<string>();
    importanceById.forEach((score, id) => {
      if (score >= threshold && score > 0) criticalIds.add(id);
    });

    const positions: NodePos[] = [];
    const byId = new Map<string, NodePos>();
    LAYERS.forEach((layer, layerIndex) => {
      const arr = grouped[layer];
      const x = sidePad + (layerIndex / (LAYERS.length - 1)) * (width - sidePad * 2);
      arr.forEach((neuron, i) => {
        const y = arr.length === 1 ? topPad + usableHeight / 2 : topPad + (i / (arr.length - 1)) * usableHeight;
        const node = { neuron, x, y };
        positions.push(node);
        byId.set(neuron.id, node);
      });
    });

    return { nodePositions: positions, nodeById: byId, criticalNeuronIds: criticalIds };
  }, [topology.neurons]);

  const edgeStyle = (e: EdgeState) => {
    if (mode === "prediction") {
      const strength = clamp01(Math.abs(e.contribution) / 1.2);
      const positive = e.contribution >= 0;
      return {
        stroke: positive ? "rgba(34,211,238,0.95)" : "rgba(244,114,182,0.9)",
        opacity: 0.15 + strength * 0.75,
        width: 0.7 + strength * 2.2,
        speed: (1.8 - strength * 1.35).toFixed(2),
        dash: `${6 + strength * 8} ${8 - strength * 4}`,
      };
    }
    const g = clamp01(Math.abs(e.gradient) / 0.04);
    const positive = e.gradient >= 0;
    return {
      stroke: positive ? "rgba(59,130,246,0.95)" : "rgba(239,68,68,0.95)",
      opacity: 0.12 + g * 0.82,
      width: 0.7 + g * 2.2,
      speed: (1.7 - g * 1.25).toFixed(2),
      dash: `${6 + g * 8} ${8 - g * 4}`,
    };
  };

  return (
    <div className="h-full w-full rounded-xl border border-slate-700 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-3">
      <svg viewBox="0 0 900 560" className="w-full h-full">
        <defs>
          <filter id="selected-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="critical-glow" x="-120%" y="-120%" width="360%" height="360%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feColorMatrix
              in="blur"
              type="matrix"
              values="0 0 0 0 0.12 0 0 0 0 0.95 0 0 0 0 0.85 0 0 0 1 0"
            />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <text x="24" y="32" fill="#94a3b8" fontSize="12">
          Layer 1: Interactive Network Graph ({mode})
        </text>

        {topology.edges.map((e) => {
          const from = nodeById.get(e.from);
          const to = nodeById.get(e.to);
          if (!from || !to) return null;
          const style = edgeStyle(e);
          return (
            <line
              key={e.id}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={style.stroke}
              strokeOpacity={style.opacity}
              strokeWidth={style.width}
              strokeDasharray={style.dash}
              strokeLinecap="round"
              className={mode === "prediction" ? "flux-edge flux-edge-forward" : "flux-edge flux-edge-backward"}
              style={{
                ["--flux-speed" as string]: `${style.speed}s`,
              }}
            />
          );
        })}

        {nodePositions.map((n) => {
          const selected = selectedNeuronId === n.neuron.id;
          const activeValue = mode === "prediction" ? n.neuron.activation : Math.abs(n.neuron.gradient) * 20;
          const glow = clamp01(activeValue);
          const isDead = showNeuronHealth && Math.abs(n.neuron.activation) < 1e-9;
          const isCritical = showNeuronHealth && criticalNeuronIds.has(n.neuron.id);

          return (
            <g
              key={n.neuron.id}
              style={{
                opacity: isDead ? 0.2 : 1,
                filter: selected ? "url(#selected-glow)" : isCritical ? "url(#critical-glow)" : undefined,
              }}
            >
              <circle
                cx={n.x}
                cy={n.y}
                r={selected ? 13 : 10}
                fill={
                  isDead
                    ? "rgba(148,163,184,0.35)"
                    : `rgba(34,211,238,${0.22 + glow * 0.62})`
                }
                stroke={selected ? "#a5f3fc" : isDead ? "#64748b" : "#334155"}
                strokeWidth={selected ? 2.8 : 1.4}
                className="cursor-pointer transition-all"
                onMouseEnter={(ev) => {
                  setHoveredNeuron(n.neuron.id);
                  onHoverPosition(ev.clientX, ev.clientY);
                }}
                onMouseMove={(ev) => onHoverPosition(ev.clientX, ev.clientY)}
                onMouseLeave={() => setHoveredNeuron(null)}
                onClick={() => setSelectedNeuron(n.neuron.id)}
              />
              <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#e2e8f0" fontSize="9">
                {n.neuron.id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
