import React, { useEffect, useMemo, useRef, useState } from "react";
import { LayerInfo, ModelType } from "../../types";

interface Props {
  layers: LayerInfo[];
  modelType: ModelType;
}

interface NodePoint {
  id: string;
  layerIdx: number;
  nodeIdx: number;
  x: number;
  y: number;
  activation: number;
}

const MAX_NODES_PER_LAYER = 24;
const LAYER_PADDING_X = 80;
const NODE_RADIUS = 6;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export default function NetworkView2D({ layers, modelType }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fixedCountsRef = useRef<Map<number, number>>(new Map());
  const [size, setSize] = useState({ width: 900, height: 600 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      setSize({
        width: Math.max(400, Math.floor(entry.contentRect.width)),
        height: Math.max(320, Math.floor(entry.contentRect.height)),
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const layerData = useMemo(() => {
    return layers.map((layer, layerIdx) => {
      const acts =
        Array.isArray(layer.activations)
          ? layer.activations.map((v) => (typeof v === "number" ? v : 0))
          : typeof layer.activations === "number"
            ? [layer.activations]
            : [];
      const total = Math.max(acts.length, layer.shape?.[0] ?? 0, 1);
      const existing = fixedCountsRef.current.get(layerIdx);
      const proposed = Math.min(total, MAX_NODES_PER_LAYER);
      const count = existing == null ? proposed : Math.min(MAX_NODES_PER_LAYER, Math.max(existing, proposed));
      if (existing == null || count !== existing) {
        fixedCountsRef.current.set(layerIdx, count);
      }
      const sampled = count === acts.length || acts.length === 0
        ? Array.from({ length: count }, (_, i) => acts[i] ?? 0)
        : Array.from({ length: count }, (_, i) => {
            const src = Math.floor((i / count) * acts.length);
            return acts[src] ?? 0;
          });
      return { layer, layerIdx, activations: sampled };
    });
  }, [layers]);

  const nodes = useMemo<NodePoint[]>(() => {
    if (layerData.length === 0) return [];
    const stepX =
      layerData.length > 1
        ? (size.width - LAYER_PADDING_X * 2) / (layerData.length - 1)
        : 0;

    const out: NodePoint[] = [];
    layerData.forEach(({ activations, layerIdx }) => {
      const x = LAYER_PADDING_X + layerIdx * stepX;
      const topPad = 50;
      const bottomPad = 30;
      const usable = Math.max(40, size.height - topPad - bottomPad);
      activations.forEach((act, nodeIdx) => {
        const y =
          activations.length === 1
            ? topPad + usable / 2
            : topPad + (nodeIdx / (activations.length - 1)) * usable;
        out.push({
          id: `${layerIdx}-${nodeIdx}`,
          layerIdx,
          nodeIdx,
          x,
          y,
          activation: clamp01(act),
        });
      });
    });
    return out;
  }, [layerData, size.height, size.width]);

  const animatedNodes = nodes;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = size.width;
    canvas.height = size.height;
    ctx.clearRect(0, 0, size.width, size.height);

    const byLayer = new Map<number, NodePoint[]>();
    animatedNodes.forEach((n) => {
      const list = byLayer.get(n.layerIdx) ?? [];
      list.push(n);
      byLayer.set(n.layerIdx, list);
    });

    for (let l = 0; l < layerData.length - 1; l++) {
      const src = byLayer.get(l) ?? [];
      const dst = byLayer.get(l + 1) ?? [];
      if (src.length === 0 || dst.length === 0) continue;

      src.forEach((a) => {
        dst.forEach((b) => {
          const strength = clamp01((a.activation * 0.65 + b.activation * 0.35));
          if (strength < 0.03) return;
          const midX = (a.x + b.x) / 2;
          const baseCurve = (b.x - a.x) * 0.15;
          const offsetSign = (a.nodeIdx % 2 === 0 ? 1 : -1) * (b.nodeIdx % 2 === 0 ? 1 : -1);
          const curveY = (a.y + b.y) / 2 + offsetSign * (6 + strength * 12);
          const cp1x = midX - baseCurve;
          const cp2x = midX + baseCurve;

          ctx.lineWidth = 2.5 + strength * 3.2;
          ctx.strokeStyle = `rgba(34, 211, 238, ${0.015 + strength * 0.08})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.bezierCurveTo(cp1x, curveY, cp2x, curveY, b.x, b.y);
          ctx.stroke();

          ctx.lineWidth = 0.45 + strength * 1.2;
          ctx.strokeStyle = `rgba(125, 211, 252, ${0.08 + strength * 0.5})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.bezierCurveTo(cp1x, curveY, cp2x, curveY, b.x, b.y);
          ctx.stroke();
        });
      });
    }
  }, [animatedNodes, layerData.length, size.height, size.width]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden rounded-lg bg-gradient-to-br from-slate-950 via-slate-900 to-cyan-950"
    >
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
      {animatedNodes.map((node) => {
        const glow = Math.max(3, node.activation * 18);
        const radius = NODE_RADIUS * (0.9 + node.activation * 0.6);
        return (
          <div
            key={node.id}
            className="absolute rounded-full border border-cyan-200/20 transition-all duration-150"
            style={{
              left: node.x - radius,
              top: node.y - radius,
              width: radius * 2,
              height: radius * 2,
              background: `rgba(34, 211, 238, ${0.12 + node.activation * 0.88})`,
              boxShadow: `0 0 ${glow}px rgba(34, 211, 238, ${0.35 + node.activation * 0.5})`,
            }}
            title={`Layer ${node.layerIdx + 1}, neuron ${node.nodeIdx + 1}, a=${node.activation.toFixed(3)}`}
          />
        );
      })}
      {layerData.map(({ layer, layerIdx }) => {
        const x =
          layerData.length > 1
            ? LAYER_PADDING_X +
              layerIdx * ((size.width - LAYER_PADDING_X * 2) / (layerData.length - 1))
            : size.width / 2;
        return (
          <div
            key={`${layer.name}-${layerIdx}`}
            className="absolute -translate-x-1/2 text-center"
            style={{ left: x, top: 10 }}
          >
            <div className="text-xs font-semibold text-cyan-200">{layer.name}</div>
            <div className="text-[10px] text-slate-400">{layer.type}</div>
            <div className="text-[10px] text-slate-500">{layer.shape?.join("x")}</div>
          </div>
        );
      })}
      <div className="absolute bottom-3 right-3 text-[11px] text-slate-300 bg-black/30 border border-white/10 rounded px-2 py-1">
        {modelType.toUpperCase()} activation graph
      </div>
    </div>
  );
}
