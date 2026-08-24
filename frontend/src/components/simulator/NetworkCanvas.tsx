import React, { useEffect, useMemo, useRef, useState } from "react";
import { useArchitectureStore } from "../../store/architectureStore";
import { useComputationStore } from "../../store/computationStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useBackpropStore } from "../../store/backpropStore";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { useSessionStore } from "../../store/sessionStore";
import { activationColor, gradientHealthColor, neuralPalette, lerpColor } from "@/design-system/tokens/colors";
import { useReducedMotion } from "@/design-system/hooks/useReducedMotion";

// Shared geometry so the render loop and the hit-testing handler can never
// drift apart.
const CANVAS_TOP_PAD = 56;
const CANVAS_BOTTOM_PAD = 96;
const MAX_DISPLAY_NEURONS = 10;

const hexToRgba = (hex: string, alpha: number) => {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const num = parseInt(full, 16);
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
};

const neuronRadius = (count: number) => (count > 8 ? 11 : count < 4 ? 15 : 13);

export function NetworkCanvas() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(0);
  const gridOffsetRef = useRef<number>(0);
  const architectureLayers = useArchitectureStore((s) => s.layers);
  const sessionArchitecture = useSessionStore((s) => s.architecture);
  const layers = architectureLayers.length > 0 ? architectureLayers : sessionArchitecture;
  const graphData = useSessionStore((s) => s.graphData);
  const layerOutputs = useComputationStore((s) => s.layerOutputs);
  const steps = useComputationStore((s) => s.steps);
  const selectedLayer = useSimulatorStore((s) => s.selectedLayerIndex);
  const forwardPassState = useSimulatorStore((s) => s.forwardPassState);
  const currentInput = useSimulatorStore((s) => s.currentInput);
  const currentStepIndex = useSimulatorStore((s) => s.currentStepIndex);
  const autoPlay = useSimulatorStore((s) => s.autoPlay);
  const animationSpeed = useSimulatorStore((s) => s.animationSpeed);
  const backwardSteps = useBackpropStore((s) => s.backwardSteps);
  const currentBackwardStep = useBackpropStore((s) => s.currentBackwardStep);
  const gradientSummary = useBackpropStore((s) => s.gradientSummary);
  const mode = useBackpropStore((s) => s.mode);
  const isTraining = useTrainingSimStore((s) => s.isTraining);
  const reducedMotion = useReducedMotion();
  const executionStatus = useSessionStore((s) => s.executionStatus);
  const [hovered, setHovered] = useState<{
    layerIndex: number;
    neuronIndex: number;
    x: number;
    y: number;
    activation: number;
    bias: number;
  } | null>(null);

  const displayLayers = useMemo(() => {
    const archLayers = architectureLayers || [];
    const sessLayers = sessionArchitecture || [];
    const sourceLayers = archLayers.length > 0 ? archLayers : sessLayers;
    
    if (sourceLayers.length === 0) {
      return [
        { type: 'input' as const, neurons: 2 },
        { type: 'dense' as const, neurons: 4 },
        { type: 'output' as const, neurons: 1 }
      ];
    }
    
    return sourceLayers;
  }, [architectureLayers, sessionArchitecture]);
  
  const displayCounts = useMemo(() => {
    return displayLayers.map((l) => Math.min(l.neurons || 4, MAX_DISPLAY_NEURONS));
  }, [displayLayers]);

  const weightMaps = useMemo(() => {
    // One matrix per adjacent layer pair shown on screen. Prefer real weights
    // from the backend (serialized flattened to 1-D); fall back to a stable
    // pseudo-random matrix whenever a pair has no matching data, so a
    // mismatched/shorter weights payload can never break rendering.
    const seedFrom = (a: number, b: number, c: number) => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    const rand = (seed: number) => {
      let t = seed + 0x6d2b79f5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    const maps: number[][][] = [];
    for (let i = 0; i < displayCounts.length - 1; i += 1) {
      const rows = Math.min(displayCounts[i + 1] ?? 0, MAX_DISPLAY_NEURONS);
      const cols = Math.min(displayCounts[i] ?? 0, MAX_DISPLAY_NEURONS);
      const origCols = Math.max(1, displayLayers[i]?.neurons ?? cols);
      const raw = (graphData?.weights ?? [])[i] as unknown;
      const seed = seedFrom(rows, cols, i + 1);
      const mat: number[][] = [];
      for (let r = 0; r < rows; r += 1) {
        const row: number[] = [];
        for (let c = 0; c < cols; c += 1) {
          let value: number;
          if (Array.isArray(raw)) {
            if (Array.isArray((raw as number[][])[r])) {
              value = Number((raw as number[][])[r][c] ?? 0);
            } else {
              const flat = raw as number[];
              const idx = r * origCols + c;
              value = idx < flat.length ? Number(flat[idx] ?? 0) : 0;
            }
          } else {
            value = (rand(seed + r * 31 + c * 17) - 0.5) * 2;
          }
          row.push(Number.isFinite(value) ? value : 0);
        }
        mat.push(row);
      }
      maps.push(mat);
    }
    return maps;
  }, [displayCounts, displayLayers, graphData]);

  // Backend layer_outputs are keyed by weight-layer index k, meaning the
  // activations AFTER display layer k+1. Display layer 0 is the network
  // input itself, which comes from currentInput.
  const activationsByLayer = useMemo(() => {
    const result: number[][] = [currentInput ?? []];
    for (let i = 1; i < displayLayers.length; i += 1) {
      result.push(layerOutputs[String(i - 1)] ?? []);
    }
    return result;
  }, [currentInput, displayLayers, layerOutputs]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const resize = () => {
      const rect = wrapper.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = rect.width * ratio;
      canvas.height = rect.height * ratio;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const ratio = window.devicePixelRatio || 1;

    const render = (time: number) => {
      const width = canvas.width / ratio;
      const height = canvas.height / ratio;
      const isActive = executionStatus === 'complete' || executionStatus === 'running' || forwardPassState !== "idle" || autoPlay || isTraining;
      const frameBudget = isActive ? 33 : 100;
      if (time - lastFrameRef.current < frameBudget) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }
      lastFrameRef.current = time;

      ctx.save();
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, width, height);

      const gradient = ctx.createRadialGradient(width * 0.5, height * 0.4, 0, width * 0.5, height * 0.4, width * 0.7);
      gradient.addColorStop(0, neuralPalette.obsidian);
      gradient.addColorStop(1, neuralPalette.void);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      const drift = !isActive && !reducedMotion ? (time * 0.0005) % 40 : 0;
      gridOffsetRef.current = drift;
      ctx.strokeStyle = "rgba(28,25,23,0.05)";
      ctx.lineWidth = 0.5;
      for (let x = -40 + drift; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = -40 + drift; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const displayLayers = layers.length > 0 ? layers : [
        { type: 'input', neurons: 2 },
        { type: 'dense', neurons: 4 },
        { type: 'output', neurons: 1 }
      ];
      const colCount = Math.max(displayLayers.length, 3);
      const colGap = width / (colCount + 1);
      const topPad = CANVAS_TOP_PAD;
      const bottomPad = CANVAS_BOTTOM_PAD;
      const usableHeight = height - topPad - bottomPad;

      const layerPositions = displayLayers.map((layer, i) => {
        const n = displayCounts[i] ?? 0;
        if (n <= 0) return { x: colGap * (i + 1), yPositions: [], count: 0, layer };
        const spacing = usableHeight / (n + 1);
        const x = colGap * (i + 1);
        const yPositions = Array.from({ length: n }, (_, j) => topPad + spacing * (j + 1));
        return { x, yPositions, count: n, layer };
      });

      const maxWeight = 1;
      const maxActivation = 1;
      const forwardActiveLayer = forwardPassState !== "idle" ? (steps[currentStepIndex]?.layer_index ?? null) : null;
      const backwardActiveLayer = backwardSteps[currentBackwardStep]?.layer_index ?? null;

      const hoveredLayer = hovered?.layerIndex ?? -1;
      const hoveredNeuron = hovered?.neuronIndex ?? -1;

      const drawConnection = (
        startX: number,
        startY: number,
        endX: number,
        endY: number,
        weight: number,
        sourceAct: number,
        targetAct: number,
        dim: boolean,
      ) => {
        const magnitude = Math.min(Math.abs(weight) / maxWeight, 1);
        const widthLine = 0.5 + magnitude * 1.8;
        const opacity = dim ? 0.04 : 0.05 + magnitude * 0.28;
        const controlX = (startX + endX) / 2;
        const controlY = (startY + endY) / 2 + (startX > endX ? -5 : 5);
        const grad = ctx.createLinearGradient(startX, startY, endX, endY);
        const sourceColor = activationColor(sourceAct, maxActivation);
        const targetColor = activationColor(targetAct, maxActivation);
        grad.addColorStop(0, sourceColor);
        grad.addColorStop(1, targetColor);
        ctx.strokeStyle = grad;
        ctx.lineWidth = widthLine;
        ctx.globalAlpha = opacity;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.quadraticCurveTo(controlX, controlY, endX, endY);
        ctx.stroke();
        ctx.globalAlpha = 1;
      };

      let totalConnections = 0;
      for (let i = 0; i < displayCounts.length - 1; i += 1) {
        totalConnections += displayCounts[i] * displayCounts[i + 1];
      }

      const drawFlowDot = (
        startX: number,
        startY: number,
        endX: number,
        endY: number,
        weight: number,
        dir: "forward" | "backward",
        phase: number,
      ) => {
        const speed = 0.0008 * animationSpeed;
        let t = (time * speed + phase) % 1;
        if (dir === "backward") t = 1 - t;
        const tGradient = dir === "forward" ? t : 1 - t;
        const x = startX + (endX - startX) * t;
        const y = startY + (endY - startY) * t;
        const color = dir === "forward" 
          ? lerpColor(neuralPalette.dendrite.glow, neuralPalette.axon.glow, t)
          : neuralPalette.lesion.glow;
        const size = 2 + Math.min(Math.abs(weight), 1) * 2.5;
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      };

      for (let i = 0; i < displayCounts.length - 1; i += 1) {
        const source = layerPositions[i];
        const target = layerPositions[i + 1];
        const weights = weightMaps[i];
        // Dense all-to-all meshes are the main source of visual noise: hide
        // near-zero connections when there are many, keep only strong ones
        // for unselected layers in very dense graphs.
        const limitConnections = totalConnections > 200 && selectedLayer !== i;
        const pruneWeak = totalConnections > 120;
        for (let t = 0; t < target.count; t += 1) {
          for (let s = 0; s < source.count; s += 1) {
            const wMag = Math.abs(weights[t][s]);
            if ((limitConnections && wMag < 0.75) || (pruneWeak && wMag < 0.06)) continue;
            const inactiveForPass =
              (!isTraining && forwardActiveLayer != null && forwardActiveLayer !== i) ||
              (!isTraining && backwardActiveLayer != null && backwardActiveLayer !== i);
            const dim =
              (hovered && !(hoveredLayer === i && hoveredNeuron === s) && !(hoveredLayer === i + 1 && hoveredNeuron === t)) ||
              inactiveForPass;
            const sourceAct = activationsByLayer[i]?.[s] ?? 0;
            const targetAct = activationsByLayer[i + 1]?.[t] ?? 0;
            drawConnection(
              source.x,
              source.yPositions[s],
              target.x,
              target.yPositions[t],
              weights[t][s],
              sourceAct,
              targetAct,
              dim ?? false,
            );

            const isActiveForward = forwardActiveLayer === i;
            const isActiveBackward = backwardActiveLayer === i;
            if ((isTraining || isActiveForward || isActiveBackward) && !reducedMotion) {
              const dir = isActiveBackward && mode !== "forward" ? "backward" : "forward";
              const phase = (s * 17 + t * 31) * 0.003;
              drawFlowDot(source.x, source.yPositions[s], target.x, target.yPositions[t], weights[t][s], dir, phase);
            }
          }
        }
      }

      layerPositions.forEach((info, i) => {
        const outputs = activationsByLayer[i] ?? [];
        const label = i === 0 ? "Input" : i === displayLayers.length - 1 ? "Output" : `Hidden ${i}`;
        ctx.fillStyle = "#79716b";
        ctx.font = '600 10px "Inter", sans-serif';
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(`${label.toUpperCase()} · ${info.layer.neurons}`, info.x, 24);

        info.yPositions.forEach((y, j) => {
          const val = outputs[j] ?? 0;
          const intensity = Math.min(Math.abs(val) / maxActivation, 1);
          const radiusBase = neuronRadius(info.count);
          const isHovered = hoveredLayer === i && hoveredNeuron === j;
          const radius = isHovered ? radiusBase * 1.25 : radiusBase;
          const color = activationColor(val, maxActivation);
          const isDead = val === 0 && (info.layer.activation || "").toLowerCase() === "relu";

          if (intensity > 0.02) {
            const glow = ctx.createRadialGradient(info.x, y, radius * 0.6, info.x, y, radius * 2.2);
            glow.addColorStop(0, hexToRgba(color, 0.3 + intensity * 0.35));
            glow.addColorStop(1, hexToRgba(color, 0));
            ctx.beginPath();
            ctx.arc(info.x, y, radius * 2.2, 0, Math.PI * 2);
            ctx.fillStyle = glow;
            ctx.fill();
          }

          // White core + activation-colored ring: cleaner than a saturated
          // disc and keeps the number legible on top.
          ctx.beginPath();
          ctx.arc(info.x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = neuralPalette.obsidian;
          ctx.fill();

          ctx.setLineDash(isDead ? [4, 3] : []);
          ctx.strokeStyle = isDead
            ? "rgba(239, 68, 68, 0.55)"
            : i === selectedLayer
              ? neuralPalette.synapse.bright
              : intensity > 0.02
                ? color
                : "rgba(168, 162, 158, 0.85)";
          ctx.lineWidth = i === selectedLayer || isHovered ? 2 : 1.25 + intensity * 1.25;
          ctx.stroke();
          ctx.setLineDash([]);

          if ((val !== 0 || isHovered) && !isDead) {
            ctx.fillStyle = `rgba(28, 25, 23, ${0.5 + intensity * 0.45})`;
            ctx.font = '600 9px "JetBrains Mono", monospace';
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(val.toFixed(2), info.x, y);
          }

          if (isDead) {
            ctx.fillStyle = neuralPalette.lesion.bright;
            ctx.font = '600 12px "JetBrains Mono", monospace';
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("x", info.x, y);
          }
        });
      });

      for (let i = 0; i < displayLayers.length - 1; i += 1) {
        const activation = displayLayers[i + 1]?.activation;
        if (!activation) continue;
        const midX = (layerPositions[i].x + layerPositions[i + 1].x) / 2;
        const midY = height - 74;
        ctx.fillStyle = neuralPalette.obsidian;
        ctx.strokeStyle = "rgba(28,25,23,0.12)";
        ctx.lineWidth = 1;
        const text = activation.toUpperCase();
        const pad = 8;
        ctx.font = '600 9px "JetBrains Mono", monospace';
        const metrics = ctx.measureText(text);
        const w = metrics.width + pad * 2;
        const h = 18;
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(midX - w / 2, midY - h / 2, w, h, 9);
        } else {
          ctx.rect(midX - w / 2, midY - h / 2, w, h);
        }
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#44403c";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, midX, midY + 0.5);
      }

      if (gradientSummary?.per_layer?.length) {
        // Gradient bars live in their own band below the pills so the two
        // bottom overlays can never collide.
        const baseline = height - 36;
        gradientSummary.per_layer.forEach((g, idx) => {
          const midX = (layerPositions[idx]?.x + layerPositions[idx + 1]?.x) / 2;
          if (!midX) return;
          const norm = g.dW_norm ?? 0;
          const barH = Math.min(22, Math.log10(norm + 1) * 10 + 4);
          ctx.fillStyle = gradientHealthColor(norm);
          ctx.beginPath();
          if (typeof ctx.roundRect === "function") {
            ctx.roundRect(midX - 7, baseline - barH, 14, barH, [3, 3, 0, 0]);
          } else {
            ctx.rect(midX - 7, baseline - barH, 14, barH);
          }
          ctx.fill();
          ctx.fillStyle = "#79716b";
          ctx.font = '600 8px "JetBrains Mono", monospace';
          ctx.textAlign = "center";
          ctx.textBaseline = "alphabetic";
          ctx.fillText(norm.toExponential(1), midX, height - 20);
        });
      }

      ctx.restore();

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [
    layers,
    activationsByLayer,
    selectedLayer,
    forwardPassState,
    autoPlay,
    reducedMotion,
    displayCounts,
    weightMaps,
    hovered,
    steps,
    currentStepIndex,
    backwardSteps,
    currentBackwardStep,
    gradientSummary,
    mode,
    animationSpeed,
    isTraining,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const x = (event.clientX - rect.left);
      const y = (event.clientY - rect.top);
      const width = rect.width;
      const height = rect.height;
      const colGap = width / (layers.length + 1);
      const topPad = CANVAS_TOP_PAD;
      const bottomPad = CANVAS_BOTTOM_PAD;
      const usableHeight = height - topPad - bottomPad;
      let found = null as typeof hovered;
      layers.forEach((layer, i) => {
        const n = displayCounts[i];
        const spacing = usableHeight / (n + 1);
        const cx = colGap * (i + 1);
        for (let j = 0; j < n; j += 1) {
          const cy = topPad + spacing * (j + 1);
          const radiusBase = neuronRadius(n);
          const dx = x - cx;
          const dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radiusBase * 1.2) {
            const activation = activationsByLayer[i]?.[j] ?? 0;
            found = {
              layerIndex: i,
              neuronIndex: j,
              x: cx,
              y: cy,
              activation,
              bias: 0,
            };
          }
        }
      });
      setHovered(found);
    };
    const handleLeave = () => setHovered(null);
    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("mouseleave", handleLeave);
    return () => {
      canvas.removeEventListener("mousemove", handleMove);
      canvas.removeEventListener("mouseleave", handleLeave);
    };
  }, [layers, activationsByLayer, displayCounts]);

  return (
    <div ref={wrapperRef} className="network-canvas-wrap">
      <canvas ref={canvasRef} className="network-canvas" />
      {hovered ? (
        <div
          className="network-tooltip"
          style={{ left: hovered.x + 16, top: hovered.y - 12 }}
        >
          <div className="network-tooltip-title">{`Layer ${hovered.layerIndex + 1}, Neuron ${hovered.neuronIndex + 1}`}</div>
          <div className="network-tooltip-row">Activation: {hovered.activation.toFixed(3)}</div>
          <div className="network-tooltip-row">Bias: {hovered.bias.toFixed(3)}</div>
        </div>
      ) : null}
    </div>
  );
}
