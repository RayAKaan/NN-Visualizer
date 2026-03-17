import React, { useEffect, useMemo, useRef, useState } from "react";
import { useArchitectureStore } from "../../store/architectureStore";
import { useComputationStore } from "../../store/computationStore";
import { useSimulatorStore } from "../../store/simulatorStore";
import { useBackpropStore } from "../../store/backpropStore";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { activationColor, gradientHealthColor, neuralPalette, weightColor, lerpColor } from "@/design-system/tokens/colors";
import { useReducedMotion } from "@/design-system/hooks/useReducedMotion";

export function NetworkCanvas() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(0);
  const gridOffsetRef = useRef<number>(0);
  const layers = useArchitectureStore((s) => s.layers);
  const layerOutputs = useComputationStore((s) => s.layerOutputs);
  const steps = useComputationStore((s) => s.steps);
  const selectedLayer = useSimulatorStore((s) => s.selectedLayerIndex);
  const forwardPassState = useSimulatorStore((s) => s.forwardPassState);
  const currentStepIndex = useSimulatorStore((s) => s.currentStepIndex);
  const autoPlay = useSimulatorStore((s) => s.autoPlay);
  const animationSpeed = useSimulatorStore((s) => s.animationSpeed);
  const backwardSteps = useBackpropStore((s) => s.backwardSteps);
  const currentBackwardStep = useBackpropStore((s) => s.currentBackwardStep);
  const gradientSummary = useBackpropStore((s) => s.gradientSummary);
  const mode = useBackpropStore((s) => s.mode);
  const isTraining = useTrainingSimStore((s) => s.isTraining);
  const reducedMotion = useReducedMotion();
  const [hovered, setHovered] = useState<{
    layerIndex: number;
    neuronIndex: number;
    x: number;
    y: number;
    activation: number;
    bias: number;
  } | null>(null);

  const displayCounts = useMemo(() => layers.map((l) => Math.min(l.neurons, 12)), [layers]);

  const weightMaps = useMemo(() => {
    const maps: number[][][] = [];
    const seedFrom = (a: number, b: number, c: number) => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    const rand = (seed: number) => {
      let t = seed + 0x6d2b79f5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let i = 0; i < displayCounts.length - 1; i += 1) {
      const rows = displayCounts[i + 1];
      const cols = displayCounts[i];
      const seed = seedFrom(rows, cols, i + 1);
      const mat: number[][] = Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => {
          const value = rand(seed + r * 31 + c * 17);
          return (value - 0.5) * 2;
        }),
      );
      maps.push(mat);
    }
    return maps;
  }, [displayCounts]);

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
      const active = forwardPassState !== "idle" || autoPlay || isTraining;
      const frameBudget = active ? 33 : 100;
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

      const drift = !active && !reducedMotion ? (time * 0.0005) % 30 : 0;
      gridOffsetRef.current = drift;
      ctx.strokeStyle = "rgba(36,40,54,0.08)";
      ctx.lineWidth = 0.5;
      for (let x = -30 + drift; x < width; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = -30 + drift; y < height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const colGap = width / (layers.length + 1);
      const topPad = 48;
      const bottomPad = 72;
      const usableHeight = height - topPad - bottomPad;

      const layerPositions = layers.map((layer, i) => {
        const n = displayCounts[i];
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
        const widthLine = 0.5 + magnitude * 2.5;
        const opacity = dim ? 0.1 : 0.08 + magnitude * 0.5;
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
        const limitConnections = totalConnections > 200 && selectedLayer !== i;
        for (let t = 0; t < target.count; t += 1) {
          for (let s = 0; s < source.count; s += 1) {
            if (limitConnections && Math.abs(weights[t][s]) < 0.75) continue;
            const inactiveForPass =
              (!isTraining && forwardActiveLayer != null && forwardActiveLayer !== i) ||
              (!isTraining && backwardActiveLayer != null && backwardActiveLayer !== i);
            const dim =
              (hovered && !(hoveredLayer === i && hoveredNeuron === s) && !(hoveredLayer === i + 1 && hoveredNeuron === t)) ||
              inactiveForPass;
            const sourceAct = layerOutputs[String(i)]?.[s] ?? 0;
            const targetAct = layerOutputs[String(i + 1)]?.[t] ?? 0;
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
        const outputs = layerOutputs[String(i)] ?? [];
        const label = i === 0 ? "Input" : i === layers.length - 1 ? "Output" : `Hidden ${i}`;
        ctx.fillStyle = neuralPalette.silver;
        ctx.font = '10px "Inter", sans-serif';
        ctx.textAlign = "center";
        ctx.fillText(label.toUpperCase(), info.x, 18);
        ctx.fillStyle = neuralPalette.ash;
        ctx.font = '9px "Inter", sans-serif';
        ctx.fillText(`(${info.layer.neurons} neurons)`, info.x, height - 24);

        info.yPositions.forEach((y, j) => {
          const val = outputs[j] ?? 0;
          const intensity = Math.min(Math.abs(val) / maxActivation, 1);
          const radiusBase = info.count > 8 ? 12 : info.count < 4 ? 16 : 14;
          const radius = (hoveredLayer === i && hoveredNeuron === j) ? radiusBase * 1.2 : radiusBase;
          const color = activationColor(val);
          const isDead = val === 0 && (info.layer.activation || "").toLowerCase() === "relu";

          if (val !== 0) {
            const glow = ctx.createRadialGradient(info.x, y, radius, info.x, y, radius * 2.5);
            glow.addColorStop(0, color);
            glow.addColorStop(1, "rgba(0,0,0,0)");
            ctx.globalAlpha = 0.25 + intensity * 0.5;
            ctx.beginPath();
            ctx.arc(info.x, y, radius * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = glow;
            ctx.fill();
            ctx.globalAlpha = 1;
          }

          const core = ctx.createRadialGradient(info.x - radius * 0.3, y - radius * 0.3, radius * 0.3, info.x, y, radius);
          core.addColorStop(0, color);
          core.addColorStop(1, weightColor(val, 1));
          ctx.beginPath();
          ctx.arc(info.x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = core;
          ctx.fill();

          ctx.beginPath();
          ctx.ellipse(info.x - radius * 0.25, y - radius * 0.25, radius * 0.3, radius * 0.2, 0, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(232, 236, 248, 0.15)";
          ctx.fill();

          ctx.setLineDash(isDead ? [4, 3] : []);
          ctx.strokeStyle = isDead ? "rgba(239, 68, 68, 0.5)" : i === selectedLayer ? neuralPalette.synapse.bright : neuralPalette.graphite;
          ctx.lineWidth = i === selectedLayer ? 2 : 1.5;
          ctx.stroke();
          ctx.setLineDash([]);

          if (radius > 10) {
            ctx.fillStyle = `rgba(200, 206, 228, ${0.3 + intensity * 0.6})`;
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(val.toFixed(2), info.x, y);
          }

          if (isDead) {
            ctx.fillStyle = neuralPalette.lesion.bright;
            ctx.font = '12px "JetBrains Mono", monospace';
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("x", info.x, y);
          }
        });
      });

      for (let i = 0; i < layers.length - 1; i += 1) {
        const activation = layers[i + 1]?.activation;
        if (!activation) continue;
        const midX = (layerPositions[i].x + layerPositions[i + 1].x) / 2;
        const midY = height - 48;
        ctx.fillStyle = neuralPalette.obsidian;
        ctx.strokeStyle = neuralPalette.graphite;
        ctx.lineWidth = 1;
        const text = activation.toUpperCase();
        const pad = 6;
        ctx.font = '9px "JetBrains Mono", monospace';
        const metrics = ctx.measureText(text);
        const w = metrics.width + pad * 2;
        const h = 18;
        ctx.beginPath();
        ctx.roundRect(midX - w / 2, midY - h / 2, w, h, 6);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = neuralPalette.cloud;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, midX, midY);
      }

      if (gradientSummary?.per_layer?.length) {
        const barY = height - 56;
        gradientSummary.per_layer.forEach((g, idx) => {
          const midX = (layerPositions[idx]?.x + layerPositions[idx + 1]?.x) / 2;
          if (!midX) return;
          const norm = g.dW_norm ?? 0;
          const barH = Math.min(30, Math.log10(norm + 1) * 10 + 4);
          ctx.fillStyle = gradientHealthColor(norm);
          ctx.fillRect(midX - 4, barY - barH, 8, barH);
          ctx.fillStyle = neuralPalette.ash;
          ctx.font = '8px "JetBrains Mono", monospace';
          ctx.textAlign = "center";
          ctx.fillText(norm.toExponential(1), midX, barY + 12);
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
    layerOutputs,
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
      const topPad = 48;
      const bottomPad = 72;
      const usableHeight = height - topPad - bottomPad;
      let found = null as typeof hovered;
      layers.forEach((layer, i) => {
        const n = displayCounts[i];
        const spacing = usableHeight / (n + 1);
        const cx = colGap * (i + 1);
        for (let j = 0; j < n; j += 1) {
          const cy = topPad + spacing * (j + 1);
          const radiusBase = n > 8 ? 12 : n < 4 ? 16 : 14;
          const dx = x - cx;
          const dy = y - cy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= radiusBase * 1.2) {
            const activation = layerOutputs[String(i)]?.[j] ?? 0;
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
  }, [layers, layerOutputs, displayCounts]);

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
