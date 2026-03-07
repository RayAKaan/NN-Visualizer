import React, { useEffect, useMemo, useRef, useState } from "react";
import { Brain, Eraser } from "lucide-react";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { useWebSocketStream } from "../../hooks/useWebSocketStream";
import ANNVisualizer from "./ANNVisualizer";
import CNNVisualizer from "./CNNVisualizer";
import MetricsDashboard from "./MetricsDashboard";
import NeuronHoverCard from "./NeuronHoverCard";
import NeuronInspector from "./NeuronInspector";
import PlaybackControls from "./PlaybackControls";
import RNNVisualizer from "./RNNVisualizer";
import { ArchitectureType, NeuronState } from "./types";

const PAD_DISPLAY = 170;
const PAD_INTERNAL = 340;

class LabSectionBoundary extends React.Component<
  { title: string; children: React.ReactNode },
  { hasError: boolean; message: string }
> {
  constructor(props: { title: string; children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: unknown) {
    return { hasError: true, message: error instanceof Error ? error.message : "Unknown error" };
  }

  componentDidCatch(error: unknown) {
    console.error(`Lab section "${this.props.title}" crashed`, error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full w-full rounded-xl border border-rose-700/50 bg-rose-950/30 p-4">
          <div className="text-sm font-semibold text-rose-300">{this.props.title} failed to render</div>
          <div className="mt-2 text-xs text-rose-200/90 font-mono break-words">{this.state.message}</div>
          <div className="mt-3 text-xs text-rose-200/80">Switch architecture/mode or reload the page.</div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function NeurofluxionLayout() {
  const mode = useNeurofluxStore((s) => s.mode);
  const currentArchitecture = useNeurofluxStore((s) => s.currentArchitecture);
  const topology = useNeurofluxStore((s) => s.topology);
  const selectedNeuronId = useNeurofluxStore((s) => s.selectedNeuronId);
  const hoveredNeuronId = useNeurofluxStore((s) => s.hoveredNeuronId);
  const showNeuronHealth = useNeurofluxStore((s) => s.showNeuronHealth);
  const setMode = useNeurofluxStore((s) => s.setMode);
  const setCurrentArchitecture = useNeurofluxStore((s) => s.setCurrentArchitecture);
  const toggleNeuronHealth = useNeurofluxStore((s) => s.toggleNeuronHealth);

  const [hoverPos, setHoverPos] = useState({ x: 24, y: 90 });
  const [vizSelectedNeuron, setVizSelectedNeuron] = useState<NeuronState | null>(null);
  const [vizHoveredNeuron, setVizHoveredNeuron] = useState<NeuronState | null>(null);
  const [labInputPixels, setLabInputPixels] = useState<number[]>(Array.from({ length: 784 }, () => 0));

  const padRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  const { status } = useWebSocketStream();

  useEffect(() => {
    const ctx = padRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, PAD_INTERNAL, PAD_INTERNAL);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 22;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const exportPixels = () => {
    const src = padRef.current;
    if (!src) return;
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d");
    if (!sctx) return;
    sctx.fillStyle = "black";
    sctx.fillRect(0, 0, 28, 28);
    sctx.imageSmoothingEnabled = true;
    sctx.drawImage(src, 0, 0, 28, 28);
    const d = sctx.getImageData(0, 0, 28, 28).data;
    const out: number[] = [];
    for (let i = 0; i < d.length; i += 4) out.push(d[i] / 255);
    setLabInputPixels(out);
  };

  const drawAt = (ev: React.MouseEvent) => {
    const cv = padRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx || !isDrawing.current) return;
    const rect = cv.getBoundingClientRect();
    const scale = cv.width / rect.width;
    const x = (ev.clientX - rect.left) * scale;
    const y = (ev.clientY - rect.top) * scale;
    if (!last.current) {
      last.current = { x, y };
      return;
    }
    const midX = (last.current.x + x) / 2;
    const midY = (last.current.y + y) / 2;
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.quadraticCurveTo(last.current.x, last.current.y, midX, midY);
    ctx.stroke();
    last.current = { x, y };
    exportPixels();
  };

  const startDraw = (ev: React.MouseEvent) => {
    isDrawing.current = true;
    const cv = padRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const rect = cv.getBoundingClientRect();
    const scale = cv.width / rect.width;
    const x = (ev.clientX - rect.left) * scale;
    const y = (ev.clientY - rect.top) * scale;
    last.current = { x, y };
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = "white";
    ctx.fill();
    exportPixels();
  };

  const endDraw = () => {
    isDrawing.current = false;
    last.current = null;
    exportPixels();
  };

  const clearPad = () => {
    const ctx = padRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, PAD_INTERNAL, PAD_INTERNAL);
    setLabInputPixels(Array.from({ length: 784 }, () => 0));
  };

  const selectedNeuronFromTopology = useMemo(
    () => topology.neurons.find((n) => n.id === selectedNeuronId) ?? null,
    [topology.neurons, selectedNeuronId],
  );
  const hoveredNeuronFromTopology = useMemo(
    () => topology.neurons.find((n) => n.id === hoveredNeuronId) ?? null,
    [topology.neurons, hoveredNeuronId],
  );

  const selectedNeuron = vizSelectedNeuron ?? selectedNeuronFromTopology;
  const hoveredNeuron = vizHoveredNeuron ?? hoveredNeuronFromTopology;

  return (
    <div className="h-full grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_420px] 2xl:grid-cols-[minmax(0,1fr)_460px] bg-slate-950 text-slate-100">
      <div className="relative border-r border-slate-800 min-h-0 flex flex-col overflow-hidden">
        <div className="h-14 px-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/70 shrink-0">
          <div className="flex items-center gap-2">
            <Brain size={18} className="text-cyan-400" />
            <h2 className="font-semibold">Neurofluxion Laboratory</h2>
          </div>

          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1 text-xs text-slate-300 border border-slate-700 rounded px-2 py-1.5 bg-slate-900">
              <input
                type="checkbox"
                checked={showNeuronHealth}
                onChange={toggleNeuronHealth}
                className="accent-cyan-500"
              />
              Show Dead/Critical
            </label>
            <select
              value={currentArchitecture}
              onChange={(e) => {
                setCurrentArchitecture(e.target.value as ArchitectureType);
                setVizHoveredNeuron(null);
                setVizSelectedNeuron(null);
              }}
              className="text-xs bg-slate-900 border border-slate-700 rounded px-2 py-1.5"
            >
              <option value="ann">ANN</option>
              <option value="cnn">CNN</option>
              <option value="rnn">RNN</option>
            </select>
            <div className="flex rounded-md border border-slate-700 overflow-hidden text-xs">
              <button
                onClick={() => setMode("prediction")}
                className={`px-3 py-1.5 ${mode === "prediction" ? "bg-cyan-700 text-cyan-100" : "bg-slate-900 text-slate-300"}`}
              >
                Prediction
              </button>
              <button
                onClick={() => setMode("training")}
                className={`px-3 py-1.5 ${mode === "training" ? "bg-fuchsia-700 text-fuchsia-100" : "bg-slate-900 text-slate-300"}`}
              >
                Training
              </button>
            </div>
          </div>
        </div>

        <div className="px-4 pt-3 shrink-0">
          <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_260px] gap-2 items-start">
            <PlaybackControls connectionStatus={status} />
            <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-2">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[11px] font-mono text-slate-300">Shared Lab Input (28x28)</div>
                <button onClick={clearPad} className="text-xs px-2 py-1 rounded border border-slate-700 bg-slate-800 text-slate-300 flex items-center gap-1">
                  <Eraser size={12} /> Clear
                </button>
              </div>
              <div className="relative w-[170px] h-[170px] mx-auto">
                <canvas
                  ref={padRef}
                  width={PAD_INTERNAL}
                  height={PAD_INTERNAL}
                  className="absolute inset-0 rounded border border-slate-600 bg-black cursor-crosshair"
                  style={{ width: `${PAD_DISPLAY}px`, height: `${PAD_DISPLAY}px` }}
                  onMouseDown={startDraw}
                  onMouseMove={drawAt}
                  onMouseUp={endDraw}
                  onMouseLeave={endDraw}
                />
                <div
                  className="absolute inset-0 pointer-events-none rounded"
                  style={{
                    backgroundImage:
                      "linear-gradient(to right, rgba(148,163,184,0.14) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.14) 1px, transparent 1px)",
                    backgroundSize: "calc(100% / 28) calc(100% / 28)",
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 pt-2 flex-1 min-h-[360px] overflow-hidden">
          <div className="h-full rounded-xl border border-slate-700 bg-slate-900/45 p-2">
            <LabSectionBoundary title="Visualizer">
              {currentArchitecture === "ann" ? (
                <ANNVisualizer
                  inputPixels={labInputPixels}
                  onHoverPosition={(x, y) => setHoverPos({ x, y })}
                  onHoverNeuronData={setVizHoveredNeuron}
                  onSelectNeuronData={setVizSelectedNeuron}
                />
              ) : currentArchitecture === "cnn" ? (
                <CNNVisualizer
                  inputPixels={labInputPixels}
                  onHoverPosition={(x, y) => setHoverPos({ x, y })}
                  onHoverNeuronData={setVizHoveredNeuron}
                  onSelectNeuronData={setVizSelectedNeuron}
                />
              ) : (
                <RNNVisualizer inputPixels={labInputPixels} onHoverPosition={(x, y) => setHoverPos({ x, y })} onHoverNeuronData={setVizHoveredNeuron} onSelectNeuronData={setVizSelectedNeuron} />
              )}
            </LabSectionBoundary>
          </div>
        </div>

        {mode === "training" && (
          <div className="px-4 pb-4 h-[260px] shrink-0">
            <LabSectionBoundary title="Training Metrics">
              <MetricsDashboard />
            </LabSectionBoundary>
          </div>
        )}

        {hoveredNeuron && (
          <div
            className="absolute z-30 pointer-events-none"
            style={{
              left: hoverPos.x + 14,
              top: hoverPos.y + 14,
            }}
          >
            <NeuronHoverCard neuron={hoveredNeuron} />
          </div>
        )}
      </div>

      <div className="p-4 bg-slate-950 h-full min-h-[360px]">
        <NeuronInspector neuron={selectedNeuron} mode={mode} />
      </div>
    </div>
  );
}

