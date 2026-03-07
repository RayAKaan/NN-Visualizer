import React, { useMemo, useState } from "react";
import { Brain } from "lucide-react";
import NeuronHoverCard from "./NeuronHoverCard";
import NeuronInspector from "./NeuronInspector";
import NetworkGraph from "./NetworkGraph";
import PlaybackControls from "./PlaybackControls";
import { useNeurofluxStore } from "../../store/useNeurofluxStore";
import { useWebSocketStream } from "../../hooks/useWebSocketStream";

export default function NeurofluxionApp() {
  const mode = useNeurofluxStore((s) => s.mode);
  const topology = useNeurofluxStore((s) => s.topology);
  const selectedNeuronId = useNeurofluxStore((s) => s.selectedNeuronId);
  const hoveredNeuronId = useNeurofluxStore((s) => s.hoveredNeuronId);
  const setMode = useNeurofluxStore((s) => s.setMode);
  const history = useNeurofluxStore((s) => s.history);
  const currentEpoch = useNeurofluxStore((s) => s.currentEpoch);
  const playbackState = useNeurofluxStore((s) => s.playbackState);
  const { status } = useWebSocketStream();
  const [hoverPos, setHoverPos] = useState({ x: 24, y: 90 });

  const selectedNeuron = useMemo(
    () => topology.neurons.find((n) => n.id === selectedNeuronId) ?? null,
    [topology.neurons, selectedNeuronId],
  );
  const hoveredNeuron = useMemo(
    () => topology.neurons.find((n) => n.id === hoveredNeuronId) ?? null,
    [topology.neurons, hoveredNeuronId],
  );

  return (
    <div className="h-full min-h-[720px] grid grid-cols-1 xl:grid-cols-[1fr_390px] bg-slate-950 text-slate-100">
      <div className="relative border-r border-slate-800 overflow-hidden">
        <div className="h-14 px-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/70">
          <div className="flex items-center gap-2">
            <Brain size={18} className="text-cyan-400" />
            <h2 className="font-semibold">Neurofluxion - Phase 2 ANN Graph</h2>
          </div>
          <div className="flex rounded-md border border-slate-700 overflow-hidden text-xs">
            <button
              onClick={() => setMode("prediction")}
              className={`px-3 py-1.5 ${mode === "prediction" ? "bg-cyan-700 text-cyan-100" : "bg-slate-900 text-slate-300"}`}
            >
              Prediction Mode
            </button>
            <button
              onClick={() => setMode("training")}
              className={`px-3 py-1.5 ${mode === "training" ? "bg-fuchsia-700 text-fuchsia-100" : "bg-slate-900 text-slate-300"}`}
            >
              Training Mode
            </button>
          </div>
        </div>

        <div className="px-4 pt-3">
          <PlaybackControls connectionStatus={status} />
          <div className="mt-2 text-[11px] text-slate-500">
            Stream snapshots: {history.length} | Playback: {playbackState} | Inspector epoch: {currentEpoch}
          </div>
        </div>

        <div className="h-[calc(100%-118px)] p-4 pt-2">
          <NetworkGraph onHoverPosition={(x, y) => setHoverPos({ x, y })} />
        </div>

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

      <div className="p-4 bg-slate-950">
        <NeuronInspector neuron={selectedNeuron} mode={mode} />
      </div>
    </div>
  );
}
