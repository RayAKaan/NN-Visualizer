import React from "react";
import { NeuronState } from "./types";

interface Props {
  neuron: NeuronState;
}

export default function NeuronHoverCard({ neuron }: Props) {
  const totalConnections = neuron.incomingEdges.length + neuron.outgoingEdges.length;

  return (
    <div className="w-64 rounded-xl border border-cyan-500/30 bg-slate-900/95 backdrop-blur p-3 shadow-2xl shadow-cyan-900/30">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-slate-400">Neuron</div>
        <div className="text-[11px] text-cyan-300">{neuron.layerType}</div>
      </div>
      <div className="mt-1 font-mono text-sm text-cyan-200">{neuron.id}</div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-slate-700 bg-slate-800/80 p-2">
          <div className="text-slate-400">Activation</div>
          <div className="font-semibold text-emerald-300">{neuron.activation.toFixed(4)}</div>
        </div>
        <div className="rounded border border-slate-700 bg-slate-800/80 p-2">
          <div className="text-slate-400">Bias</div>
          <div className="font-semibold text-sky-300">{neuron.bias.toFixed(4)}</div>
        </div>
        <div className="rounded border border-slate-700 bg-slate-800/80 p-2">
          <div className="text-slate-400">Gradient</div>
          <div className="font-semibold text-fuchsia-300">{neuron.gradient.toFixed(6)}</div>
        </div>
        <div className="rounded border border-slate-700 bg-slate-800/80 p-2">
          <div className="text-slate-400">Connections</div>
          <div className="font-semibold text-amber-300">{totalConnections}</div>
        </div>
      </div>
    </div>
  );
}
