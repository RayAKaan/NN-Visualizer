import React, { useMemo } from "react";
import { NeuralState } from "../../types";
import InputHeatmap from "./InputHeatmap";
import NeuronGrid from "./NeuronGrid";
import OutputNeurons from "./OutputNeurons";
import ConnectionLines from "./ConnectionLines";

const NetworkView2D: React.FC<{ state: NeuralState | null }> = ({ state }) => {
  const topEdges = useMemo(() => {
    if (!state) return [];
    const joined = [...state.edges.hidden1_hidden2, ...state.edges.hidden2_output];
    return joined.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength)).slice(0, 100);
  }, [state]);

  if (!state) return <div className="card">Awaiting prediction...</div>;

  return (
    <div className="card network2d">
      <ConnectionLines edges={topEdges} width={860} height={260} />
      <div className="layer-flow">
        <div className="layer-card">
          <div className="layer-label">Input Layer (784) · pixels</div>
          <InputHeatmap pixels={state.input} />
        </div>
        <NeuronGrid activations={state.layers.hidden1} columns={16} label="Hidden Layer 1 (128) · ReLU" />
        <NeuronGrid activations={state.layers.hidden2} columns={8} label="Hidden Layer 2 (64) · ReLU" />
        <div className="layer-card">
          <div className="layer-label">Output Layer (10) · softmax</div>
          <OutputNeurons output={state.layers.output} winner={state.prediction} />
        </div>
      </div>
      <div className="activation-legend">Neuron Activation Strength · 0.0 → 1.0</div>
    </div>
  );
};

export default NetworkView2D;
