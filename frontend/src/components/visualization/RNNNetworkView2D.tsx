import React, { useMemo } from "react";
import NeuronGrid from "../prediction/NeuronGrid";
import FlowArrow from "./FlowArrow";
import InputHeatmap from "./InputHeatmap";
import OutputNeurons from "./OutputNeurons";

interface Props {
  pixels?: number[];
  timestepActivations?: number[];
  lstmOutput?: number[];
  denseLayers?: Record<string, number[]>;
  probabilities?: number[];
  prediction?: number;
  result?: any;
}

function activationColor(a: number): string {
  const c = Math.max(0, Math.min(1, a));
  if (c <= 0.5) {
    const t = c / 0.5;
    return `rgb(${Math.round(10 + t * 129)},${Math.round(14 + t * 78)},${Math.round(23 + t * 223)})`;
  }
  const t = (c - 0.5) / 0.5;
  return `rgb(${Math.round(139 + t * (6 - 139))},${Math.round(92 + t * 90)},${Math.round(246 + t * (212 - 246))})`;
}

export default function RNNNetworkView2D({ pixels = [], timestepActivations, lstmOutput, denseLayers, probabilities, prediction = 0, result }: Props) {
  const tsActs: number[] = timestepActivations ?? result?.timestep_activations ?? [];
  const lstm: number[] = lstmOutput ?? result?.lstm_output ?? [];
  const dense: Record<string, number[]> = denseLayers ?? result?.dense_layers ?? {};
  const probs = probabilities ?? result?.probabilities ?? Array(10).fill(0);
  const pred = prediction ?? result?.prediction ?? 0;
  const maxAct = useMemo(() => Math.max(0.001, ...tsActs.map((n: number) => Math.abs(n))), [tsActs]);
  const denseActs: number[] = Object.values(dense)[0] || [];

  return (
    <div className="cnn-flow">
      <div className="cnn-layer-block">
        <div className="cnn-layer-title">Input 28×28</div>
        <InputHeatmap pixels={pixels} />
        <span className="layer-label" style={{ marginTop: 4 }}>28 timesteps × 28 features</span>
      </div>

      <FlowArrow label="LSTM" />

      <div className="cnn-layer-block" style={{ minWidth: 260 }}>
        <div className="cnn-layer-title">Timestep Activations (28)</div>
        <div className="rnn-flow" style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 100, padding: "8px 4px" }}>
          {tsActs.map((act: number, t: number) => {
            const normAct = Math.abs(act) / maxAct;
            const barH = 10 + normAct * 80;
            return (
              <div key={t} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                <div className="rnn-timestep" title={`t=${t} | activation: ${act.toFixed(4)}`} style={{ width: 7, height: barH, backgroundColor: activationColor(normAct), borderRadius: 2, transition: "height 200ms, background-color 200ms" }} />
                {t % 7 === 0 && <span style={{ fontSize: 7, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>t={t}</span>}
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginTop: 4, textAlign: "center" }}>Sequential processing →</div>
      </div>

      <FlowArrow label="Final State" />

      {lstm.length > 0 ? (
        <div className="cnn-layer-block"><div className="cnn-layer-title">LSTM Output ({lstm.length})</div><NeuronGrid values={lstm} /></div>
      ) : (
        <div className="cnn-layer-block"><div className="cnn-layer-title">LSTM Output</div><div className="rnn-cell-state">No state available</div></div>
      )}

      <FlowArrow label="Dense" />
      {denseActs.length > 0 && <div className="cnn-layer-block"><div className="cnn-layer-title">Dense ({denseActs.length})</div><NeuronGrid values={denseActs} /></div>}
      <FlowArrow label="Softmax" />
      <div className="cnn-layer-block"><div className="cnn-layer-title">Output (10)</div><OutputNeurons probabilities={probs} prediction={pred} /></div>
    </div>
  );
}
