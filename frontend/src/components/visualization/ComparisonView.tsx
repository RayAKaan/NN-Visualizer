import React from "react";
import { AnyPredictionResult, CNNPredictionResult, PredictionResult, RNNPredictionResult } from "../../types";
import NeuronGrid from "../prediction/NeuronGrid";
import OutputBars from "../prediction/OutputBars";
import FeatureMapViewer from "./FeatureMapViewer";

interface Props {
  pixels?: number[];
  annResult?: any;
  cnnResult?: any;
  rnnResult?: any;
  results?: Array<{ model: string; prediction: string | number }>;
}

interface ModelEntry {
  key: string;
  label: string;
  badge: string;
  arch: string;
  pred: number;
  conf: number;
  probs: number[];
  result: AnyPredictionResult | null;
}

export default function ComparisonView({ annResult = null, cnnResult = null, rnnResult = null, results = [] }: Props) {
  const fallback: Record<string, number> = Object.fromEntries(results.map((r) => [r.model, Number(r.prediction) || 0]));
  const models: ModelEntry[] = [];
  if (annResult || fallback.ann !== undefined) models.push({ key: "ann", label: "ANN", badge: "ann", arch: "784→256→128→64→10", pred: annResult?.prediction ?? fallback.ann ?? -1, conf: annResult?.confidence ?? 0, probs: annResult?.probabilities ?? Array(10).fill(0), result: annResult });
  if (cnnResult || fallback.cnn !== undefined) models.push({ key: "cnn", label: "CNN", badge: "cnn", arch: "Conv32→Pool→Conv64→Pool→Dense128→10", pred: cnnResult?.prediction ?? fallback.cnn ?? -1, conf: cnnResult?.confidence ?? 0, probs: cnnResult?.probabilities ?? Array(10).fill(0), result: cnnResult });
  if (rnnResult || fallback.rnn !== undefined) models.push({ key: "rnn", label: "RNN", badge: "rnn", arch: "LSTM(128)→Dense(64)→10", pred: rnnResult?.prediction ?? fallback.rnn ?? -1, conf: rnnResult?.confidence ?? 0, probs: rnnResult?.probabilities ?? Array(10).fill(0), result: rnnResult });

  const hasResults = models.length > 0 && models.some((m) => m.pred >= 0);
  const allAgree = hasResults && models.every((m) => m.pred === models[0].pred);
  const mostConfident = hasResults ? models.reduce((a, b) => (a.conf > b.conf ? a : b)) : null;
  const colTemplate = models.length === 3 ? "1fr 1fr 1fr" : models.length === 2 ? "1fr 1fr" : "1fr";

  const confColor = (c: number) => (c > 0.9 ? "var(--accent-green)" : c > 0.7 ? "var(--accent-amber)" : "var(--accent-red)");

  return (
    <div className="comparison-container" style={{ display: "grid", gridTemplateColumns: colTemplate, gap: 12 }}>
      {models.map((m) => (
        <div key={m.key} className="comparison-panel">
          <div className="comparison-header"><span className={`comparison-badge ${m.badge}`}>{m.label}</span><span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-muted)", marginLeft: "auto" }}>{m.arch}</span></div>
          <div style={{ textAlign: "center", margin: "12px 0" }}><div style={{ fontSize: 48, fontWeight: 700, fontFamily: "var(--font-mono)", color: m.pred >= 0 ? "var(--text-primary)" : "var(--text-muted)" }}>{m.pred >= 0 ? m.pred : "?"}</div><div style={{ fontSize: 14, fontFamily: "var(--font-mono)", color: confColor(m.conf), fontWeight: 600 }}>{m.pred >= 0 ? `${(m.conf * 100).toFixed(1)}%` : "—"}</div></div>
          <OutputBars probabilities={m.probs} winner={m.pred} />
          <div style={{ marginTop: 12 }}>
            {m.key === "ann" && (m.result as PredictionResult | null)?.layers?.hidden1 && <NeuronGrid values={(m.result as PredictionResult).layers.hidden1.slice(0, 64)} />}
            {m.key === "cnn" && (m.result as CNNPredictionResult | null)?.feature_maps?.[0] && <FeatureMapViewer layer={(m.result as CNNPredictionResult).feature_maps[0]} maxTiles={4} />}
            {m.key === "rnn" && (m.result as RNNPredictionResult | null)?.timestep_activations && <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 40, padding: "4px 0" }}>{(m.result as RNNPredictionResult).timestep_activations.map((a, t, arr) => { const mx = Math.max(0.001, ...arr.map(Math.abs)); return <div key={t} style={{ width: 5, height: 5 + (Math.abs(a) / mx) * 30, backgroundColor: `rgba(245,158,11,${0.3 + (Math.abs(a) / mx) * 0.7})`, borderRadius: 1 }} />; })}</div>}
          </div>
        </div>
      ))}

      <div className="comparison-summary" style={{ gridColumn: "1 / -1" }}>
        {hasResults ? (
          <>
            <div className={allAgree ? "comparison-agree" : "comparison-disagree"} style={{ fontSize: 16, marginBottom: 12 }}>{allAgree ? `✓ All ${models.length} models agree: digit ${models[0].pred}` : `✗ Disagreement: ${models.map((m) => `${m.label}→${m.pred}`).join(", ")}`}</div>
            <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap", marginBottom: 12 }}>{models.map((m) => <div key={m.key} style={{ width: 160 }}><div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 4 }}>{m.label}: {(m.conf * 100).toFixed(1)}%</div><div style={{ height: 8, background: "var(--bg-secondary)", borderRadius: 4, overflow: "hidden" }}><div style={{ height: "100%", width: `${m.conf * 100}%`, background: m.key === "ann" ? "var(--accent-purple)" : m.key === "cnn" ? "var(--accent-cyan)" : "var(--accent-amber)", borderRadius: 4, transition: "width 300ms" }} /></div></div>)}</div>
            {mostConfident && <div style={{ fontSize: 12, color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>Most confident: {mostConfident.label} at {(mostConfident.conf * 100).toFixed(1)}%</div>}
          </>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Draw a digit to compare all models</div>
        )}
      </div>
    </div>
  );
}
