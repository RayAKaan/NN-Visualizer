import React from "react";
import { PredictionResult } from "../../types";
import OutputBars from "../prediction/OutputBars";
import PredictionCard from "../prediction/PredictionCard";
import ExplanationPanel from "../prediction/ExplanationPanel";

const RightPanel: React.FC<{ result: PredictionResult | null }> = ({ result }) => {
  if (!result) {
    return <aside className="card">Awaiting prediction...</aside>;
  }

  const winner = result.prediction;
  const confidence = result.probabilities[winner] ?? 0;

  return (
    <aside className="right-stack">
      <OutputBars probabilities={result.probabilities} winner={winner} />
      <PredictionCard digit={winner} confidence={confidence} />
      <ExplanationPanel explanation={result.explanation} />
    </aside>
  );
};

export default RightPanel;
