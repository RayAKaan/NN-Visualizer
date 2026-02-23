import React from "react";
import OutputBars from "./OutputBars";
import NeuronGrid from "./NeuronGrid";
import FeatureMapViewer from "./FeatureMapViewer";
import { PredictionResult, CNNPredictionResult } from "../../types";

interface Props {
  pixels: number[];
  annResult: PredictionResult | null;
  cnnResult: CNNPredictionResult | null;
}

export default function ComparisonView({ annResult, cnnResult }: Props) {
  const annPred = annResult?.prediction ?? -1;
  const cnnPred = cnnResult?.prediction ?? -1;
  const annConf = annResult?.confidence ?? 0;
  const cnnConf = cnnResult?.confidence ?? 0;
  const agree = annPred === cnnPred && annPred >= 0;
  const annProbs = annResult?.probabilities ?? new Array(10).fill(0);
  const cnnProbs = cnnResult?.probabilities ?? new Array(10).fill(0);
  const confDiff = Math.abs(annConf - cnnConf) * 100;
  const moreConfident = annConf > cnnConf ? "ANN" : "CNN";
  const firstConv = cnnResult?.feature_maps?.find((f) => f.layer_type === "conv");

  return (
    <div className="comparison-container">
      <div className="comparison-panel">
        <div className="comparison-header"><span className="comparison-badge ann">ANN (Dense)</span></div>
        <div style={{ textAlign: "center", margin: "12px 0" }}><div style={{ fontSize: 48, fontWeight: 700 }}>{annPred >= 0 ? annPred : "?"}</div><div>{(annConf * 100).toFixed(1)}%</div></div>
        <OutputBars probabilities={annProbs} winner={annPred} />
        {annResult && annResult.model_type === "ann" ? <div style={{ marginTop: 12 }}><NeuronGrid activations={annResult.layers.hidden1} columns={16} label="Hidden 1 (128)" /><div style={{ height: 8 }} /><NeuronGrid activations={annResult.layers.hidden2} columns={8} label="Hidden 2 (64)" /></div> : null}
      </div>
      <div className="comparison-panel">
        <div className="comparison-header"><span className="comparison-badge cnn">CNN (Conv)</span></div>
        <div style={{ textAlign: "center", margin: "12px 0" }}><div style={{ fontSize: 48, fontWeight: 700 }}>{cnnPred >= 0 ? cnnPred : "?"}</div><div>{(cnnConf * 100).toFixed(1)}%</div></div>
        <OutputBars probabilities={cnnProbs} winner={cnnPred} />
        {firstConv ? <div style={{ marginTop: 12 }}><FeatureMapViewer layer={firstConv} maxTiles={6} /></div> : null}
      </div>
      <div className="comparison-summary">
        {annPred >= 0 && cnnPred >= 0 ? <><div className={agree ? "comparison-agree" : "comparison-disagree"}>{agree ? `✓ Both models agree: digit ${annPred}` : `✗ Disagreement: ANN says ${annPred}, CNN says ${cnnPred}`}</div><div style={{ marginTop: 8 }}>{moreConfident} is {confDiff.toFixed(1)}% more confident</div></> : <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Draw a digit to compare both models</div>}
      </div>
    </div>
  );
}
