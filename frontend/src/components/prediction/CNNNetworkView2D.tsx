import React from "react";
import { CNNPredictionResult } from "../../types";
import FeatureMapViewer from "./FeatureMapViewer";
import InputHeatmap from "./InputHeatmap";
import NeuronGrid from "./NeuronGrid";
import OutputNeurons from "./OutputNeurons";

interface Props {
  result: CNNPredictionResult | null;
  pixels: number[];
}

const CNNNetworkView2D: React.FC<Props> = ({ result, pixels }) => {
  if (!result) return <div className="card">Awaiting CNN prediction...</div>;
  const dense = result.dense_layers.dense ?? [];
  return (
    <div className="card">
      <div className="cnn-flow">
        <div className="cnn-layer-block">
          <div className="layer-title">Input 28×28</div>
          <InputHeatmap pixels={pixels} />
        </div>
      </div>
      <FeatureMapViewer featureMaps={result.feature_maps} kernels={result.kernels} />
      <div className="cnn-flow">
        <div className="cnn-layer-block">
          <div className="layer-title">Dense 128</div>
          <NeuronGrid activations={dense} columns={16} label="Dense" />
        </div>
        <div className="cnn-flow-arrow"><span>softmax</span></div>
        <div className="cnn-layer-block">
          <div className="layer-title">Output 10</div>
          <OutputNeurons output={result.probabilities} winner={result.prediction} />
        </div>
      </div>
    </div>
  );
};

export default CNNNetworkView2D;
