import React from "react";
import { FeatureMapLayer, KernelLayer } from "../../types";
import InputHeatmap from "./InputHeatmap";
import FlowArrow from "./FlowArrow";
import FeatureMapViewer from "./FeatureMapViewer";
import KernelViewer from "./KernelViewer";
import NeuronGrid from "./NeuronGrid";
import OutputNeurons from "./OutputNeurons";

interface Props {
  pixels: number[];
  featureMaps: FeatureMapLayer[];
  kernels: KernelLayer[];
  denseLayers: Record<string, number[]>;
  probabilities: number[];
  prediction: number;
}

export default function CNNNetworkView2D({ pixels, featureMaps, kernels, denseLayers, probabilities, prediction }: Props) {
  const convLayers = featureMaps.filter((f) => f.layer_type === "conv");
  const poolLayers = featureMaps.filter((f) => f.layer_type === "pool");
  const dense = denseLayers.dense ?? denseLayers.dense1 ?? [];

  return (
    <div className="cnn-flow">
      <div className="cnn-layer-block"><div className="cnn-layer-title">Input</div><InputHeatmap pixels={pixels} /></div>
      <FlowArrow label="3×3 Conv" />
      {convLayers[0] ? <div className="cnn-layer-block"><div className="cnn-layer-title">{convLayers[0].layer_name}</div><FeatureMapViewer layer={convLayers[0]} maxTiles={8} /></div> : null}
      {kernels[0] ? <div className="cnn-layer-block"><div className="cnn-layer-title">Kernels</div><KernelViewer layer={kernels[0]} maxKernels={6} /></div> : null}
      <FlowArrow label="2×2 MaxPool" />
      {poolLayers[0] ? <div className="cnn-layer-block"><div className="cnn-layer-title">{poolLayers[0].layer_name}</div><FeatureMapViewer layer={poolLayers[0]} maxTiles={6} /></div> : null}
      <FlowArrow label="Dense" />
      <div className="cnn-layer-block"><div className="cnn-layer-title">Dense</div><NeuronGrid activations={dense} columns={16} label="Dense (128) · ReLU" /></div>
      <FlowArrow label="Softmax" />
      <div className="cnn-layer-block"><div className="cnn-layer-title">Output</div><OutputNeurons probabilities={probabilities} prediction={prediction} /></div>
    </div>
  );
}
