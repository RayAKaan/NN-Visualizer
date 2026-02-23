import React from "react";
import { FeatureMapLayer, KernelLayer } from "../../types";
import NeuronGrid from "../prediction/NeuronGrid";
import FeatureMapViewer from "./FeatureMapViewer";
import FlowArrow from "./FlowArrow";
import InputHeatmap from "./InputHeatmap";
import OutputNeurons from "./OutputNeurons";

interface Props {
  pixels?: number[];
  featureMaps?: FeatureMapLayer[];
  kernels?: KernelLayer[];
  denseLayers?: Record<string, number[]>;
  probabilities?: number[];
  prediction?: number;
  result?: any;
}

export default function CNNNetworkView2D({ pixels = [], featureMaps, kernels = [], denseLayers, probabilities, prediction = 0, result }: Props) {
  const fMaps: FeatureMapLayer[] = featureMaps ?? result?.feature_maps ?? [];
  const dLayers: Record<string, number[]> = denseLayers ?? result?.dense_layers ?? {};
  const probs = probabilities ?? result?.probabilities ?? Array(10).fill(0);
  const pred = prediction ?? result?.prediction ?? 0;

  const findLayer = (keyword: string, channels?: number, spatial?: number) =>
    fMaps.find((f: FeatureMapLayer) => {
      if (keyword && !f.layer_name.toLowerCase().includes(keyword)) return false;
      if (channels !== undefined && f.shape[2] !== channels) return false;
      if (spatial !== undefined && f.shape[0] !== spatial) return false;
      return true;
    });

  const conv1 = findLayer("conv", 32, 28) || fMaps[0];
  const pool1 = findLayer("pool", 32, 14) || fMaps[1];
  const conv2 = findLayer("conv", 64, 14) || fMaps[2];
  const pool2 = findLayer("pool", 64, 7) || fMaps[3];
  const denseActs: number[] = Object.values(dLayers)[0] || [];
  void kernels;

  return (
    <div className="cnn-flow">
      <div className="cnn-layer-block"><div className="cnn-layer-title">Input 28×28</div><InputHeatmap pixels={pixels} /></div>
      <FlowArrow label="3×3 Conv, 32" />
      {conv1 && <div className="cnn-layer-block"><div className="cnn-layer-title">Conv1 ({conv1.shape[0]}×{conv1.shape[1]}×{conv1.shape[2]})</div><FeatureMapViewer layer={conv1} maxTiles={8} /></div>}
      <FlowArrow label="2×2 MaxPool" />
      {pool1 && <div className="cnn-layer-block"><div className="cnn-layer-title">Pool1 ({pool1.shape[0]}×{pool1.shape[1]}×{pool1.shape[2]})</div><FeatureMapViewer layer={pool1} maxTiles={6} /></div>}
      <FlowArrow label="3×3 Conv, 64" />
      {conv2 && <div className="cnn-layer-block"><div className="cnn-layer-title">Conv2 ({conv2.shape[0]}×{conv2.shape[1]}×{conv2.shape[2]})</div><FeatureMapViewer layer={conv2} maxTiles={8} /></div>}
      <FlowArrow label="2×2 MaxPool" />
      {pool2 && <div className="cnn-layer-block"><div className="cnn-layer-title">Pool2 ({pool2.shape[0]}×{pool2.shape[1]}×{pool2.shape[2]})</div><FeatureMapViewer layer={pool2} maxTiles={4} /></div>}
      <FlowArrow label="Flatten + Dense" />
      {denseActs.length > 0 && <div className="cnn-layer-block"><div className="cnn-layer-title">Dense ({denseActs.length})</div><NeuronGrid values={denseActs} /></div>}
      <FlowArrow label="Softmax" />
      <div className="cnn-layer-block"><div className="cnn-layer-title">Output (10)</div><OutputNeurons probabilities={probs} prediction={pred} /></div>
    </div>
  );
}
