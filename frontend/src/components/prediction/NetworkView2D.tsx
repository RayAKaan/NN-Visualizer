import React, { useRef, useState, useEffect } from "react";
import InputHeatmap from "./InputHeatmap";
import NeuronGrid from "./NeuronGrid";
import OutputNeurons from "./OutputNeurons";
import FlowArrow from "./FlowArrow";
import ConnectionLines from "./ConnectionLines";
import ActivationLegend from "./ActivationLegend";

interface Props {
  pixels: number[];
  hidden1: number[];
  hidden2: number[];
  probabilities: number[];
  prediction: number;
  weights: Record<string, { kernel: number[][] }> | null;
}

export default function NetworkView2D({ pixels, hidden1, hidden2, probabilities, prediction, weights }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 800, height: 400 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setDims({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div className="network-2d" ref={containerRef}>
      <div className="network-2d-layers">
        <InputHeatmap pixels={pixels} />
        <FlowArrow label="Dense" />
        <NeuronGrid activations={hidden1} columns={16} label="Hidden 1 (128) · ReLU" highlightId="hidden1" />
        <FlowArrow label="Dense" />
        <NeuronGrid activations={hidden2} columns={8} label="Hidden 2 (64) · ReLU" highlightId="hidden2" />
        <FlowArrow label="Dense" />
        <OutputNeurons probabilities={probabilities} prediction={prediction} />
      </div>

      <ConnectionLines hidden1={hidden1} hidden2={hidden2} probabilities={probabilities} weights={weights} width={dims.width} height={dims.height} />
      <ActivationLegend />
    </div>
  );
}
