import React, { useEffect, useMemo, useRef, useState } from "react";
import NeuronGrid from "../prediction/NeuronGrid";
import ActivationLegend from "./ActivationLegend";
import ConnectionLines from "./ConnectionLines";
import FlowArrow from "./FlowArrow";
import InputHeatmap from "./InputHeatmap";
import OutputNeurons from "./OutputNeurons";

interface DirectProps {
  pixels: number[];
  hidden1: number[];
  hidden2: number[];
  hidden3: number[];
  probabilities: number[];
  prediction: number;
  weights: any;
}

interface StateProps {
  state?: any;
  weights: any;
}

type Props = Partial<DirectProps> & StateProps & Record<string, any>;

export default function NetworkView2D({ state, weights, pixels = [], hidden1 = [], hidden2 = [], hidden3 = [], probabilities = [], prediction = 0 }: Props) {
  const p = pixels.length ? pixels : Array(784).fill(0);
  const h1 = hidden1.length ? hidden1 : state?.layers?.hidden1 ?? [];
  const h2 = hidden2.length ? hidden2 : state?.layers?.hidden2 ?? [];
  const h3 = hidden3.length ? hidden3 : state?.layers?.hidden3 ?? [];
  const probs = probabilities.length ? probabilities : state?.probabilities ?? Array(10).fill(0);
  const pred = Number.isFinite(prediction) ? prediction : state?.prediction ?? 0;

  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 800, height: 420 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      setDims({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const signalStrength = useMemo(() => {
    const arr = [...h1.slice(0, 16), ...h2.slice(0, 16), ...h3.slice(0, 16), ...probs];
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + Math.max(0, b), 0) / arr.length;
  }, [h1, h2, h3, probs]);

  return (
    <div className="network-2d" ref={containerRef} style={{ position: "relative", minHeight: 460, height: "100%", overflow: "hidden" }}>
      <div className="network-2d-layers" style={{ position: "relative", zIndex: 2 }}>
        <InputHeatmap pixels={p} />
        <FlowArrow label="Dense" />
        <div className="layer-block" data-highlight="hidden1"><NeuronGrid values={h1.slice(0, 256)} highlightId="hidden1" /><span className="layer-label">Hidden 1 (256) · ReLU</span></div>
        <FlowArrow label="Dense" />
        <div className="layer-block" data-highlight="hidden2"><NeuronGrid values={h2.slice(0, 128)} highlightId="hidden2" /><span className="layer-label">Hidden 2 (128) · ReLU</span></div>
        <FlowArrow label="Dense" />
        <div className="layer-block" data-highlight="hidden3"><NeuronGrid values={h3.slice(0, 64)} highlightId="hidden3" /><span className="layer-label">Hidden 3 (64) · ReLU</span></div>
        <FlowArrow label="Dense" />
        <OutputNeurons probabilities={probs} prediction={pred} />
      </div>

      <ConnectionLines hidden1={h1} hidden2={h2} hidden3={h3} probabilities={probs} weights={weights} width={dims.width} height={dims.height} />
      <div style={{ position: "absolute", top: 8, right: 10, zIndex: 3, fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
        Signal: {(signalStrength * 100).toFixed(1)}%
      </div>
      <ActivationLegend />
    </div>
  );
}
