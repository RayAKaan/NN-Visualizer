import { useMemo } from "react";
import type { Architecture, Dataset, StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  architecture: Architecture;
  dataset: Dataset;
}

export function KeyInsight({ stage, activation, dataset }: Props) {
  const data = activation.outputData;
  const insight = useMemo(() => {
    if (stage.type === "activation_relu") {
      let zeros = 0;
      for (let i = 0; i < data.length; i += 1) if (data[i] === 0) zeros += 1;
      const rate = (zeros / Math.max(1, data.length)) * 100;
      return {
        icon: rate > 65 ? "??" : "?",
        severity: rate > 65 ? "var(--warning)" : "var(--success)",
        text: `${rate.toFixed(1)}% of activations were zeroed. ${rate > 65 ? "Strong filtering." : "Healthy sparsity."}`,
      };
    }
    if (stage.type === "softmax") {
      const probs = Array.from(data);
      const top = probs.reduce((best, v, i, arr) => (v > arr[best] ? i : best), 0);
      const conf = probs[top] * 100;
      return {
        icon: conf > 90 ? "??" : "??",
        severity: conf > 90 ? "var(--success)" : "var(--info)",
        text: `Top class is ${dataset === "catdog" ? (top === 0 ? "Cat" : "Dog") : top} at ${conf.toFixed(1)}% confidence.`,
      };
    }
    if (stage.type === "conv2d") {
      return {
        icon: "??",
        severity: "var(--arch-cnn)",
        text: `Convolution created ${activation.metadata.outputShape[0] ?? "multiple"} feature maps that highlight learned spatial patterns.`,
      };
    }
    return {
      icon: "??",
      severity: "var(--info)",
      text: `Stage transformed shape ${JSON.stringify(activation.metadata.inputShape)} ? ${JSON.stringify(activation.metadata.outputShape)}.`,
    };
  }, [activation.metadata.inputShape, activation.metadata.outputShape, data, dataset, stage.type]);

  return (
    <section className="flex items-start gap-2 rounded-xl border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <span className="text-lg">{insight.icon}</span>
      <div>
        <div className="text-xs font-semibold uppercase" style={{ color: insight.severity }}>Key Insight</div>
        <p className="text-xs" style={{ color: "var(--text-2)" }}>{insight.text}</p>
      </div>
    </section>
  );
}
