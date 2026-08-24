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
        icon: rate > 65 ? "\u26a0\ufe0f" : "\u2705",
        severity: rate > 65 ? "text-status-warning" : "text-status-success",
        text: `${rate.toFixed(1)}% of activations were zeroed. ${rate > 65 ? "Strong filtering." : "Healthy sparsity."}`,
      };
    }
    if (stage.type === "softmax") {
      const probs = Array.from(data);
      const top = probs.reduce((best, v, i, arr) => (v > arr[best] ? i : best), 0);
      const conf = probs[top] * 100;
      return {
        icon: conf > 90 ? "\ud83c\udfaf" : "\ud83e\udd14",
        severity: conf > 90 ? "text-status-success" : "text-arch-ann",
        text: `Top class is ${dataset === "catdog" ? (top === 0 ? "Cat" : "Dog") : top} at ${conf.toFixed(1)}% confidence.`,
      };
    }
    if (stage.type === "conv2d") {
      return {
        icon: "\ud83d\udd0d",
        severity: "text-arch-cnn",
        text: `Convolution created ${activation.metadata.outputShape[0] ?? "multiple"} feature maps that highlight learned spatial patterns.`,
      };
    }
    return {
      icon: "\ud83d\udca1",
      severity: "text-arch-ann",
      text: `Stage transformed shape ${JSON.stringify(activation.metadata.inputShape)} \u2192 ${JSON.stringify(activation.metadata.outputShape)}.`,
    };
  }, [activation.metadata.inputShape, activation.metadata.outputShape, data, dataset, stage.type]);

  return (
    <section className="flex items-start gap-2 rounded-xl border border-barley-linestrong bg-white p-3">
      <span className="text-lg">{insight.icon}</span>
      <div>
        <div className={`text-xs font-semibold uppercase ${insight.severity}`}>Key Insight</div>
        <p className="text-xs text-ink-soft">{insight.text}</p>
      </div>
    </section>
  );
}
