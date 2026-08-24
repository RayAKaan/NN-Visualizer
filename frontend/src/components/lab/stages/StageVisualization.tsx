import { useLabStore } from "../../../store/labStore";
import type { StageActivation, StageDefinition } from "../../../types/pipeline";
import { DenseLayerViz } from "./ann/DenseLayerViz";
import { ActivationViz } from "./ann/ActivationViz";
import { ConvolutionViz } from "./cnn/ConvolutionViz";
import { PoolingViz } from "./cnn/PoolingViz";
import { FlattenViz } from "./cnn/FlattenViz";
import { RecurrentCellViz } from "./rnn/RecurrentCellViz";
import { SoftmaxViz } from "./shared/SoftmaxViz";
import { PreprocessingViz } from "./shared/PreprocessingViz";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
}

export function StageVisualization({ stage, activation }: Props) {
  const dataset = useLabStore((s) => s.dataset);

  return (
    <div className="rounded-xl border border-barley-linestrong bg-ink/45 p-3">
      <div className="mb-2 text-[12px] font-semibold uppercase tracking-wide text-ember-700">Visualization</div>
      {stage.type === "preprocessing" && <PreprocessingViz activation={activation} dataset={dataset} />}
      {stage.type === "dense" && <DenseLayerViz activation={activation} stage={stage} />}
      {stage.type === "activation_relu" && <ActivationViz activation={activation} />}
      {stage.type === "conv2d" && <ConvolutionViz activation={activation} stage={stage} dataset={dataset} />}
      {stage.type === "max_pool" && <PoolingViz activation={activation} />}
      {stage.type === "flatten" && <FlattenViz activation={activation} />}
      {stage.type === "softmax" && <SoftmaxViz activation={activation} stage={stage} dataset={dataset} />}
      {stage.type === "lstm_cell" && <RecurrentCellViz activation={activation} stage={stage} />}
    </div>
  );
}
