import type { Architecture, Dataset, StageActivation, StageDefinition } from "../../../../types/pipeline";
import { Conv2DTransformation } from "./transformations/Conv2DTransformation";
import { DenseTransformation } from "./transformations/DenseTransformation";
import { FlattenTransformation } from "./transformations/FlattenTransformation";
import { InputTransformation } from "./transformations/InputTransformation";
import { LSTMTransformation } from "./transformations/LSTMTransformation";
import { MaxPoolTransformation } from "./transformations/MaxPoolTransformation";
import { PreprocessTransformation } from "./transformations/PreprocessTransformation";
import { ReLUTransformation } from "./transformations/ReLUTransformation";
import { SoftmaxTransformation } from "./transformations/SoftmaxTransformation";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  prevActivation: StageActivation | null;
  architecture: Architecture;
  dataset: Dataset;
  highlightedVariable: string | null;
  onHighlightVariable: (v: string | null) => void;
}

export function TheTransformation(props: Props) {
  const { stage } = props;
  return (
    <section className="overflow-hidden rounded-xl border" style={{ borderColor: "var(--glass-border)", background: "var(--bg-void)" }}>
      <div className="border-b px-4 py-2 text-xs font-semibold uppercase" style={{ borderColor: "var(--glass-border)", color: "var(--fwd)" }}>
        The Transformation
      </div>
      <div className="p-4">
        {stage.type === "input" ? <InputTransformation {...props} /> : null}
        {stage.type === "preprocessing" ? <PreprocessTransformation {...props} /> : null}
        {stage.type === "dense" ? <DenseTransformation {...props} /> : null}
        {stage.type === "activation_relu" ? <ReLUTransformation {...props} /> : null}
        {stage.type === "conv2d" ? <Conv2DTransformation {...props} /> : null}
        {stage.type === "max_pool" ? <MaxPoolTransformation {...props} /> : null}
        {stage.type === "flatten" ? <FlattenTransformation {...props} /> : null}
        {stage.type === "softmax" ? <SoftmaxTransformation {...props} /> : null}
        {stage.type === "lstm_cell" ? <LSTMTransformation {...props} /> : null}
        {stage.type === "output" ? <SoftmaxTransformation {...props} /> : null}
      </div>
    </section>
  );
}
