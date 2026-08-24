import { StageVisualization } from "../StageVisualization";
import type { Architecture, Dataset, StageActivation, StageDefinition } from "../../../../types/pipeline";

interface Props {
  stage: StageDefinition;
  activation: StageActivation;
  architecture: Architecture;
  dataset: Dataset;
}

export function DeepDive({ stage, activation }: Props) {
  return (
    <div className="mt-2 rounded-xl border border-barley-linestrong bg-white p-3">
      <StageVisualization stage={stage} activation={activation} />
    </div>
  );
}
