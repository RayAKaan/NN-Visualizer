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
    <div className="mt-2 rounded-xl border p-3" style={{ borderColor: "var(--glass-border)", background: "var(--bg-panel)" }}>
      <StageVisualization stage={stage} activation={activation} />
    </div>
  );
}
