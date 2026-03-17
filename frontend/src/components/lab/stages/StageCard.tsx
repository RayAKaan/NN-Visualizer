import type { StageActivation, StageDefinition, StageStatus } from "../../../types/pipeline";
import { StageCardV2 } from "./StageCardV2";

interface Props {
  stage: StageDefinition;
  status: StageStatus;
  activation: StageActivation | null;
  isCurrent: boolean;
  stageNumber: number;
  totalStages: number;
}

export function StageCard(props: Props) {
  return <StageCardV2 {...props} />;
}
