import { useLabStore } from "../store/labStore";

export function useWeightInspection() {
  const inspectedStageId = useLabStore((s) => s.inspectedStageId);
  const weightInspection = useLabStore((s) => s.weightInspection);
  const inspectStageWeights = useLabStore((s) => s.inspectStageWeights);
  const clearWeightInspection = useLabStore((s) => s.clearWeightInspection);
  return { inspectedStageId, weightInspection, inspectStageWeights, clearWeightInspection };
}
