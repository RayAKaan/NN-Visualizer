import { useLabStore } from "../store/labStore";

export function useBackwardPass() {
  const startBackwardPass = useLabStore((s) => s.startBackwardPass);
  const stepBackward = useLabStore((s) => s.stepBackward_bwd);
  const stepForward = useLabStore((s) => s.stepForward_bwd);
  const skipToEnd = useLabStore((s) => s.skipToEnd_bwd);
  const reset = useLabStore((s) => s.resetBackwardPass);
  return { startBackwardPass, stepBackward, stepForward, skipToEnd, reset };
}
