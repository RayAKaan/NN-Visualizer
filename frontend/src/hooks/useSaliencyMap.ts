import { useLabStore } from "../store/labStore";

export function useSaliencyMap() {
  const saliencyData = useLabStore((s) => s.saliencyData);
  const computeSaliency = useLabStore((s) => s.computeSaliency);
  return { saliencyData, computeSaliency };
}
