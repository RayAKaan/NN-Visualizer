import { useEffect, useRef } from "react";

export function useAutoScroll(currentStageIndex: number) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (currentStageIndex < 0 || !containerRef.current) return;
    const nodes = containerRef.current.querySelectorAll('[data-stage-item="true"]');
    const current = nodes[currentStageIndex];
    if (!(current instanceof HTMLElement)) return;
    const timer = window.setTimeout(() => {
      current.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [currentStageIndex]);

  return containerRef;
}
