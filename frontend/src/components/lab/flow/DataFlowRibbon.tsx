import { useMemo } from "react";
import { useDataFlow } from "../../../hooks/useDataFlow";
import { useLabStore } from "../../../store/labStore";
import { FlowDimensionLabel } from "./FlowDimensionLabel";
import { FlowMorphTransition } from "./FlowMorphTransition";
import { FlowThumbnail } from "./FlowThumbnail";

export function DataFlowRibbon() {
  const stages = useLabStore((s) => s.stages);
  const activations = useLabStore((s) => s.activations);
  const stageStatuses = useLabStore((s) => s.stageStatuses);
  const currentStageIndex = useLabStore((s) => s.currentStageIndex);

  const flowSnapshots = useDataFlow(stages, activations);
  const visible = useMemo(
    () => flowSnapshots.filter((f) => stageStatuses[f.stageId] === "completed" || stageStatuses[f.stageId] === "active"),
    [flowSnapshots, stageStatuses],
  );

  if (visible.length === 0) return null;

  return (
    <aside className="lab-flow-ribbon" aria-label="Data flow snapshots">
      <div className="lab-flow-ribbon-heading">
        <span>Flow</span>
        <small>Completed activations</small>
      </div>
      <div className="lab-flow-ribbon-list">
      {visible.map((snapshot, i) => {
        const prev = i > 0 ? visible[i - 1] : null;
        const isActive = stages[currentStageIndex]?.id === snapshot.stageId;
        return (
          <div key={snapshot.stageId} className="flex flex-col items-center">
            {prev ? <FlowMorphTransition from={prev} to={snapshot} isActive={isActive} /> : null}
            <FlowThumbnail
              snapshot={snapshot}
              isActive={isActive}
              onClick={() => {
                const el = document.querySelector(`[data-tutorial-id="stage-${snapshot.stageId}"]`);
                if (el instanceof HTMLElement) el.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
            />
            <FlowDimensionLabel shape={snapshot.shape} statistics={snapshot.statistics} isActive={isActive} />
          </div>
        );
      })}
      </div>
    </aside>
  );
}
