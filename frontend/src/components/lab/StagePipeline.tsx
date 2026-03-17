import { useLabStore } from "../../store/labStore";
import { StageCard } from "./stages/StageCard";
import { StageConnector } from "./StageConnector";

export function StagePipeline() {
  const stages = useLabStore((s) => s.stages);
  const statuses = useLabStore((s) => s.stageStatuses);
  const current = useLabStore((s) => s.currentStageIndex);
  const activations = useLabStore((s) => s.activations);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col" role="list" aria-label="Neural pipeline stages">
      {stages.map((stage, index) => (
        <div key={stage.id} data-stage-item="true" role="listitem" className="w-full">
          {index > 0 && (
            <StageConnector
              fromStatus={statuses[stages[index - 1].id]}
              toStatus={statuses[stage.id]}
            />
          )}
          <StageCard
            stage={stage}
            status={statuses[stage.id] ?? "locked"}
            activation={activations[stage.id] ?? null}
            isCurrent={index === current}
            stageNumber={index + 1}
            totalStages={stages.length}
          />
        </div>
      ))}
    </div>
  );
}
