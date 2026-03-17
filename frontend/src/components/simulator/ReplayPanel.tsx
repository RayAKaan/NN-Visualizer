import React, { useEffect } from "react";
import { useReplayStore } from "../../store/replayStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function ReplayPanel() {
  const snapshots = useReplayStore((s) => s.snapshots);
  const loadSnapshots = useReplayStore((s) => s.loadSnapshots);
  const loadSnapshot = useReplayStore((s) => s.loadSnapshot);
  const currentIndex = useReplayStore((s) => s.currentSnapshotIndex);

  useEffect(() => {
    void loadSnapshots();
  }, [loadSnapshots]);

  return (
    <NeuralPanel className="replay-panel" variant="base">
      <div className="replay-title">Replay</div>
      <div className="replay-list neural-scroll-area">
        {snapshots.map((s: any, idx: number) => (
          <NeuralButton
            key={s.index ?? idx}
            variant={currentIndex === idx ? "primary" : "secondary"}
            className="replay-chip"
            onClick={() => void loadSnapshot(idx)}
          >
            E{s.epoch ?? idx}
          </NeuralButton>
        ))}
      </div>
      {snapshots.length === 0 && <div className="replay-empty">No snapshots yet.</div>}
    </NeuralPanel>
  );
}
