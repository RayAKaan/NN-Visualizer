import React, { useEffect, useMemo, useRef, useState } from "react";
import { TrainingHistorySnapshot } from "../../hooks/useTrainingHistory";

interface TrainingReplayProps {
  history: TrainingHistorySnapshot[];
  onSnapshotChange: (snapshot: TrainingHistorySnapshot) => void;
}

const TrainingReplay: React.FC<TrainingReplayProps> = ({ history, onSnapshotChange }) => {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef(0);

  const maxIndex = useMemo(() => Math.max(0, history.length - 1), [history.length]);

  useEffect(() => {
    if (index > maxIndex) {
      setIndex(maxIndex);
    }
    if (history.length === 0) {
      setPlaying(false);
    }
  }, [history.length, index, maxIndex]);

  useEffect(() => {
    if (!playing || history.length === 0) {
      return;
    }

    const frame = (ts: number) => {
      if (ts - lastTickRef.current >= 100 / speed) {
        lastTickRef.current = ts;
        setIndex((prev) => {
          const next = Math.min(maxIndex, prev + 1);
          const snapshot = history[next];
          if (snapshot) {
            onSnapshotChange(snapshot);
          }
          if (next >= maxIndex) {
            setPlaying(false);
          }
          return next;
        });
      }
      if (playing) {
        rafRef.current = requestAnimationFrame(frame);
      }
    };

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [playing, speed, history, maxIndex, onSnapshotChange]);

  return (
    <div className="training-replay">
      <div className="replay-controls">
        <button
          type="button"
          disabled={history.length === 0}
          onClick={() => setPlaying((p) => (history.length === 0 ? false : !p))}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <label>
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {[0.5, 1, 2, 4].map((s) => (
              <option key={s} value={s}>
                {s}x
              </option>
            ))}
          </select>
        </label>
      </div>

      <input
        type="range"
        min={0}
        max={maxIndex}
        value={index}
        onChange={(e) => {
          const i = Number(e.target.value);
          setIndex(i);
          const snapshot = history[i];
          if (snapshot) {
            onSnapshotChange(snapshot);
          }
        }}
      />

      <div className="replay-meta">Batch {index + 1} / {maxIndex + 1}</div>
    </div>
  );
};

export default TrainingReplay;
