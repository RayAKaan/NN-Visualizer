import React, { useState, useEffect, useRef, useCallback } from "react";
import { BatchUpdate } from "../../types";

interface Props {
  history: BatchUpdate[];
  onSnapshotChange: (snapshot: BatchUpdate) => void;
}

export default function TrainingReplay({ history, onSnapshotChange }: Props) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef<number>(0);

  const maxIndex = Math.max(0, history.length - 1);

  useEffect(() => {
    if (!playing || history.length === 0) return;

    const interval = 50 / speed;

    const tick = (timestamp: number) => {
      if (lastTimeRef.current === 0) lastTimeRef.current = timestamp;
      const delta = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;
      accumulatorRef.current += delta;

      if (accumulatorRef.current >= interval) {
        accumulatorRef.current -= interval;
        setIndex((prev) => {
          const next = prev + 1;
          if (next > maxIndex) {
            setPlaying(false);
            return prev;
          }
          if (history[next]) {
            onSnapshotChange(history[next]);
          }
          return next;
        });
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, history, maxIndex, onSnapshotChange]);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const i = parseInt(e.target.value, 10);
      setIndex(i);
      if (history[i]) {
        onSnapshotChange(history[i]);
      }
    },
    [history, onSnapshotChange]
  );

  const togglePlay = () => setPlaying((p) => !p);

  const currentSnapshot = history[index];

  if (history.length === 0) {
    return (
      <div className="timeline-container" style={{ justifyContent: "center" }}>
        <span className="timeline-info">No training history available</span>
      </div>
    );
  }

  return (
    <div className="timeline-container">
      <button className="btn btn-sm btn-secondary" onClick={togglePlay}>
        {playing ? "⏸" : "▶"}
      </button>

      <input
        type="range"
        className="timeline-slider"
        min={0}
        max={maxIndex}
        value={index}
        onChange={handleSliderChange}
      />

      <div style={{ display: "flex", gap: 4 }}>
        {[0.5, 1, 2, 5, 10].map((s) => (
          <button
            key={s}
            className={`btn btn-sm ${speed === s ? "btn-primary" : "btn-secondary"}`}
            onClick={() => setSpeed(s)}
            style={{ padding: "2px 6px", fontSize: 10 }}
          >
            {s}x
          </button>
        ))}
      </div>

      <span className="timeline-info">
        {currentSnapshot
          ? `E${currentSnapshot.epoch} B${currentSnapshot.batch} | ${index + 1}/${history.length}`
          : `${index + 1}/${history.length}`}
      </span>
    </div>
  );
}
