import React from "react";

interface TrainingTimelineProps {
  value: number;
  max: number;
  onChange: (next: number) => void;
  onReplay: () => void;
}

const TrainingTimeline: React.FC<TrainingTimelineProps> = ({ value, max, onChange, onReplay }) => {
  return (
    <div className="timeline">
      <div className="timeline-header">
        <strong>Training timeline</strong>
        <button type="button" onClick={onReplay}>
          Replay live
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={Math.max(0, max)}
        value={Math.min(value, Math.max(0, max))}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span>
        Step {value + 1} / {Math.max(1, max + 1)}
      </span>
    </div>
  );
};

export default TrainingTimeline;
