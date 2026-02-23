import React from "react";

interface Props {
  connected: boolean;
  modelType: string;
  inferenceTime: number;
  isCanvasEmpty: boolean;
}

const StatusBar: React.FC<Props> = ({ connected, modelType, inferenceTime, isCanvasEmpty }) => {
  return (
    <footer className="status-bar">
      <span className={`dot ${connected ? "ok" : "bad"}`} /> {connected ? "Connected" : "Disconnected"}
      <span>{modelType.toUpperCase()}</span>
      <span>Inference: {inferenceTime ? `${inferenceTime.toFixed(1)}ms` : "--"}</span>
      {isCanvasEmpty && <span>Draw a digit to begin</span>}
    </footer>
  );
};

export default StatusBar;
