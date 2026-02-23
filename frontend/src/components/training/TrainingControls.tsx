import React from "react";

type Props = {
  connected: boolean;
  status: "idle" | "running" | "paused" | "stopping" | "completed" | "error";
  onConnect: () => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  disabled?: boolean;
};

export default function TrainingControls({ connected, status, onConnect, onStart, onPause, onResume, onStop, disabled = false }: Props) {
  const running = status === "running";
  const paused = status === "paused";

  return (
    <div className="card training-controls">
      <h3>🎛️ Training Controls</h3>
      <div className="runtime-buttons" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {!connected && (
          <button className="btn secondary" onClick={onConnect}>Connect WS</button>
        )}
        {!running && !paused && (
          <button className="btn start" onClick={onStart} disabled={!connected || disabled}>Start</button>
        )}
        {running && (
          <button className="btn pause" onClick={onPause}>Pause</button>
        )}
        {paused && (
          <button className="btn start" onClick={onResume}>Resume</button>
        )}
        {(running || paused || status === "stopping") && (
          <button className="btn stop" onClick={onStop}>Stop</button>
        )}
      </div>
      <div style={{ marginTop: 10 }}>
        <span className={`status-badge ${status}`}>{status.toUpperCase()}</span>
      </div>
    </div>
  );
}
