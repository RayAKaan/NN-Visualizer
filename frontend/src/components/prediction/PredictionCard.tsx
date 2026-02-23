import React from "react";

const PredictionCard: React.FC<{ digit: number; confidence: number }> = ({ digit, confidence }) => {
  const levelClass = confidence > 0.9 ? "good" : confidence > 0.7 ? "warn" : "bad";
  return (
    <div className="card prediction-card">
      <div className="pred-digit">{digit}</div>
      <div className={`pred-confidence ${levelClass}`}>{(confidence * 100).toFixed(1)}%</div>
      <div className="confidence-ring">
        <div style={{ width: `${Math.max(8, confidence * 100)}%` }} />
      </div>
    </div>
  );
};

export default PredictionCard;
