import React from "react";

interface OutputPanelProps {
  probabilities: number[];
}

const OutputPanel: React.FC<OutputPanelProps> = ({ probabilities }) => {
  const maxIndex = probabilities.reduce(
    (max, value, index) => (value > probabilities[max] ? index : max),
    0
  );

  return (
    <div className="output-panel">
      {probabilities.map((value, index) => (
        <div key={`output-${index}`} className="output-bar">
          <span>{index}</span>
          <div className="bar-track">
            <div
              className={`bar-fill ${index === maxIndex ? "active" : ""}`}
              style={{ width: `${(value * 100).toFixed(1)}%` }}
            />
          </div>
          <span>{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

export default OutputPanel;
