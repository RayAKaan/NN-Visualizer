import React from "react";

const OutputBars: React.FC<{ probabilities: number[]; winner: number }> = ({ probabilities, winner }) => {
  return (
    <div className="card">
      <h3>Output Probabilities</h3>
      {probabilities.map((value, index) => (
        <div key={index} className="output-row-item">
          <span>{index}</span>
          <div className={`output-bar ${winner === index ? "winner" : ""}`} style={{ width: `${Math.max(4, value * 100)}%` }} />
          <span>{(value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

export default OutputBars;
