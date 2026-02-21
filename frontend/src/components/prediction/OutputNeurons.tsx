import React from "react";

interface Props {
  output: number[];
  winner: number;
}

const OutputNeurons: React.FC<Props> = ({ output, winner }) => {
  return (
    <div className="output-neurons">
      {output.map((v, i) => (
        <div key={i} className={`output-neuron ${winner === i ? "winner" : ""}`} style={{ opacity: 0.35 + v * 0.65 }}>
          <span>{i}</span>
          <strong>{(v * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
};

export default OutputNeurons;
