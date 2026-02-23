import React from "react";

interface Props {
  probabilities: number[];
  prediction: number;
}

const OutputNeurons = React.memo(function OutputNeurons({ probabilities, prediction }: Props) {
  return (
    <div className="layer-block" data-highlight="output">
      <div className="output-neurons">
        {probabilities.map((p, i) => {
          const isWinner = i === prediction;
          const prob = Math.max(0, Math.min(1, p || 0));
          return (
            <div
              key={i}
              className={`output-neuron ${isWinner ? "winner" : ""}`}
              title={`Digit ${i} | Prob: ${(prob * 100).toFixed(1)}%`}
              style={{
                backgroundColor: isWinner
                  ? `rgba(6,182,212,${0.2 + prob * 0.6})`
                  : `rgba(139,92,246,${0.1 + prob * 0.4})`,
              }}
            >
              {i}
            </div>
          );
        })}
      </div>
      <span className="layer-label">Output (10) · Softmax</span>
    </div>
  );
});

export default OutputNeurons;
