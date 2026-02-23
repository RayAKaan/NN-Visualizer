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
          const prob = Math.max(0, Math.min(1, p || 0));
          const isWinner = i === prediction;
          return (
            <div
              key={i}
              className={`output-neuron ${isWinner ? "winner" : ""}`}
              title={`Digit ${i} | Prob: ${(prob * 100).toFixed(1)}%`}
              style={{
                width: 36,
                height: 44,
                display: "grid",
                placeItems: "center",
                borderRadius: 8,
                transform: isWinner ? "scale(1.1)" : "scale(1)",
                backgroundColor: isWinner
                  ? `rgba(6,182,212,${0.15 + prob * 0.6})`
                  : `rgba(139,92,246,${0.08 + prob * 0.35})`,
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
