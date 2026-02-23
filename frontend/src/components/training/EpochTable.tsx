import React from "react";

type Props = {
  losses: number[];
  accuracies: number[];
  valLosses: number[];
  valAccuracies: number[];
};

export default function EpochTable({ losses, accuracies, valLosses, valAccuracies }: Props) {
  const rows = Math.max(valLosses.length, valAccuracies.length);

  return (
    <div className="card epoch-table">
      <h3>📋 Epoch Summary</h3>
      <table style={{ width: "100%", fontSize: 12 }}>
        <thead>
          <tr><th>#</th><th>Loss</th><th>Acc</th><th>Val Loss</th><th>Val Acc</th></tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }).map((_, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{(losses[i] ?? 0).toFixed(4)}</td>
              <td>{((accuracies[i] ?? 0) * 100).toFixed(2)}%</td>
              <td>{(valLosses[i] ?? 0).toFixed(4)}</td>
              <td>{((valAccuracies[i] ?? 0) * 100).toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
