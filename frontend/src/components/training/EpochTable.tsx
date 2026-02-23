import React from "react";
import { EpochUpdate } from "../../types";

interface Props {
  epochs: EpochUpdate[];
}

export default function EpochTable({ epochs }: Props) {
  if (epochs.length === 0) return null;

  return (
    <div className="card">
      <div className="card-title">Epoch Summary</div>
      <div style={{ overflowX: "auto" }}>
        <table className="epoch-table">
          <thead>
            <tr>
              <th>Epoch</th>
              <th>Loss</th>
              <th>Acc</th>
              <th>Val Loss</th>
              <th>Val Acc</th>
            </tr>
          </thead>
          <tbody>
            {epochs.map((ep, i) => (
              <tr key={i}>
                <td>{ep.epoch}</td>
                <td>{ep.loss.toFixed(4)}</td>
                <td>{(ep.accuracy * 100).toFixed(1)}%</td>
                <td>{ep.val_loss.toFixed(4)}</td>
                <td
                  style={{
                    color:
                      ep.val_accuracy > 0.95
                        ? "var(--accent-green)"
                        : ep.val_accuracy > 0.9
                        ? "var(--accent-cyan)"
                        : "var(--text-secondary)",
                    fontWeight: 600,
                  }}
                >
                  {(ep.val_accuracy * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
