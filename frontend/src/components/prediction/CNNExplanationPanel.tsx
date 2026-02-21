import React from "react";
import { CNNExplanationResult } from "../../types";

const CNNExplanationPanel: React.FC<{ explanation: CNNExplanationResult | null }> = ({ explanation }) => {
  if (!explanation) return <div className="card">No CNN explanation yet.</div>;
  return (
    <div className="card">
      <h3>CNN Explanation</h3>
      <div>
        {explanation.active_filters.slice(0, 10).map((f) => (
          <div key={`${f.layer}-${f.filter_index}`} className="active-filter-item">
            <span>{f.layer} · f{f.filter_index}</span>
            <div className="active-filter-bar">
              <div className="active-filter-bar-fill" style={{ width: `${Math.min(100, f.mean_activation * 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
      <p>Spatial evidence:</p>
      <ul>{explanation.spatial_evidence.map((s) => <li key={s}>{s}</li>)}</ul>
      <p>Competitors:</p>
      <ul>{explanation.competitors.map((c) => <li key={c.digit}>{c.digit} · {(c.probability * 100).toFixed(1)}% · {c.reason}</li>)}</ul>
      {explanation.uncertainty_notes.length > 0 && <ul>{explanation.uncertainty_notes.map((u) => <li key={u}>{u}</li>)}</ul>}
    </div>
  );
};

export default CNNExplanationPanel;
