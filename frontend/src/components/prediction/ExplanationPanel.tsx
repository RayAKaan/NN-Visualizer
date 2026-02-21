import React, { useState } from "react";
import { ExplanationResult } from "../../types";

const ExplanationPanel: React.FC<{ explanation: ExplanationResult | null }> = ({ explanation }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="card">
      <button className="btn btn-secondary" onClick={() => setOpen((v) => !v)}>
        {open ? "Hide" : "Show"} Explanation
      </button>
      {open && explanation && (
        <div className="explanation-block">
          <p>Confidence level: <strong>{explanation.confidence_level}</strong></p>
          <p>Evidence:</p>
          <ul>{explanation.evidence.map((item) => <li key={item}>{item}</li>)}</ul>
          <p>Top neurons:</p>
          <ul>{explanation.top_neurons.map((n) => <li key={`${n.layer}-${n.index}`}>{n.layer} #{n.index} · {n.contribution.toFixed(3)}</li>)}</ul>
          <p>Competitors:</p>
          <ul>{explanation.competitors.map((c) => <li key={c.digit}>{c.digit} · {(c.probability * 100).toFixed(1)}%</li>)}</ul>
          {explanation.uncertainty_notes.length > 0 && <ul>{explanation.uncertainty_notes.map((u) => <li key={u}>{u}</li>)}</ul>}
        </div>
      )}
    </div>
  );
};

export default ExplanationPanel;
