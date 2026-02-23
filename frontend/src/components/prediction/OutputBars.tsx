import React from "react";

function OutputBars({ probabilities, winner }: { probabilities: number[]; winner: number }) {
  return <div className="card">{Array.from({ length: 10 }).map((_, i) => <div className="bar-row" key={i}><span>{i}</span><div className="bar-track"><div className={`bar-fill ${winner === i ? "winner" : ""}`} style={{ width: `${((probabilities[i] || 0) * 100).toFixed(1)}%` }} /></div><span>{((probabilities[i] || 0) * 100).toFixed(1)}%</span></div>)}</div>;
}

export default React.memo(OutputBars);
