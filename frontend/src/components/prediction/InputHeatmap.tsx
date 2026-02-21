import React from "react";

const InputHeatmap: React.FC<{ pixels: number[] }> = ({ pixels }) => {
  return (
    <div className="input-heatmap">
      {pixels.map((v, idx) => (
        <div key={idx} style={{ opacity: Math.max(0.08, Math.min(1, v)) }} />
      ))}
    </div>
  );
};

export default InputHeatmap;
