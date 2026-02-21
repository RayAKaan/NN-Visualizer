import React from "react";
import CanvasGrid from "../prediction/CanvasGrid";
import { sampleDigits } from "../../utils/sampleDigits";

interface Props {
  pixels: number[];
  onPixelsChange: (pixels: number[]) => void;
  brushSize: 1 | 2 | 3;
  onBrushSizeChange: (size: 1 | 2 | 3) => void;
}

const LeftPanel: React.FC<Props> = ({ pixels, onPixelsChange, brushSize, onBrushSizeChange }) => {
  const active = pixels.filter((v) => v > 0.05).length;
  return (
    <aside className="card">
      <h3>Input Canvas</h3>
      <CanvasGrid pixels={pixels} onChange={onPixelsChange} brushSize={brushSize} />
      <div className="control-row">
        <button className="btn btn-secondary" onClick={() => onPixelsChange(Array.from({ length: 784 }, () => 0))}>Clear</button>
        <select value={brushSize} onChange={(e) => onBrushSizeChange(Number(e.target.value) as 1 | 2 | 3)}>
          <option value={1}>Brush 1px</option>
          <option value={2}>Brush 2px</option>
          <option value={3}>Brush 3px</option>
        </select>
      </div>
      <p className="muted">Active pixels: {active}/784</p>
      <div className="sample-row">
        {Object.keys(sampleDigits).map((digit) => (
          <button key={digit} className="btn btn-secondary sample-btn" onClick={() => onPixelsChange(sampleDigits[digit])}>{digit}</button>
        ))}
      </div>
    </aside>
  );
};

export default LeftPanel;
