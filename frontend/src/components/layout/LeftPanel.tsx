import CanvasGrid from "../prediction/CanvasGrid";

type Props = { pixels: number[]; onPixelsChange: (p: number[]) => void; brushSize: number; onBrushSizeChange: (n: number) => void; onClear: () => void; onSampleLoad: (d: number) => void; activePixelCount: number };

export default function LeftPanel({ pixels, onPixelsChange, brushSize, onBrushSizeChange, onClear, onSampleLoad, activePixelCount }: Props) {
  return <div className="left-panel"><div className="card"><h3>Input Canvas</h3><CanvasGrid pixels={pixels} onChange={onPixelsChange} brushSize={brushSize} /><div className="row"><span>Brush</span>{[1,2,3].map(s => <button key={s} className={brushSize===s?"active":""} onClick={() => onBrushSizeChange(s)}>{s}</button>)}</div><div>Active pixels: {activePixelCount}</div><button className="btn secondary block" onClick={onClear}>Clear</button><hr /><h4>Quick Test</h4><div className="digit-grid">{Array.from({length:10}).map((_,i)=><button key={i} className="sample-btn" onClick={()=>onSampleLoad(i)}>{i}</button>)}</div></div></div>;
}
