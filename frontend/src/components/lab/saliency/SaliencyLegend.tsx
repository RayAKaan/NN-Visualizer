export function SaliencyLegend() {
  return (
    <div>
      <div className="mb-1 text-xs text-ink-mute">Saliency scale</div>
      <div className="h-3 rounded" style={{ background: "linear-gradient(to right, transparent, rgba(60,80,200,0.5), rgba(200,70,180,0.65), rgba(250,160,60,0.8), rgba(255,255,120,0.9))" }} />
      <div className="mt-1 flex justify-between text-[12px] text-ink-faint">
        <span>Low</span><span>High</span>
      </div>
    </div>
  );
}
