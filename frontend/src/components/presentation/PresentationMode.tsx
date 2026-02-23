export default function PresentationMode({ active, onExit }: { active: boolean; onExit: () => void; onAction: (action: string) => void }) {
  if (!active) return null;
  return <div className="presentation-overlay"><div className="presentation-panel"><h3>Presentation Mode — Coming in Phase 4</h3><button className="btn" onClick={onExit}>Close</button></div></div>;
}
