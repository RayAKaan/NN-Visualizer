export default function ShortcutHelp({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  if (!visible) return null;
  return <div className="modal-overlay"><div className="modal-content"><h3>Keyboard Shortcuts — Phase 4</h3><button className="btn" onClick={onClose}>Close</button></div></div>;
}
