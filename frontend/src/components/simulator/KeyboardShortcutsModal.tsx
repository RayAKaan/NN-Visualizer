import React from "react";
import { NeuralModal } from "@/design-system/components/NeuralModal";
import { NeuralButton } from "@/design-system/components/NeuralButton";

interface ShortcutItem {
  key: string;
  description: string;
}

const sections: Array<{ title: string; items: ShortcutItem[] }> = [
  {
    title: "Navigation",
    items: [
      { key: "1-9", description: "Switch views" },
      { key: "0", description: "Embeddings view" },
      { key: "?", description: "Open shortcuts" },
    ],
  },
  {
    title: "Training",
    items: [
      { key: "Space", description: "Start / pause training" },
      { key: "S", description: "Stop training" },
    ],
  },
  {
    title: "Forward / Backward",
    items: [
      { key: "F", description: "Run forward pass" },
      { key: "B", description: "Run backward pass" },
    ],
  },
  {
    title: "Architecture",
    items: [
      { key: "A", description: "Add hidden layer" },
      { key: "D", description: "Delete selected layer" },
    ],
  },
];

export function KeyboardShortcutsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <NeuralModal open={open} onClose={onClose}>
      <div className="shortcuts-modal">
        <div className="shortcuts-header">
          <div>
            <div className="shortcuts-title">Keyboard Shortcuts</div>
            <div className="shortcuts-subtitle">Power user navigation</div>
          </div>
          <NeuralButton variant="ghost" onClick={onClose}>
            X
          </NeuralButton>
        </div>
        <div className="shortcuts-grid">
          {sections.map((section) => (
            <div key={section.title} className="shortcuts-section">
              <div className="shortcuts-section-title">{section.title}</div>
              {section.items.map((item) => (
                <div key={item.key} className="shortcuts-row">
                  <kbd className="shortcuts-kbd">{item.key}</kbd>
                  <span>{item.description}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </NeuralModal>
  );
}
