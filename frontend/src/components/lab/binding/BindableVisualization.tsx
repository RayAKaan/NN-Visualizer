import type { ReactNode } from "react";
import { useBindingContext } from "./MathVisualBinder";

interface Props {
  variableId?: string;
  children: ReactNode;
}

export function BindableVisualization({ variableId, children }: Props) {
  const { enabled, activeVariable, activateFromVisual, deactivate } = useBindingContext();
  const active = Boolean(enabled && variableId && activeVariable === variableId);

  return (
    <div
      data-binding-visual={variableId}
      onMouseEnter={() => {
        if (enabled && variableId) activateFromVisual(variableId);
      }}
      onMouseLeave={() => {
        if (enabled) deactivate();
      }}
      style={{
        outline: active ? "2px solid var(--fwd)" : "none",
        borderRadius: 6,
      }}
    >
      {children}
    </div>
  );
}
