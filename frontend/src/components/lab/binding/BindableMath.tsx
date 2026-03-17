import { useBindingContext } from "./MathVisualBinder";

interface Props {
  latex: string;
  variableId?: string;
}

export function BindableMath({ latex, variableId }: Props) {
  const { enabled, activeVariable, activateFromMath, deactivate } = useBindingContext();
  const active = Boolean(enabled && variableId && activeVariable === variableId);

  return (
    <span
      data-binding-math={variableId}
      onMouseEnter={() => {
        if (enabled && variableId) activateFromMath(variableId);
      }}
      onMouseLeave={() => {
        if (enabled) deactivate();
      }}
      style={{
        cursor: enabled && variableId ? "pointer" : "default",
        color: active ? "var(--fwd)" : "var(--math-text)",
        textDecoration: active ? "underline" : "none",
      }}
      title={enabled && variableId ? `Variable ${variableId}` : undefined}
    >
      {latex}
    </span>
  );
}
