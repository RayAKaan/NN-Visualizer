import { createContext, useContext, useMemo } from "react";
import type { ReactNode } from "react";
import { useBindingStore } from "../../../store/bindingStore";

interface BindingApi {
  activeVariable: string | null;
  enabled: boolean;
  activateFromMath: (id: string) => void;
  activateFromVisual: (id: string) => void;
  deactivate: () => void;
}

const Ctx = createContext<BindingApi | null>(null);

export function MathVisualBinder({ children }: { children: ReactNode }) {
  const activeVariable = useBindingStore((s) => s.activeVariable);
  const enabled = useBindingStore((s) => s.enabled);
  const activateFromMath = useBindingStore((s) => s.activateFromMath);
  const activateFromVisual = useBindingStore((s) => s.activateFromVisual);
  const deactivate = useBindingStore((s) => s.deactivate);

  const value = useMemo(
    () => ({ activeVariable, enabled, activateFromMath, activateFromVisual, deactivate }),
    [activeVariable, enabled, activateFromMath, activateFromVisual, deactivate],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBindingContext() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useBindingContext must be used inside MathVisualBinder");
  return ctx;
}
