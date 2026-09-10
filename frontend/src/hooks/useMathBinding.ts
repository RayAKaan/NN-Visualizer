import { useBindingStore } from "../store/bindingStore";

export function useMathBinding() {
  return {
    enabled: useBindingStore((s) => s.enabled),
    activeVariable: useBindingStore((s) => s.activeVariable),
    toggleEnabled: useBindingStore((s) => s.toggleEnabled),
    activateFromMath: useBindingStore((s) => s.activateFromMath),
    activateFromVisual: useBindingStore((s) => s.activateFromVisual),
    deactivate: useBindingStore((s) => s.deactivate),
  };
}
