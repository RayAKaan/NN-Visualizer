import { useEffect, useRef } from "react";
import type { MutableRefObject, ReactNode } from "react";

interface Props {
  scrollRefs: MutableRefObject<Record<"ANN" | "CNN" | "RNN", HTMLDivElement | null>>;
  children: ReactNode;
}

export function ComparisonSyncScroll({ scrollRefs, children }: Props) {
  const syncing = useRef(false);

  useEffect(() => {
    const archs: Array<"ANN" | "CNN" | "RNN"> = ["ANN", "CNN", "RNN"];
    const handlers: Array<{ el: HTMLDivElement; fn: () => void }> = [];

    const wire = (source: "ANN" | "CNN" | "RNN") => {
      const el = scrollRefs.current[source];
      if (!el) return;
      const fn = () => {
        if (syncing.current) return;
        syncing.current = true;
        const denom = Math.max(1, el.scrollHeight - el.clientHeight);
        const pct = el.scrollTop / denom;

        for (const a of archs) {
          if (a === source) continue;
          const target = scrollRefs.current[a];
          if (!target) continue;
          target.scrollTop = pct * Math.max(1, target.scrollHeight - target.clientHeight);
        }

        requestAnimationFrame(() => {
          syncing.current = false;
        });
      };
      el.addEventListener("scroll", fn, { passive: true });
      handlers.push({ el, fn });
    };

    wire("ANN");
    wire("CNN");
    wire("RNN");

    return () => {
      for (const h of handlers) h.el.removeEventListener("scroll", h.fn);
    };
  }, [scrollRefs]);

  return <>{children}</>;
}
