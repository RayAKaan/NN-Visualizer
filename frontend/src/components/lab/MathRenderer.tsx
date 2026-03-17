import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface Props {
  latex: string;
  displayMode?: boolean;
}

export function MathRenderer({ latex, displayMode = false }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(latex, ref.current, { throwOnError: false, displayMode });
    } catch {
      ref.current.textContent = latex;
    }
  }, [latex, displayMode]);

  return <div ref={ref} className="overflow-x-auto text-slate-100" />;
}
