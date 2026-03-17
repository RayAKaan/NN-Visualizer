import { useMemo, useState } from "react";
import { useLabStore } from "../../../store/labStore";
import type { SaliencyData } from "../../../types/pipeline";
import { renderSaliencyOverlay } from "../../../utils/colorRamps";
import { SaliencyLegend } from "./SaliencyLegend";

interface Props {
  saliencyData: SaliencyData;
}

export function SaliencyOverlay({ saliencyData }: Props) {
  const [opacity, setOpacity] = useState(0.65);
  const inputPixels = useLabStore((s) => s.inputPixels);
  const width = 28;
  const height = 28;

  const base = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    const img = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      const v = Math.round((inputPixels[i] ?? 0) * 255);
      const o = i * 4;
      img.data[o] = v;
      img.data[o + 1] = v;
      img.data[o + 2] = v;
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }, [inputPixels]);

  const overlay = useMemo(() => renderSaliencyOverlay(saliencyData.inputGradient, width, height), [saliencyData.inputGradient]);

  return (
    <div className="mt-4 rounded-2xl border p-4" style={{ borderColor: "var(--bwd-border)", background: "var(--bg-card)" }}>
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>Saliency Map</h3>
      <p className="mt-1 text-xs" style={{ color: "var(--text-3)" }}>Brighter overlay regions had higher influence on loss.</p>
      <div className="mt-3 grid gap-4 md:grid-cols-[280px_1fr]">
        <div className="relative h-[280px] w-[280px] rounded-xl border" style={{ borderColor: "var(--glass-border)" }}>
          <img src={base} alt="Input" className="absolute inset-0 h-full w-full rounded-xl object-cover" style={{ imageRendering: "pixelated" }} />
          <img src={overlay} alt="Saliency" className="absolute inset-0 h-full w-full rounded-xl object-cover saliency-overlay" style={{ imageRendering: "pixelated", opacity }} />
          {saliencyData.topPixels.slice(0, 5).map((p, i) => (
            <div key={i} className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-orange-300" style={{ left: `${(p.col / width) * 100}%`, top: `${(p.row / height) * 100}%`, background: "rgba(251,146,60,0.45)" }} />
          ))}
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-xs" style={{ color: "var(--text-3)" }}>Overlay opacity {Math.round(opacity * 100)}%</label>
            <input type="range" min={0} max={100} value={opacity * 100} onChange={(e) => setOpacity(Number(e.target.value) / 100)} className="mt-1 w-full" />
          </div>
          <SaliencyLegend />
          <div className="space-y-1">
            {saliencyData.topPixels.slice(0, 8).map((p, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span style={{ color: "var(--bwd)" }}>{i + 1}.</span>
                <span className="font-mono" style={{ color: "var(--text-2)" }}>({p.row},{p.col})</span>
                <div className="h-1.5 flex-1 rounded" style={{ background: "var(--bg-panel)" }}>
                  <div className="h-full rounded" style={{ width: `${p.normalizedImportance * 100}%`, background: "var(--bwd)" }} />
                </div>
                <span className="font-mono" style={{ color: "var(--text-3)" }}>{(p.normalizedImportance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
