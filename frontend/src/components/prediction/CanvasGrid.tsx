import { useEffect, useRef } from "react";

type Props = { pixels: number[]; onChange: (p: number[]) => void; brushSize: number };

export default function CanvasGrid({ pixels, onChange, brushSize }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current!; const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, 280, 280);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const v = pixels[y * 28 + x] || 0; ctx.fillStyle = `rgba(6,182,212,${v})`; ctx.fillRect(x * 10, y * 10, 10, 10); }
  }, [pixels]);
  const paint = (evt: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = Math.floor((evt.clientX - rect.left) / 10), y = Math.floor((evt.clientY - rect.top) / 10);
    const next = [...pixels];
    for (let dy = -brushSize + 1; dy < brushSize; dy++) for (let dx = -brushSize + 1; dx < brushSize; dx++) {
      const px = x + dx, py = y + dy;
      if (px >= 0 && py >= 0 && px < 28 && py < 28) next[py * 28 + px] = Math.min(1, (next[py * 28 + px] || 0) + 0.25);
    }
    onChange(next);
  };
  return <canvas data-highlight="canvas" className="canvas-grid" ref={ref} width={280} height={280} onMouseMove={(e) => e.buttons === 1 && paint(e)} onClick={paint} />;
}
